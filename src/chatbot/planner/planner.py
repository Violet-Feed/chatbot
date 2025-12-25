from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from chatbot.dal.redis.service import RedisService
from chatbot.proto_gen import im_pb2
from chatbot.settings import Settings
from chatbot.utils.json import safe_obj_loads, json_dumps
from chatbot.utils.time import now_ms

from chatbot.memory.episodic import EpisodicEvent, extract_episodic_rule_based
from chatbot.memory.style import compute_style_features, dumps_features
from chatbot.memory.summary import RollingSummaryState, update_rolling_summary
from chatbot.planner.decision import AgentView, decide_0_2_agents
from chatbot.planner.llm import LLMClient, NoopLLMClient, ReplyInputs
from chatbot.planner.quality import enforce_short, split_human_like, too_similar

logger = logging.getLogger(__name__)


@dataclass
class AgentProfile:
    agent_id: int
    name: str
    personality: str
    description: str = ""
    avatar_uri: str = ""
    aliases: Sequence[str] = ()
    activeness: float = 0.5  # 0~1 越高越愿意说话


class Planner:
    """
    Planner：群聊多 Agent 的“决策与调度中心”
    - MQ 进入：on_message（轻量写窗口）
    - 窗口关闭：on_window_close（决策 0~2 条 bot 并写入延迟发送队列）
    - 可选：触发式滚动摘要 / 事件记忆 / 风格指纹的更新与落库

    重要变更：
    - agents：从 IM 服务 GetConversationAgents 获取（不查数据库）
    - memory：直接注入 MySQL services（RollingSummaryService / EpisodicMemoryService / StyleService）
    """

    def __init__(
        self,
        settings: Settings,
        redis_svc: RedisService,
        im: Any,  # IMClient（避免循环 import；duck typing）
        summary_svc: Any,  # RollingSummaryService
        episodic_svc: Any,  # EpisodicMemoryService
        style_svc: Any,  # StyleService
        llm: Optional[LLMClient] = None,
        enable_memory: bool = True,
    ) -> None:
        self.s = settings
        self.redis = redis_svc

        self.im = im
        self.summary_svc = summary_svc
        self.episodic_svc = episodic_svc
        self.style_svc = style_svc

        self.llm: LLMClient = llm or NoopLLMClient()
        self.enable_memory = enable_memory

    # -------------------------
    # MQ 入口：每条消息只做轻量操作
    # -------------------------
    async def on_message(self, evt: im_pb2.MessageEvent) -> None:
        m = evt.msg_body
        g = int(m.con_short_id)
        if g <= 0:
            return

        extra_obj = safe_obj_loads(m.extra)
        is_bot = isinstance(extra_obj, dict) and ("bot_meta" in extra_obj)

        # 人类消息：重置 bot 链深，并取消第二条（严格“人类打断”）
        if not is_bot:
            await self.redis.reset_bot_chain_depth(g)
            await self.redis.cancel_pending_second_for_group(g)

        # 热度判断：最小可用启发式（后续可换成“窗口消息数/间隔”）
        is_hot = self._is_hot()

        # upsert 窗口（5s + hot 扩展到 10s）
        res = await self.redis.upsert_window(
            g=g,
            con_index=int(m.con_index or evt.con_index or 0),
            is_hot=is_hot,
        )

        # 写入窗口 buffer（轻量）
        item: Dict[str, Any] = {
            "con_index": int(m.con_index or evt.con_index or 0),
            "msg_id": int(m.msg_id or 0),
            "sender_id": int(getattr(m, "sender_id", 0) or m.user_id or 0),  # 兼容你之后改 sender_id
            "is_bot": bool(is_bot),
            "msg_type": int(m.msg_type or 0),
            "msg_content": (m.msg_content or ""),
            "create_time": int(m.create_time or 0),
            "extra": extra_obj,
            # 发送时需要的会话信息（避免二次查）
            "con_id": (m.con_id or ""),
            "con_type": int(m.con_type or 2),
        }
        await self.redis.append_window_item(res.win_id, item)

        # 确保存在一个窗口关闭 watcher（deadline 可能被延长，因此循环检查）
        await self._ensure_window_close_task(g, res.win_id)

    def _is_hot(self) -> bool:
        # 保守：少量窗口会被延长，避免“永远固定 5s”造成体验僵硬
        return random.random() < 0.15

    # -------------------------
    # 窗口 watcher：可延长 deadline，因此循环检查
    # -------------------------
    async def _ensure_window_close_task(self, g: int, win_id: str) -> None:
        if not await self.redis.mark_close_scheduled(win_id):
            return
        asyncio.create_task(self._window_close_loop(g, win_id), name=f"winclose:{win_id}")

    async def _window_close_loop(self, g: int, win_id: str) -> None:
        while True:
            ddl = await self.redis.get_deadline_ms(g)
            if ddl is None:
                return

            now = now_ms()
            if ddl > now:
                await asyncio.sleep(min((ddl - now) / 1000.0, 2.0))
                continue

            # 分布式锁：避免多实例重复 close
            if not await self.redis.acquire_close_lock(win_id):
                return

            try:
                # 再次确认 deadline 未延长
                ddl2 = await self.redis.get_deadline_ms(g)
                if ddl2 is not None and ddl2 > now_ms():
                    await self.redis.release_close_lock(win_id)
                    continue

                await self.on_window_close(g, win_id)
                return
            except Exception:
                logger.exception("窗口关闭处理失败 g=%s win=%s", g, win_id)
                return

    # -------------------------
    # 窗口关闭：决策 0~2 条 bot 并写入延迟发送队列
    # -------------------------
    async def on_window_close(self, g: int, win_id: str) -> None:
        items = await self.redis.get_window_items(win_id)
        if not items:
            await self.redis.clear_window(g, win_id)
            return

        # 群级频控：拿不到 token 则本窗口不发言
        if not await self.redis.bucket_take(g, cost=1):
            await self._maybe_update_memory(g, items)
            await self.redis.clear_window(g, win_id)
            return

        agents = await self._load_agents_from_im(g)
        if not agents:
            await self._maybe_update_memory(g, items)
            await self.redis.clear_window(g, win_id)
            return

        # 冷却过滤
        available: List[AgentProfile] = []
        for a in agents:
            if await self.redis.is_agent_in_cooldown(a.agent_id):
                continue
            available.append(a)

        if not available:
            await self._maybe_update_memory(g, items)
            await self.redis.clear_window(g, win_id)
            return

        # ---- 纯决策：0~2 bot（不含 token/cooldown）----
        views = [
            AgentView(agent_id=a.agent_id, name=a.name, aliases=a.aliases, activeness=a.activeness)
            for a in available
        ]
        d = decide_0_2_agents(views, items, bot2bot_strict=True)
        if d.primary_id is None:
            await self._maybe_update_memory(g, items)
            await self.redis.clear_window(g, win_id)
            return

        primary = next((a for a in available if a.agent_id == d.primary_id), None)
        secondary = next((a for a in available if a.agent_id == d.secondary_id), None) if d.secondary_id else None
        if primary is None:
            await self._maybe_update_memory(g, items)
            await self.redis.clear_window(g, win_id)
            return

        # 主 bot：生成并入队
        await self._schedule_reply(
            g=g,
            win_id=win_id,
            agent=primary,
            rank=1,
            trigger_reason=d.reason1 or "STRONG",
            window_items=items,
        )
        await self.redis.set_bot_chain_depth(g, 1)

        # 第二 bot：严格门控 + 间隔
        if secondary is not None:
            allow_second = True

            # bot↔bot 严格：只在“最后一条是 bot 且决策原因是 BOT2BOT_STRICT”时消耗 token
            if d.reason2 == "BOT2BOT_STRICT":
                allow_second = await self.redis.take_bot2bot_token(g)

            if allow_second:
                await self._schedule_reply(
                    g=g,
                    win_id=win_id,
                    agent=secondary,
                    rank=2,
                    trigger_reason=d.reason2 or "STRONG",
                    window_items=items,
                )

        # 可选：记忆更新（摘要/事件/风格）
        await self._maybe_update_memory(g, items)

        await self.redis.clear_window(g, win_id)

    async def _load_agents_from_im(self, con_short_id: int) -> List[AgentProfile]:
        """
        从 IM 获取群内 agents：
        GetConversationAgentsResponse.agents[].agent_info (AgentInfo)
        """
        try:
            con_agents = await self.im.get_conversation_agents(con_short_id)
        except Exception:
            logger.exception("get_conversation_agents failed con=%s", con_short_id)
            return []

        out: List[AgentProfile] = []
        for ca in con_agents or []:
            ai = getattr(ca, "agent_info", None)
            if ai is None:
                continue

            agent_id = int(getattr(ai, "agent_id", 0) or 0)
            if agent_id <= 0:
                continue

            # aliases/activeness 允许塞在 extra（ConversationAgentInfo.extra 优先，其次 AgentInfo.extra）
            extra1 = safe_obj_loads(getattr(ca, "extra", "")) or {}
            extra2 = safe_obj_loads(getattr(ai, "extra", "")) or {}
            extra = extra1 if isinstance(extra1, dict) and extra1 else (extra2 if isinstance(extra2, dict) else {})

            aliases = extra.get("aliases") or []
            activeness = extra.get("activeness", 0.5)
            try:
                activeness_f = float(activeness)
            except Exception:
                activeness_f = 0.5
            activeness_f = max(0.0, min(1.0, activeness_f))

            out.append(
                AgentProfile(
                    agent_id=agent_id,
                    name=str(getattr(ai, "agent_name", "") or ""),
                    personality=str(getattr(ai, "personality", "") or ""),
                    description=str(getattr(ai, "description", "") or ""),
                    avatar_uri=str(getattr(ai, "avatar_uri", "") or ""),
                    aliases=tuple(str(x) for x in aliases if x),
                    activeness=activeness_f,
                )
            )
        return out

    # -------------------------
    # 生成回复（LLM）并写入延迟发送队列
    # -------------------------
    async def _schedule_reply(
        self,
        g: int,
        win_id: str,
        agent: AgentProfile,
        rank: int,
        trigger_reason: str,
        window_items: List[Dict[str, Any]],
    ) -> None:
        # 未接 LLM：保持克制，不发（系统稳定优先）
        if isinstance(self.llm, NoopLLMClient):
            return

        strict = rank == 2

        recent = self._format_recent(window_items[-30:])

        # memory load：失败也不阻断主链路
        rolling_summary = ""
        episodic: List[Dict[str, Any]] = []
        style: Dict[str, Any] = {}
        try:
            rolling_summary, _last_idx = await self.summary_svc.load(g)
        except Exception:
            logger.exception("load rolling summary failed con=%s", g)

        try:
            episodic = await self.episodic_svc.load_active(g, now_ms())
        except Exception:
            logger.exception("load episodic memory failed con=%s", g)

        try:
            style = await self.style_svc.load(g)
        except Exception:
            logger.exception("load style fingerprint failed con=%s", g)

        inputs = ReplyInputs(
            personality=agent.personality or "",
            intent="补一句" if strict else "自然回应并参与",
            trigger_reason=trigger_reason,
            recent_messages=recent,
            rolling_summary=rolling_summary or "",
            episodic_memory=json_dumps(episodic) if episodic else "[]",
            style_hint=json_dumps(style) if style else "",
            group_rules="",
        )

        raw = (await self.llm.generate_reply(inputs)).strip()
        if not raw:
            return

        # 质量闸门：长度 + 复读
        text = enforce_short(raw, strict=strict)
        if not text:
            return

        recent_bot_texts = [str(it.get("msg_content", "") or "") for it in window_items if it.get("is_bot")]
        if too_similar(text, recent_bot_texts, threshold=0.75):
            return

        # 拟人分句：多条短消息
        segments = split_human_like(text, max_segments=3)
        if not segments:
            return

        # 模拟输入延迟 + 第二条严格间隔
        base_delay = min(6000, 800 + len(segments[0]) * 35 + random.randint(0, 900))
        send_ts = now_ms() + base_delay
        if strict:
            gap = random.randint(self.s.SECOND_GAP_MIN_MS, self.s.SECOND_GAP_MAX_MS)
            send_ts = max(send_ts, now_ms() + gap)

        # 会话信息：从窗口最后一条带过来
        last = window_items[-1]
        con_id = str(last.get("con_id", "") or "")
        con_type = int(last.get("con_type", 2) or 2)

        first_task_id: Optional[str] = None

        for idx, seg in enumerate(segments):
            task_id = uuid.uuid4().hex
            if idx == 0 and strict:
                first_task_id = task_id

            extra_obj = {
                "bot_meta": {
                    "trace_id": uuid.uuid4().hex,
                    "window_id": win_id,
                    "rank": rank,
                    "strict": strict,
                    "trigger": trigger_reason,
                    "agent_id": agent.agent_id,
                }
            }

            task = {
                "task_id": task_id,
                "con_short_id": g,
                "con_id": con_id,
                "con_type": con_type,
                "agent_id": agent.agent_id,
                "rank": rank,
                "send_ts_ms": int(send_ts),
                "msg_type": 1,
                "msg_content": seg,
                "client_msg_id": uuid.uuid4().int & 0x7FFF_FFFF_FFFF_FFFF,
                "extra": json_dumps(extra_obj),
                "retry_count": 0,
            }

            await self.redis.enqueue_send_task(json_dumps(task), int(send_ts))

            # 分段之间加一点“打字停顿”
            send_ts += random.randint(1200, 2500)

        # 冷却：避免打扰（按 activeness 偏置）
        cooldown_ms = self._cooldown_ms(agent=agent, strict=strict)
        await self.redis.set_agent_cooldown_until(
            agent_id=agent.agent_id,
            until_ms=int(send_ts + cooldown_ms),
            ttl_ms=int(cooldown_ms + 30_000),
        )

        # 第二条允许被人类打断取消：只记录“第二 bot 的第一段”
        if first_task_id:
            await self.redis.set_pending_second(win_id, first_task_id)

    # -------------------------
    # 记忆更新（可选）
    # -------------------------
    async def _maybe_update_memory(self, g: int, window_items: List[Dict[str, Any]]) -> None:
        if not self.enable_memory:
            return

        try:
            # 1) 风格指纹：轻量统计（建议总是更新）
            features = compute_style_features(window_items)
            if features:
                try:
                    await self.style_svc.upsert(g, dumps_features(features))
                except Exception:
                    logger.exception("save style fingerprint failed con=%s", g)

            # 2) 事件记忆：规则抽取为主；若 llm 提供 extract_episodic/summarize 再升级
            events: List[EpisodicEvent] = []

            # 优先：如果 llm 有 extract_episodic 方法就用，否则走规则
            extract_fn = getattr(self.llm, "extract_episodic", None)
            if (not isinstance(self.llm, NoopLLMClient)) and callable(extract_fn):
                try:
                    messages_text = self._format_recent(window_items[-120:])
                    raw = await extract_fn(messages_text)  # type: ignore[misc]
                    if isinstance(raw, list):
                        for d in raw[:8]:
                            if not isinstance(d, dict):
                                continue
                            content = str(d.get("content", "") or "").strip()
                            if not content:
                                continue
                            events.append(
                                EpisodicEvent(
                                    event_type=str(d.get("type", "generic") or "generic"),
                                    content=content,
                                    importance=float(d.get("importance", 0.5) or 0.5),
                                    ttl_ms=int(d.get("ttl_ms", 0) or 0),
                                )
                            )
                except Exception:
                    events = []

            if not events:
                events = extract_episodic_rule_based(window_items)

            if events:
                # 兼容：如果你没实现 add_many，就逐条 add
                add_many = getattr(self.episodic_svc, "add_many", None)
                if callable(add_many):
                    payload = [
                        {
                            "type": e.event_type,
                            "content": e.content,
                            "importance": e.importance,
                            "ttl_ms": e.ttl_ms,
                        }
                        for e in events
                    ]
                    await add_many(g, payload)  # type: ignore[misc]
                else:
                    for e in events[:8]:
                        await self.episodic_svc.add(
                            con_short_id=g,
                            event_type=e.event_type,
                            content=e.content,
                            importance=e.importance,
                            ttl_ms=e.ttl_ms,
                        )

            # 3) 滚动摘要：触发式（新增消息到一定量再更新）
            try:
                cur_summary, last_idx = await self.summary_svc.load(g)

                summarize_fn = getattr(self.llm, "summarize", None)

                async def _llm_summarize(cur: str, new_msgs: str) -> str:
                    if callable(summarize_fn):
                        return await summarize_fn(cur, new_msgs)  # type: ignore[misc]
                    return cur

                state0 = RollingSummaryState(
                    summary=cur_summary or "",
                    last_con_index=int(last_idx or 0),
                    updated_at_ms=now_ms(),
                )
                state1 = await update_rolling_summary(
                    window_items=window_items,
                    current_state=state0,
                    llm_summarize=None if (isinstance(self.llm, NoopLLMClient) or not callable(summarize_fn)) else _llm_summarize,
                    min_new_msgs=20,
                )
                if (state1.summary != state0.summary) or (state1.last_con_index != state0.last_con_index):
                    await self.summary_svc.upsert(g, state1.summary, state1.last_con_index)

            except Exception:
                logger.exception("rolling summary update failed con=%s", g)

        except Exception:
            # 记忆更新不影响主链路（绝不让它打断发言流程）
            logger.exception("memory update failed con=%s", g)

    # -------------------------
    # 工具函数
    # -------------------------
    def _format_recent(self, items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for it in items:
            who = "BOT" if it.get("is_bot") else "USER"
            sid = it.get("sender_id", 0)
            txt = (it.get("msg_content") or "").replace("\n", " ").strip()
            if txt:
                lines.append(f"[{who}:{sid}] {txt}")
        return "\n".join(lines)

    def _cooldown_ms(self, agent: AgentProfile, strict: bool) -> int:
        base = random.randint(40_000, 120_000)
        act = float(agent.activeness or 0.5)
        act = max(0.0, min(1.0, act))
        bias = int((1.0 - act) * 40_000)
        if strict:
            base += 20_000
        return max(20_000, base + bias)
