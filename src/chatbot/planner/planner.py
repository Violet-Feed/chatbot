from __future__ import annotations

import asyncio
import logging
import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from chatbot.dal.redis.service import RedisService
from chatbot.proto_gen import im_pb2
from chatbot.settings import Settings
from chatbot.utils.json import safe_obj_loads, json_dumps
from chatbot.utils.time import now_ms

from chatbot.memory.episodic import EpisodicEvent, extract_episodic_rule_based
from chatbot.planner.llm import (
    DecisionInputs,
    DecisionOutput,
    LLMClient,
    NoopLLMClient,
    ReplyInputs,
    SearchDecisionInputs,
    SearchDecisionOutput,
    UnknownTermsInputs,
    UnknownTermsOutput,
    TermMeaningInputs,
    TermMeaningOutput,
    ClarifyInputs,
    StyleLearnInputs,
    StyleLearnOutput,
)
from chatbot.planner.quality import enforce_short, split_human_like, too_similar
from chatbot.planner.triggers import mentioned_agent
from chatbot.utils.web_search import search_jina

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


class WindowGraphState(TypedDict, total=False):
    g: int
    win_id: str
    items: List[Dict[str, Any]]
    agents: List[AgentProfile]
    chain_depth: int
    token_taken: bool
    rate_limited: bool
    decision: DecisionOutput
    scheduled_primary: bool
    scheduled_secondary: bool
    web_context: str
    glossary_updates: List[Dict[str, Any]]
    override_reply_text: str


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

        self._graph = self._build_graph()

    # -------------------------
    # MQ 入口：每条消息只做轻量操作
    # -------------------------
    async def on_message(self, evt: im_pb2.MessageEvent) -> None:
        m = evt.msg_body
        g = int(m.con_short_id)
        if g <= 0:
            return

        extra_obj = safe_obj_loads(m.extra)
        is_bot = int(getattr(m, "sender_type", 0) or 0) == 3

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
            "sender_id": int(getattr(m, "sender_id", 0) or 0),
            "sender_type": int(getattr(m, "sender_type", 0) or 0),
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
        try:
            await self._graph.ainvoke({"g": g, "win_id": win_id, "items": items})
        except Exception:
            logger.exception("langgraph window flow failed con=%s win=%s", g, win_id)
        finally:
            await self.redis.clear_window(g, win_id)

    # -------------------------
    # LangGraph orchestration
    # -------------------------
    def _build_graph(self):
        from langgraph.graph import StateGraph, END

        graph = StateGraph(WindowGraphState)
        graph.add_node("load_context", self._lg_load_context)
        graph.add_node("rate_limit", self._lg_rate_limit)
        graph.add_node("llm_decide", self._lg_llm_decide)
        graph.add_node("resolve_search", self._lg_resolve_search)
        graph.add_node("schedule_primary", self._lg_schedule_primary)
        graph.add_node("schedule_secondary", self._lg_schedule_secondary)
        graph.add_node("finalize", self._lg_finalize)
        graph.add_node("memory_update", self._lg_memory_update)

        graph.set_entry_point("load_context")
        graph.add_edge("load_context", "rate_limit")
        graph.add_edge("rate_limit", "llm_decide")
        graph.add_edge("llm_decide", "resolve_search")
        graph.add_edge("resolve_search", "schedule_primary")
        graph.add_edge("schedule_primary", "schedule_secondary")
        graph.add_edge("schedule_secondary", "finalize")
        graph.add_edge("finalize", "memory_update")
        graph.add_edge("memory_update", END)
        return graph.compile()

    async def _lg_load_context(self, state: WindowGraphState) -> WindowGraphState:
        g = int(state["g"])
        agents = await self._load_agents_from_im(g)

        available: List[AgentProfile] = []
        for a in agents:
            if await self.redis.is_agent_in_cooldown(a.agent_id):
                continue
            available.append(a)

        chain_depth = await self.redis.get_bot_chain_depth(g)
        return {"agents": available, "chain_depth": int(chain_depth)}

    async def _lg_rate_limit(self, state: WindowGraphState) -> WindowGraphState:
        g = int(state["g"])
        agents = state.get("agents") or []
        if not agents:
            return {"rate_limited": True, "token_taken": False}

        ok = await self.redis.bucket_take(g, cost=1)
        return {"rate_limited": not ok, "token_taken": bool(ok)}

    async def _lg_llm_decide(self, state: WindowGraphState) -> WindowGraphState:
        items = state.get("items") or []
        agents = state.get("agents") or []
        chain_depth = int(state.get("chain_depth") or 0)

        if not items or not agents:
            return {"decision": DecisionOutput(False, None, None, "no_items_or_agents")}
        if state.get("rate_limited"):
            return {"decision": DecisionOutput(False, None, None, "rate_limited")}
        if chain_depth >= 2:
            return {"decision": DecisionOutput(False, None, None, "bot_chain_depth")}
        if isinstance(self.llm, NoopLLMClient):
            return {"decision": DecisionOutput(False, None, None, "noop_llm")}

        agents_payload = []
        for a in agents:
            agents_payload.append(
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "aliases": list(a.aliases),
                    "personality": a.personality,
                    "description": a.description,
                    "activeness": a.activeness,
                }
            )

        msgs_payload = self._decision_messages_payload(items)
        inputs = DecisionInputs(
            agents_json=json_dumps(agents_payload),
            recent_messages_json=msgs_payload["recent_messages_json"],
            last_message_json=msgs_payload["last_message_json"],
            bot_chain_depth=chain_depth,
            group_rules="",
        )

        decision = await self.llm.decide_reply(inputs)
        decision = self._sanitize_decision(decision, agents, items)
        return {"decision": decision}

    async def _lg_resolve_search(self, state: WindowGraphState) -> WindowGraphState:
        decision = state.get("decision")
        if not decision or not decision.respond:
            return {}
        if isinstance(self.llm, NoopLLMClient):
            return {}

        items = state.get("items") or []
        recent_text = self._format_recent(items[-30:])
        web_contexts: List[str] = []
        glossary_updates: List[Dict[str, Any]] = []

        search_decision: SearchDecisionOutput = await self.llm.decide_search(
            SearchDecisionInputs(recent_messages=recent_text)
        )
        if search_decision.need_search and search_decision.query:
            res = await self._web_search(search_decision.query)
            if res:
                web_contexts.append(f"[search:{search_decision.query}]\n{res}")

        known_terms = await self._load_known_terms(int(state["g"]))
        unknown_terms: UnknownTermsOutput = await self.llm.extract_unknown_terms(
            UnknownTermsInputs(recent_messages=recent_text, known_terms_json=json_dumps(known_terms))
        )

        unresolved: List[str] = []
        for term in unknown_terms.terms[:2]:
            web = await self._web_search(term) or ""
            if web:
                web_contexts.append(f"[term:{term}]\n{web}")

            meaning: TermMeaningOutput = await self.llm.infer_term_meaning(
                TermMeaningInputs(term=term, recent_messages=recent_text, web_context=web)
            )
            if meaning.meaning and meaning.confidence >= 0.4:
                glossary_updates.append(
                    {
                        "term": term,
                        "meaning": meaning.meaning,
                        "source": "search" if web else "context",
                    }
                )
            else:
                unresolved.append(term)

        override_reply_text = ""
        if unresolved:
            question = await self.llm.generate_clarify(
                ClarifyInputs(terms="，".join(unresolved), recent_messages=recent_text)
            )
            override_reply_text = question.strip()

        web_context = "\n\n".join(web_contexts)
        if len(web_context) > 6000:
            web_context = web_context[:6000] + "..."

        return {
            "web_context": web_context,
            "glossary_updates": glossary_updates,
            "override_reply_text": override_reply_text,
        }

    async def _lg_schedule_primary(self, state: WindowGraphState) -> WindowGraphState:
        decision = state.get("decision")
        agents = state.get("agents") or []
        items = state.get("items") or []
        if not decision or not decision.respond or decision.primary_agent_id is None:
            return {"scheduled_primary": False}

        primary = next((a for a in agents if a.agent_id == decision.primary_agent_id), None)
        if not primary:
            return {"scheduled_primary": False}

        scheduled = await self._schedule_reply(
            g=int(state["g"]),
            win_id=str(state["win_id"]),
            agent=primary,
            rank=1,
            trigger_reason=decision.reason or "LLM",
            window_items=items,
            web_context=str(state.get("web_context") or ""),
            glossary_updates=state.get("glossary_updates") or [],
            forced_text=str(state.get("override_reply_text") or "").strip() or None,
        )
        return {"scheduled_primary": bool(scheduled)}

    async def _lg_schedule_secondary(self, state: WindowGraphState) -> WindowGraphState:
        decision = state.get("decision")
        if not decision or decision.secondary_agent_id is None:
            return {"scheduled_secondary": False}

        if not state.get("scheduled_primary"):
            return {"scheduled_secondary": False}

        agents = state.get("agents") or []
        items = state.get("items") or []
        secondary = next((a for a in agents if a.agent_id == decision.secondary_agent_id), None)
        if not secondary:
            return {"scheduled_secondary": False}

        last = items[-1] if items else {}
        if bool(last.get("is_bot")):
            if not await self.redis.take_bot2bot_token(int(state["g"])):
                return {"scheduled_secondary": False}

        scheduled = await self._schedule_reply(
            g=int(state["g"]),
            win_id=str(state["win_id"]),
            agent=secondary,
            rank=2,
            trigger_reason=decision.reason or "LLM_SECOND",
            window_items=items,
            web_context=str(state.get("web_context") or ""),
            glossary_updates=state.get("glossary_updates") or [],
        )
        return {"scheduled_secondary": bool(scheduled)}

    async def _lg_finalize(self, state: WindowGraphState) -> WindowGraphState:
        g = int(state["g"])
        if state.get("token_taken") and not state.get("scheduled_primary"):
            await self.redis.bucket_refund(g, cost=1)

        if state.get("scheduled_primary"):
            depth = 2 if state.get("scheduled_secondary") else 1
            await self.redis.set_bot_chain_depth(g, depth)

        return {}

    async def _lg_memory_update(self, state: WindowGraphState) -> WindowGraphState:
        items = state.get("items") or []
        await self._maybe_update_memory(int(state["g"]), items)
        if state.get("glossary_updates"):
            try:
                await self.style_svc.merge_update(
                    int(state["g"]),
                    {"glossary": state.get("glossary_updates")},
                )
            except Exception:
                logger.exception("save glossary update failed con=%s", state.get("g"))
        return {}

    def _decision_messages_payload(self, items: List[Dict[str, Any]]) -> Dict[str, str]:
        payload: List[Dict[str, Any]] = []
        for it in items[-50:]:
            mentions = (it.get("extra") or {}).get("mentions") or []
            payload.append(
                {
                    "sender_id": int(it.get("sender_id", 0) or 0),
                    "is_bot": bool(it.get("is_bot", False)),
                    "msg_type": int(it.get("msg_type", 0) or 0),
                    "msg_content": str(it.get("msg_content", "") or "")[:300],
                    "mentions": mentions,
                }
            )
        last = payload[-1] if payload else {}
        return {
            "recent_messages_json": json_dumps(payload),
            "last_message_json": json_dumps(last),
        }

    def _sanitize_decision(
        self,
        decision: DecisionOutput,
        agents: List[AgentProfile],
        items: List[Dict[str, Any]],
    ) -> DecisionOutput:
        if not decision.respond or decision.primary_agent_id is None:
            return DecisionOutput(False, None, None, decision.reason)

        ids = {a.agent_id for a in agents}
        if decision.primary_agent_id not in ids:
            return DecisionOutput(False, None, None, "invalid_primary")

        secondary = decision.secondary_agent_id
        if secondary is not None:
            if secondary == decision.primary_agent_id or secondary not in ids:
                secondary = None

        last = items[-1] if items else {}
        last_mentions = (last.get("extra") or {}).get("mentions") or []
        if bool(last.get("is_bot")):
            if not mentioned_agent(decision.primary_agent_id, last_mentions):
                return DecisionOutput(False, None, None, "bot_last_no_mention")
            if secondary is not None and (not mentioned_agent(secondary, last_mentions)):
                secondary = None

        return DecisionOutput(True, decision.primary_agent_id, secondary, decision.reason)

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
        web_context: str = "",
        glossary_updates: Optional[List[Dict[str, Any]]] = None,
        forced_text: Optional[str] = None,
    ) -> bool:
        try:
            # 未接 LLM：保持克制，不发（系统稳定优先）
            if isinstance(self.llm, NoopLLMClient):
                return False

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

            glossary_json = self._merge_glossary_for_prompt(style, glossary_updates)

            if forced_text:
                raw = forced_text.strip()
            else:
                inputs = ReplyInputs(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    agent_personality=agent.personality or "",
                    agent_description=agent.description or "",
                    rank=rank,
                    trigger_reason=trigger_reason,
                    intent="补一句" if strict else "自然回应并参与",
                    recent_messages=recent,
                    rolling_summary=rolling_summary or "",
                    episodic_memory_json=json_dumps(episodic) if episodic else "[]",
                    style_hint=json_dumps(style) if style else "",
                    glossary_json=glossary_json,
                    web_context=web_context or "",
                    group_rules="",
                )

                raw = (await self.llm.generate_reply(inputs)).strip()
                if not raw:
                    return False

            # 质量闸门：长度 + 复读
            text = enforce_short(raw, strict=strict)
            if not text:
                return False

            recent_bot_texts = [str(it.get("msg_content", "") or "") for it in window_items if it.get("is_bot")]
            if too_similar(text, recent_bot_texts, threshold=0.75):
                return False

            # 拟人分句：多条短消息
            segments = split_human_like(text, max_segments=3)
            if not segments:
                return False

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
                    "sender_id": agent.agent_id,
                    "sender_type": 3,
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

            return True
        except Exception:
            logger.exception("schedule reply failed con=%s agent=%s", g, agent.agent_id)
            return False

    # -------------------------
    # 记忆更新（可选）
    # -------------------------
    async def _maybe_update_memory(self, g: int, window_items: List[Dict[str, Any]]) -> None:
        if not self.enable_memory:
            return

        try:
            # 1) 风格指纹：LLM 学习语言风格与热词（每窗口）
            try:
                style_update = await self._learn_style(window_items)
                if style_update:
                    await self.style_svc.merge_update(g, style_update)
            except Exception:
                logger.exception("save style fingerprint failed con=%s", g)

            # 2) 事件记忆：优先 LLM，失败则回退规则抽取
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

            # 3) 滚动摘要：每窗口都更新（LLM 可用时）
            try:
                cur_summary, last_idx = await self.summary_svc.load(g)
                summarize_fn = getattr(self.llm, "summarize", None)
                new_msgs_text = self._format_recent(window_items[-200:])
                new_summary = cur_summary or ""
                if (not isinstance(self.llm, NoopLLMClient)) and callable(summarize_fn):
                    new_summary = await summarize_fn(cur_summary or "", new_msgs_text)  # type: ignore[misc]

                new_last_idx = max(int(it.get("con_index", 0) or 0) for it in window_items)
                if (new_summary != cur_summary) or (int(last_idx or 0) != new_last_idx):
                    await self.summary_svc.upsert(g, new_summary, new_last_idx)
            except Exception:
                logger.exception("rolling summary update failed con=%s", g)

        except Exception:
            # 记忆更新不影响主链路（绝不让它打断发言流程）
            logger.exception("memory update failed con=%s", g)

    async def _learn_style(self, window_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(self.llm, NoopLLMClient):
            return {}

        text = self._format_style_messages(window_items)
        if not text:
            return {}

        out: StyleLearnOutput = await self.llm.learn_style(StyleLearnInputs(recent_messages=text))
        style_rules = [s for s in out.style_rules if s]
        hotwords = [s for s in out.hotwords if s]
        if not style_rules and not hotwords:
            return {}

        return {"style_rules": style_rules, "hotwords": hotwords}

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

    def _format_style_messages(self, items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        idx = 1
        for it in items:
            if it.get("is_bot"):
                continue
            txt = (it.get("msg_content") or "").replace("\n", " ").strip()
            if not txt:
                continue
            lines.append(f"[{idx}] {txt}")
            idx += 1
            if idx > 120:
                break
        return "\n".join(lines)

    def _merge_glossary_for_prompt(
        self,
        style: Dict[str, Any],
        glossary_updates: Optional[List[Dict[str, Any]]],
    ) -> str:
        base = style.get("glossary") if isinstance(style, dict) else []
        base_list = base if isinstance(base, list) else []
        updates = glossary_updates or []

        merged: List[Dict[str, Any]] = []
        seen = set()
        for it in updates + base_list:
            if not isinstance(it, dict):
                continue
            term = str(it.get("term", "") or "").strip()
            meaning = str(it.get("meaning", "") or "").strip()
            if not term or not meaning:
                continue
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append({"term": term, "meaning": meaning})
            if len(merged) >= 20:
                break
        return json_dumps(merged)

    async def _web_search(self, query: str) -> str:
        res = await search_jina(
            query=query,
            base_url=self.s.JINA_SEARCH_BASE_URL,
            api_key=self.s.JINA_API_KEY,
            timeout_sec=int(self.s.JINA_SEARCH_TIMEOUT_SEC),
        )
        return res or ""

    async def _load_known_terms(self, con_short_id: int) -> List[str]:
        try:
            style = await self.style_svc.load(con_short_id)
        except Exception:
            return []

        terms: List[str] = []
        if isinstance(style, dict):
            hotwords = style.get("hotwords") or []
            glossary = style.get("glossary") or []
            if isinstance(hotwords, list):
                for it in hotwords:
                    s = str(it or "").strip()
                    if s:
                        terms.append(s)
            if isinstance(glossary, list):
                for it in glossary:
                    if not isinstance(it, dict):
                        continue
                    s = str(it.get("term", "") or "").strip()
                    if s:
                        terms.append(s)
        return terms

    def _cooldown_ms(self, agent: AgentProfile, strict: bool) -> int:
        base = random.randint(40_000, 120_000)
        act = float(agent.activeness or 0.5)
        act = max(0.0, min(1.0, act))
        bias = int((1.0 - act) * 40_000)
        if strict:
            base += 20_000
        return max(20_000, base + bias)
