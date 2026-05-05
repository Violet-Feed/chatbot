from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from chatbot.agent.llm import DecisionInputs, GenerateInputs, GlossaryExtractInputs, GlossaryInferInputs, \
    LLMClient, ShortMemoryInputs, LongMemoryInputs
from chatbot.agent.state import Agent, ConversationState
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)

SHORT_MEMORY_MERGE_THRESHOLD = 20
SHORT_MEMORY_TIMEOUT_MS = 30 * 60 * 1000
LONG_MEMORY_BUF_THRESHOLD = 200
LONG_MEMORY_TIMEOUT_MS = 24 * 60 * 60 * 1000


class ConversationGraph:
    def __init__(
            self,
            im: Any,
            aigc: Any,
            memory_svc: Any,
            llm: LLMClient,
            redis_svc: Any,
    ) -> None:
        self._aigc = aigc
        self._memory_svc = memory_svc
        self._llm = llm
        self._redis = redis_svc
        self._im = im
        self._app = self._build()

    def _build(self):
        g = StateGraph(ConversationState)
        g.add_node("load_context", self._load_context)
        g.add_node("decide", self._decide)
        g.add_node("update_memories", self._update_memories)
        g.add_node("generate", self._generate)
        g.add_node("send", self._send)

        g.set_entry_point("load_context")
        g.add_conditional_edges("load_context", self._route_after_load)
        g.add_conditional_edges("decide", lambda s: "generate" if s["should_reply"] else END)
        g.add_edge("generate", "send")
        g.add_edge("send", END)
        g.add_edge("update_memories", END)

        return g.compile()

    @staticmethod
    def _route_after_load(state: ConversationState):
        if not state.get("agents"):
            return END
        return ["decide", "update_memories"]

    async def run(self, con_short_id: int, cur_items: List[Dict[str, Any]],
                  history_items: List[Dict[str, Any]]) -> None:
        if not cur_items:
            return
        logger.info("langgraph run con_short_id=%s", con_short_id)
        await self._app.ainvoke(
            ConversationState(
                con_short_id=con_short_id,
                con_id="",
                con_type=0,
                items=cur_items,
                history_items=history_items,
                con_name="",
                con_description="",
                agents=[],
                memory_state=None,
                glossary=[],
                should_reply=False,
                primary_agent=None,
                reply_reason="",
                reply_text="",
            )
        )

    # --- nodes ---

    async def _load_context(self, state: ConversationState) -> dict:
        con_short_id = state["con_short_id"]

        con_name = ""
        con_description = ""
        con_type = 0
        con_id = ""
        try:
            con_info = await self._im.get_conversation_info(user_id=0, con_short_id=con_short_id)
            cc = con_info.con_core_info
            con_name = str(getattr(cc, "name", "") or "")
            con_description = str(getattr(cc, "description", "") or "")
            con_type = int(getattr(cc, "con_type", 0) or 0)
            con_id = str(getattr(cc, "con_id", "") or "")
        except Exception:
            logger.exception("get_conversation_info failed con_short_id=%s", con_short_id)

        if con_type == 2:
            agents = await self._load_group_agents(con_short_id)
        elif con_type == 4:
            agent = await self._load_private_agent(con_id)
            agents = [agent] if agent else []
        else:
            agents = []

        if not agents:
            return {}

        combined = state["history_items"][-50:] + state["items"]
        recent_text = _join_recent_text(combined)
        memory_state = await self._memory_svc.get_summary_state(con_short_id)
        glossary = await self._memory_svc.get_relevant_glossary(con_short_id, recent_text, limit=8)

        return {
            "con_type": con_type,
            "con_id": con_id,
            "con_name": con_name,
            "con_description": con_description,
            "agents": agents,
            "memory_state": memory_state,
            "glossary": glossary,
        }

    async def _decide(self, state: ConversationState) -> dict:
        agents = state["agents"]
        con_type = state["con_type"]
        if not agents:
            return {"should_reply": False, "primary_agent": None, "reply_reason": "no_agents"}

        if con_type == 4:
            return {"should_reply": True, "primary_agent": agents[0], "reply_reason": "AI私聊直接回复"}

        cur_items = state["items"]
        history_items = state["history_items"][-50:]
        memory_state = state["memory_state"]

        decision = await self._llm.decide_reply(
            DecisionInputs(
                agents_json=json_dumps(
                    [{"agent_id": a.agent_id, "name": a.name, "personality": a.personality} for a in agents]
                ),
                current_messages_json=json_dumps(cur_items),
                history_messages_json=json_dumps(history_items),
                short_summary=memory_state.short_summary if memory_state else "",
                long_summary=memory_state.long_summary if memory_state else "",
                con_name=state.get("con_name", ""),
                con_description=state.get("con_description", ""),
                con_type_label=_con_type_label(state["con_type"]),
            )
        )
        logger.info("decide result=%s con_short_id=%s", decision, state["con_short_id"])

        if not decision.respond:
            return {"should_reply": False, "primary_agent": None, "reply_reason": getattr(decision, "reason", "")}

        agent = _find_agent(agents, int(getattr(decision, "primary_agent_id", 0) or 0))
        if agent is None:
            return {"should_reply": False, "primary_agent": None, "reply_reason": "agent_not_found"}

        return {
            "should_reply": True,
            "primary_agent": agent,
            "reply_reason": str(getattr(decision, "reason", "") or ""),
        }

    async def _generate(self, state: ConversationState) -> dict:
        agent = state["primary_agent"]
        cur_items = state["items"]
        memory_state = state["memory_state"]
        glossary = state["glossary"]

        inputs = GenerateInputs(
            agent_name=agent.name,
            agent_id=agent.agent_id,
            personality=agent.personality or "",
            trigger_reason=state["reply_reason"] or "",
            con_type_label=_con_type_label(state["con_type"]),
            con_name=state.get("con_name", ""),
            con_description=state.get("con_description", ""),
            current_messages_json=json_dumps(cur_items),
            history_messages_json=json_dumps(state["history_items"][-50:]),
            short_summary=memory_state.short_summary if memory_state else "",
            long_summary=memory_state.long_summary if memory_state else "",
            glossary_json=json_dumps(
                [{"term": x.term, "meaning": x.meaning, "count": x.count} for x in glossary]
            ),
            con_short_id=state["con_short_id"],
            min_con_index=_min_con_index(cur_items),
        )

        reply_text = await self._llm.generate_reply(inputs)
        logger.info("generated reply con_short_id=%s agent=%s text=%r", state["con_short_id"], agent.agent_id,
                    reply_text)
        return {"reply_text": reply_text}

    async def _send(self, state: ConversationState) -> dict:
        reply_text = state["reply_text"]
        if not reply_text:
            return {}

        agent = state["primary_agent"]

        lines = [line.strip() for line in reply_text.split("\n") if line.strip()]
        for i, line in enumerate(lines):
            send_ts_ms = now_ms()
            task = {
                "task_id": uuid.uuid4().hex,
                "con_short_id": state["con_short_id"],
                "con_id": state["con_id"],
                "con_type": state["con_type"],
                "sender_id": agent.agent_id,
                "sender_type": 2,
                "send_ts_ms": send_ts_ms,
                "msg_type": 1,
                "msg_content": line,
                "client_msg_id": 0,
                "extra": "",
                "retry_count": 0,
            }
            await self._redis.enqueue_send_task(json_dumps(task), send_ts_ms)
            logger.info(
                "sent reply con_short_id=%s agent=%s part=%d/%d text=%r",
                state["con_short_id"],
                agent.agent_id,
                i + 1,
                len(lines),
                line,
            )
        return {}

    # --- background memory updates ---

    async def _update_memories(self, state: ConversationState) -> dict:
        con_short_id = state["con_short_id"]
        con_name = state["con_name"]
        con_description = state["con_description"]
        cur_items = state["items"]
        all_items = state["history_items"] + state["items"]
        memory_state = state["memory_state"]
        try:
            merge_count = await self._redis.get_merge_count(con_short_id)
            all_buf_len = await self._redis.get_all_buf_length(con_short_id)
            now = now_ms()

            short_age = (now - memory_state.short_updated_at) if memory_state else now
            long_age = (now - memory_state.long_updated_at) if memory_state else now

            # 1. short memory: merge count >= 25 or timeout 30min
            if merge_count >= SHORT_MEMORY_MERGE_THRESHOLD or short_age >= SHORT_MEMORY_TIMEOUT_MS:
                recent_text = json_dumps(all_items[-50:])
                if recent_text.strip():
                    short_summary = memory_state.short_summary if memory_state else ""
                    new_short = await self._llm.update_short_memory(
                        ShortMemoryInputs(
                            old_short_summary=short_summary,
                            recent_messages=recent_text,
                        )
                    )
                    if new_short:
                        logger.info("updated short memory con_short_id=%s, new_short=%r", con_short_id, new_short)
                        await self._memory_svc.save_short_summary(con_short_id, new_short)
                await self._redis.reset_merge_count(con_short_id)

            # 2. long memory: all_buf >= 200 or timeout 24h
            if all_buf_len >= LONG_MEMORY_BUF_THRESHOLD or long_age >= LONG_MEMORY_TIMEOUT_MS:
                recent_text = json_dumps(all_items)
                if recent_text.strip():
                    long_summary = memory_state.long_summary if memory_state else ""
                    new_long = await self._llm.update_long_memory(
                        LongMemoryInputs(
                            old_long_summary=long_summary,
                            recent_messages=recent_text,
                            con_name=con_name,
                            con_description=con_description,
                        )
                    )
                    if new_long:
                        logger.info("updated long memory con_short_id=%s, new_long=%r", con_short_id, new_long)
                        await self._memory_svc.save_long_summary(con_short_id, new_long)
                await self._redis.trim_all_buf(con_short_id)

            # 3. glossary extract from current window items
            if cur_items:
                extract_text = json_dumps(cur_items)
                terms = await self._llm.extract_unknown_terms(
                    GlossaryExtractInputs(recent_messages=extract_text)
                )
                if terms:
                    logger.info("extracted terms=%r", terms)
                    await self._memory_svc.upsert_glossary_terms(con_short_id, terms)

                    need_meaning = await self._memory_svc.get_terms_need_meaning(con_short_id)
                    need_reinference = await self._memory_svc.get_terms_need_reinference(con_short_id)

                    all_term_strs: List[str] = list(set(need_meaning + [t.term for t in need_reinference]))
                    if all_term_strs:
                        context_text = json_dumps(all_items[-50:])
                        existing = need_reinference
                        inferred = await self._llm.infer_glossary_meanings(
                            GlossaryInferInputs(
                                terms_json=json_dumps(all_term_strs),
                                existing_meanings_json=json_dumps(
                                    [{"term": t.term, "meaning": t.meaning} for t in existing]
                                ),
                                recent_messages=context_text,
                                short_summary=memory_state.short_summary if memory_state else "",
                                long_summary=memory_state.long_summary if memory_state else "",
                            )
                        )
                        if inferred:
                            logger.info("updated glossary meanings con_short_id=%s, inferred=%r", con_short_id,
                                        inferred)
                            await self._memory_svc.save_glossary_meanings(con_short_id, inferred)

        except Exception:
            logger.exception("_update_memories failed con_short_id=%s", con_short_id)
        return {}

    # --- helpers ---

    async def _load_group_agents(self, con_short_id: int) -> List[Agent]:
        rows = await self._im.get_conversation_agents(con_short_id)
        agent_ids = [
            int(getattr(r, "agent_id", 0) or 0)
            for r in (rows or [])
            if int(getattr(r, "agent_id", 0) or 0) > 0
        ]
        return await self._load_agents_by_ids(agent_ids) if agent_ids else []

    async def _load_private_agent(self, con_id: str) -> Optional[Agent]:
        agent_id = _parse_private_agent_id(con_id)
        if agent_id <= 0:
            return None
        agents = await self._load_agents_by_ids([agent_id])
        return agents[0] if agents else None

    async def _load_agents_by_ids(self, agent_ids: List[int]) -> List[Agent]:
        infos = await self._aigc.get_agents_by_ids(agent_ids)
        return [
            Agent(
                agent_id=int(getattr(info, "agent_id", 0) or 0),
                name=str(getattr(info, "agent_name", "") or ""),
                personality=str(getattr(info, "personality", "") or ""),
            )
            for info in infos
        ]


# --- module-level helpers ---

def _con_type_label(con_type: int) -> str:
    if con_type == 2:
        return "群聊"
    elif con_type == 4:
        return "AI私聊"
    return "未知"


def _parse_private_agent_id(con_id: str) -> int:
    parts = con_id.split(":")
    if len(parts) != 3 or parts[0] != "ai":
        return 0
    return int(parts[2] or 0)


def _find_agent(agents: List[Agent], agent_id: int) -> Optional[Agent]:
    for a in agents:
        if a.agent_id == agent_id:
            return a
    return None


def _min_con_index(items: List[Dict[str, Any]]) -> int:
    indexes = [int(it.get("con_index", 0) or 0) for it in items if int(it.get("con_index", 0) or 0) > 0]
    return min(indexes) if indexes else 0


def _join_recent_text(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        str(it.get("msg_content", "") or "").strip()
        for it in items
        if str(it.get("msg_content", "") or "").strip()
    )
