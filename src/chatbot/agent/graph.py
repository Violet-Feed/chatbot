from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from chatbot.agent.prompts import AGENT_SYSTEM_TEMPLATE, AGENT_USER_TEMPLATE
from chatbot.agent.state import Agent, ConversationState
from chatbot.agent.tools import make_fetch_context_tool
from chatbot.agent.llm import DecisionInputs, LLMClient
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


class ConversationGraph:
    def __init__(
        self,
        chat: Any,
        base_tools: List[Any],
        im: Any,
        agent_svc: Any,
        memory_svc: Any,
        llm: LLMClient,
        redis_svc: Any,
        max_tool_calls: int = 1,
    ) -> None:
        self._agent_svc = agent_svc
        self._memory_svc = memory_svc
        self._llm = llm
        self._redis = redis_svc
        self._im = im
        self._max_tool_calls = max_tool_calls

        fetch_context_tool = make_fetch_context_tool(im_client=im)
        tools = base_tools + [fetch_context_tool]
        self._react_app = create_react_agent(chat, tools)
        self._app = self._build()

    def _build(self):
        g = StateGraph(ConversationState)
        g.add_node("load_context", self._load_context)
        g.add_node("decide", self._decide)
        g.add_node("generate", self._generate)
        g.add_node("send", self._send)

        g.set_entry_point("load_context")
        g.add_edge("load_context", "decide")
        g.add_conditional_edges(
            "decide",
            lambda s: "generate" if s["should_reply"] else END,
        )
        g.add_edge("generate", "send")
        g.add_edge("send", END)

        return g.compile()

    async def run(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        if not items:
            return
        last = items[-1]
        await self._app.ainvoke(
            ConversationState(
                con_short_id=con_short_id,
                con_id=str(last.get("con_id", "")),
                con_type=int(last.get("con_type", 0)),
                items=items,
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
        con_type = state["con_type"]
        con_id = state["con_id"]
        items = state["items"]

        if con_type == 2:
            agents = await self._load_group_agents(con_short_id)
        elif con_type == 4:
            agent = await self._load_private_agent(con_id)
            agents = [agent] if agent else []
        else:
            agents = []

        recent_text = _join_recent_text(items[-30:])
        memory_state = await self._memory_svc.get_summary_state(con_short_id)
        glossary = await self._memory_svc.get_relevant_glossary(con_short_id, recent_text, limit=8)

        return {"agents": agents, "memory_state": memory_state, "glossary": glossary}

    async def _decide(self, state: ConversationState) -> dict:
        agents = state["agents"]
        con_type = state["con_type"]

        if not agents:
            return {"should_reply": False, "primary_agent": None, "reply_reason": "no_agents"}

        if con_type == 4:
            return {"should_reply": True, "primary_agent": agents[0], "reply_reason": "AI私聊直接回复"}

        items = state["items"]
        memory_state = state["memory_state"]

        decision = await self._llm.decide_reply(
            DecisionInputs(
                agents_json=json_dumps(
                    [{"agent_id": a.agent_id, "name": a.name, "personality": a.personality} for a in agents]
                ),
                recent_messages_json=json_dumps(
                    [
                        {
                            "sender_type": int(it["sender_type"]),
                            "sender_id": int(it["sender_id"]),
                            "msg_type": int(it["msg_type"]),
                            "msg_content": str(it["msg_content"])[:300],
                        }
                        for it in items[-50:]
                    ]
                ),
                short_summary=memory_state.short_summary if memory_state else "",
                long_summary=memory_state.long_summary if memory_state else "",
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
        items = state["items"]
        memory_state = state["memory_state"]
        glossary = state["glossary"]
        con_short_id = state["con_short_id"]

        system_prompt = AGENT_SYSTEM_TEMPLATE.format(
            agent_name=agent.name,
            agent_id=agent.agent_id,
            personality=agent.personality or "",
            trigger_reason=state["reply_reason"] or "",
        )
        user_prompt = AGENT_USER_TEMPLATE.format(
            recent_messages=_format_messages(items[-30:]),
            short_summary=memory_state.short_summary if memory_state else "",
            long_summary=memory_state.long_summary if memory_state else "",
            glossary_json=json_dumps(
                [{"term": x.term, "meaning": x.meaning, "count": x.count} for x in glossary]
            ),
        )

        recursion_limit = self._max_tool_calls * 2 + 3
        try:
            result = await self._react_app.ainvoke(
                {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]},
                config={
                    "recursion_limit": recursion_limit,
                    "configurable": {"con_short_id": con_short_id},
                },
            )
        except Exception:
            logger.exception("generate failed con_short_id=%s", con_short_id)
            return {"reply_text": ""}

        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return {"reply_text": str(msg.content).strip()}

        return {"reply_text": ""}

    async def _send(self, state: ConversationState) -> dict:
        reply_text = state["reply_text"]
        if not reply_text:
            return {}

        agent = state["primary_agent"]
        items = state["items"]
        last = items[-1]

        task = {
            "task_id": uuid.uuid4().hex,
            "con_short_id": state["con_short_id"],
            "con_id": str(last["con_id"]),
            "con_type": int(last["con_type"]),
            "sender_id": agent.agent_id,
            "sender_type": 2,
            "send_ts_ms": now_ms(),
            "msg_type": 1,
            "msg_content": reply_text,
            "client_msg_id": 0,
            "extra": "",
            "retry_count": 0,
        }
        await self._redis.enqueue_send_task(json_dumps(task), task["send_ts_ms"])
        logger.info("sent reply con_short_id=%s agent=%s text=%r", state["con_short_id"], agent.agent_id, reply_text)
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
        infos = await self._agent_svc.get_agents_by_ids(agent_ids)
        return [
            Agent(
                agent_id=int(info.agent_id),
                name=str(info.agent_name or ""),
                personality=str(info.personality or ""),
            )
            for info in infos
        ]


# --- module-level helpers ---

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


def _format_messages(items: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for it in items:
        text = str(it["msg_content"]).replace("\n", " ").strip()
        if not text:
            continue
        sender_type = int(it["sender_type"])
        sender_id = int(it["sender_id"])
        if sender_type == 1:
            role = f"用户 {sender_id}"
        elif sender_type == 2:
            role = f"Agent {sender_id}"
        else:
            role = f"未知 {sender_id}"
        lines.append(f"[{role}]\n{text}")
    return "\n\n".join(lines)


def _join_recent_text(items: List[Dict[str, Any]]) -> str:
    return "\n".join(
        str(it.get("msg_content", "") or "").strip()
        for it in items
        if str(it.get("msg_content", "") or "").strip()
    )
