from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List

from chatbot.proto_gen import im_pb2
from chatbot.planner.llm import (
    DecisionInputs,
    ReplyInputs,
    ShortMemoryInputs,
    LongMemoryInputs,
    GlossaryExtractInputs,
    GlossaryInferInputs,
)
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Agent:
    agent_id: int
    name: str
    personality: str


class Planner:
    SHORT_MSG_TRIGGER = 20
    SHORT_TIME_TRIGGER_MS = 30 * 60 * 1000

    LONG_MSG_TRIGGER = 100
    LONG_TIME_TRIGGER_MS = 24 * 60 * 60 * 1000

    def __init__(
        self,
        redis_svc: Any,
        im: Any,
        llm: Any,
        agent_svc: Any,
        memory_svc: Any,
        window_sec: int = 5,
    ) -> None:
        self.redis = redis_svc
        self.im = im
        self.llm = llm
        self.agent_svc = agent_svc
        self.memory_svc = memory_svc
        self.window_sec = window_sec

    async def on_message(self, evt: im_pb2.MessageEvent) -> None:
        msg = evt.msg_body
        con_short_id = int(msg.con_short_id or 0)
        con_type = int(msg.con_type or 0)
        sender_type = int(msg.sender_type or 0)
        msg_type = int(msg.msg_type or 0)

        if con_short_id <= 0:
            return

        if con_type not in (2, 4):
            return

        if msg_type != 1:
            return

        item = {
            "sender_id": int(msg.sender_id or 0),
            "sender_type": sender_type,
            "msg_type": msg_type,
            "msg_content": str(msg.msg_content or ""),
            "create_time": int(msg.create_time or 0),
            "con_id": str(msg.con_id or ""),
            "con_type": con_type,
        }

        await self.redis.upsert_window(con_short_id, self.window_sec)
        await self.redis.append_window_item(con_short_id, item)

        if con_type == 4 and sender_type == 2:
            return

        started = await self.redis.mark_close_scheduled(con_short_id)
        if started:
            asyncio.create_task(self.wait_window(con_short_id))

    async def wait_window(self, con_short_id: int) -> None:
        while True:
            deadline_ms = await self.redis.get_deadline_ms(con_short_id)
            if deadline_ms is None:
                return

            delay = (deadline_ms - now_ms()) / 1000.0
            if delay > 0:
                await asyncio.sleep(delay)
                continue

            break

        await self.close_window(con_short_id)

    async def close_window(self, con_short_id: int) -> None:
        items = await self.redis.get_window_items(con_short_id)
        await self.redis.clear_window(con_short_id)

        if not items:
            return

        asyncio.create_task(self.update_memories_safe(con_short_id, items))

        last = items[-1]
        con_type = int(last.get("con_type", 0) or 0)
        con_id = str(last.get("con_id", "") or "")

        if con_type == 2:
            agents = await self.load_group_agents(con_short_id)
            if not agents:
                return

            decision = await self.decide_reply(con_short_id, agents, items)
            logger.info("decide_reply: %s for %s", decision, con_short_id)
            if not getattr(decision, "respond", False):
                return

            agent = self.find_agent(
                agents,
                int(getattr(decision, "primary_agent_id", 0) or 0),
            )
            if agent is None:
                return

            reason = str(getattr(decision, "reason", "") or "LLM")
            text = await self.generate_reply(con_short_id, agent, items, reason)
            if not text:
                return

            logger.info("generate_reply: %s for %s", text, con_short_id)
            await self.send_reply(con_short_id, agent, items, text)
            return

        if con_type == 4:
            agent = await self.load_private_agent(con_id)
            if agent is None:
                return

            text = await self.generate_reply(con_short_id, agent, items, "AI私聊直接回复")
            if not text:
                return

            logger.info("generate_reply: %s for %s", text, con_short_id)
            await self.send_reply(con_short_id, agent, items, text)

    async def load_group_agents(self, con_short_id: int) -> List[Agent]:
        rows = await self.im.get_conversation_agents(con_short_id)

        agent_ids: List[int] = []
        for row in rows or []:
            agent_id = int(getattr(row, "agent_id", 0) or 0)
            if agent_id > 0:
                agent_ids.append(agent_id)

        if not agent_ids:
            return []

        return await self.load_agents_by_ids(agent_ids)

    async def load_private_agent(self, con_id: str) -> Agent | None:
        agent_id = self.parse_private_agent_id(con_id)
        if agent_id <= 0:
            return None

        agents = await self.load_agents_by_ids([agent_id])
        if not agents:
            return None
        return agents[0]

    async def load_agents_by_ids(self, agent_ids: List[int]) -> List[Agent]:
        if not agent_ids:
            return []

        agent_infos = await self.agent_svc.get_agents_by_ids(agent_ids)

        result: List[Agent] = []
        for info in agent_infos:
            result.append(
                Agent(
                    agent_id=int(info.agent_id),
                    name=str(info.agent_name or ""),
                    personality=str(info.personality or ""),
                )
            )
        return result

    def parse_private_agent_id(self, con_id: str) -> int:
        parts = con_id.split(":")
        if len(parts) != 3:
            return 0
        if parts[0] != "ai":
            return 0
        return int(parts[2] or 0)

    async def decide_reply(self, con_short_id: int, agents: List[Agent], items: List[Dict[str, Any]]) -> Any:
        state = await self.memory_svc.get_summary_state(con_short_id)

        agents_json = json_dumps(
            [
                {
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "personality": a.personality,
                }
                for a in agents
            ]
        )

        messages = [
            {
                "sender_type": int(it["sender_type"]),
                "sender_id": int(it["sender_id"]),
                "msg_type": int(it["msg_type"]),
                "msg_content": str(it["msg_content"])[:300],
            }
            for it in items[-50:]
        ]

        inputs = DecisionInputs(
            agents_json=agents_json,
            recent_messages_json=json_dumps(messages),
            short_summary=state.short_summary,
            long_summary=state.long_summary,
        )
        return await self.llm.decide_reply(inputs)

    async def generate_reply(
        self,
        con_short_id: int,
        agent: Agent,
        items: List[Dict[str, Any]],
        reason: str,
    ) -> str:
        recent_messages = self.format_messages(items[-30:])
        recent_text = self.join_recent_text(items[-30:])

        state = await self.memory_svc.get_summary_state(con_short_id)
        glossary = await self.memory_svc.get_relevant_glossary(con_short_id, recent_text, limit=8)

        inputs = ReplyInputs(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            agent_personality=agent.personality,
            trigger_reason=reason,
            recent_messages=recent_messages,
            short_summary=state.short_summary,
            long_summary=state.long_summary,
            glossary_json=json_dumps(
                [
                    {
                        "term": x.term,
                        "meaning": x.meaning,
                        "count": x.count,
                    }
                    for x in glossary
                ]
            ),
        )
        return await self.llm.generate_reply(inputs)

    async def send_reply(
        self,
        con_short_id: int,
        agent: Agent,
        items: List[Dict[str, Any]],
        text: str,
    ) -> None:
        last = items[-1]

        task = {
            "task_id": uuid.uuid4().hex,
            "con_short_id": con_short_id,
            "con_id": str(last["con_id"]),
            "con_type": int(last["con_type"]),
            "sender_id": agent.agent_id,
            "sender_type": 2,
            "send_ts_ms": now_ms(),
            "msg_type": 1,
            "msg_content": text,
            "client_msg_id": 0,
            "extra": "",
            "retry_count": 0,
        }

        await self.redis.enqueue_send_task(json_dumps(task), task["send_ts_ms"])

    async def update_memories_safe(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        try:
            await self.update_short_memory_if_needed(con_short_id, items)
            await self.update_long_memory_if_needed(con_short_id, items)
            await self.update_glossary_if_needed(con_short_id, items)
        except Exception:
            logger.exception("update_memories failed con_short_id=%s", con_short_id)

    async def update_short_memory_if_needed(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        state = await self.memory_svc.get_summary_state(con_short_id)

        enough_msgs = len(items) >= self.SHORT_MSG_TRIGGER
        enough_time = now_ms() - state.short_updated_at >= self.SHORT_TIME_TRIGGER_MS
        topic_changed = self.detect_topic_change(items)

        if not (enough_msgs or enough_time or topic_changed):
            return

        recent_messages = self.format_messages(items[-20:])
        if not recent_messages.strip():
            return

        new_short = await self.llm.update_short_memory(
            ShortMemoryInputs(
                old_short_summary=state.short_summary,
                recent_messages=recent_messages,
            )
        )
        new_short = str(new_short or "").strip()
        if not new_short:
            return

        logger.info("short_memory: %s", new_short)

        await self.memory_svc.save_short_summary(
            con_short_id=con_short_id,
            short_summary=new_short,
        )

    async def update_long_memory_if_needed(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        state = await self.memory_svc.get_summary_state(con_short_id)

        enough_msgs = len(items) >= self.LONG_MSG_TRIGGER
        enough_time = now_ms() - state.long_updated_at >= self.LONG_TIME_TRIGGER_MS

        if not (enough_msgs or enough_time):
            return

        recent_messages = self.format_messages(items[-100:])
        if not recent_messages.strip():
            return

        new_long = await self.llm.update_long_memory(
            LongMemoryInputs(
                old_long_summary=state.long_summary,
                recent_messages=recent_messages,
            )
        )
        new_long = str(new_long or "").strip()
        if not new_long:
            return

        logger.info("long_memory: %s", new_long)

        await self.memory_svc.save_long_summary(
            con_short_id=con_short_id,
            long_summary=new_long,
        )

    async def update_glossary_if_needed(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        # todo:只看当前窗口
        window_messages = self.format_messages(items[-20:])
        if not window_messages.strip():
            return

        unknown_terms = await self.llm.extract_unknown_terms(
            GlossaryExtractInputs(
                recent_messages=window_messages,
            )
        )
        if not unknown_terms:
            return

        logger.info("glossary extract: %s", unknown_terms)

        await self.memory_svc.upsert_glossary_terms(con_short_id, unknown_terms)

        terms_need_meaning = await self.memory_svc.get_terms_need_meaning(
            con_short_id=con_short_id,
            min_count=3,
            limit=20,
        )
        if not terms_need_meaning:
            return

        state = await self.memory_svc.get_summary_state(con_short_id)

        inferred = await self.llm.infer_glossary_meanings(
            GlossaryInferInputs(
                terms_json=json_dumps(terms_need_meaning),
                recent_messages=window_messages,
                short_summary=state.short_summary,
                long_summary=state.long_summary,
            )
        )
        if not inferred:
            return

        logger.info("glossary inferred: %s", inferred)

        await self.memory_svc.save_glossary_meanings(con_short_id, inferred)

    def detect_topic_change(self, items: List[Dict[str, Any]]) -> bool:
        return False

    def find_agent(self, agents: List[Agent], agent_id: int) -> Agent | None:
        for agent in agents:
            if agent.agent_id == agent_id:
                return agent
        return None

    def format_messages(self, items: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for it in items:
            sender_id = int(it["sender_id"])
            sender_type = int(it["sender_type"])
            text = str(it["msg_content"]).replace("\n", " ").strip()
            if not text:
                continue

            if sender_type == 1:
                role = f"用户 {sender_id}"
            elif sender_type == 2:
                role = f"Agent {sender_id}"
            else:
                role = f"未知 {sender_id}"

            lines.append(f"[{role}]\n{text}")

        return "\n\n".join(lines)

    def join_recent_text(self, items: List[Dict[str, Any]]) -> str:
        texts: List[str] = []
        for it in items:
            text = str(it.get("msg_content", "") or "").strip()
            if text:
                texts.append(text)
        return "\n".join(texts)