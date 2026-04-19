from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from chatbot.proto_gen import im_pb2
from chatbot.agent.llm import (
    ShortMemoryInputs,
    LongMemoryInputs,
    GlossaryExtractInputs,
    GlossaryInferInputs,
)
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


class WindowManager:
    SHORT_MSG_TRIGGER = 20
    SHORT_TIME_TRIGGER_MS = 30 * 60 * 1000

    LONG_MSG_TRIGGER = 100
    LONG_TIME_TRIGGER_MS = 24 * 60 * 60 * 1000

    def __init__(
        self,
        redis_svc: Any,
        llm: Any,
        memory_svc: Any,
        graph: Any,
        window_sec: int = 5,
    ) -> None:
        self.redis = redis_svc
        self.llm = llm
        self.memory_svc = memory_svc
        self.graph = graph
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
        await self.graph.run(con_short_id, items)

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

        recent_messages = _format_messages(items[-20:])
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

        logger.info("short_memory updated con_short_id=%s", con_short_id)
        await self.memory_svc.save_short_summary(con_short_id=con_short_id, short_summary=new_short)

    async def update_long_memory_if_needed(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        state = await self.memory_svc.get_summary_state(con_short_id)

        enough_msgs = len(items) >= self.LONG_MSG_TRIGGER
        enough_time = now_ms() - state.long_updated_at >= self.LONG_TIME_TRIGGER_MS

        if not (enough_msgs or enough_time):
            return

        recent_messages = _format_messages(items[-100:])
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

        logger.info("long_memory updated con_short_id=%s", con_short_id)
        await self.memory_svc.save_long_summary(con_short_id=con_short_id, long_summary=new_long)

    async def update_glossary_if_needed(self, con_short_id: int, items: List[Dict[str, Any]]) -> None:
        window_messages = _format_messages(items[-20:])
        if not window_messages.strip():
            return

        unknown_terms = await self.llm.extract_unknown_terms(
            GlossaryExtractInputs(recent_messages=window_messages)
        )
        if not unknown_terms:
            return

        await self.memory_svc.upsert_glossary_terms(con_short_id, unknown_terms)

        terms_need_meaning = await self.memory_svc.get_terms_need_meaning(
            con_short_id=con_short_id, min_count=3, limit=20,
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

        await self.memory_svc.save_glossary_meanings(con_short_id, inferred)

    def detect_topic_change(self, items: List[Dict[str, Any]]) -> bool:
        return False


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
