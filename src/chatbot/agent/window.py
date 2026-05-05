from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from chatbot.proto_gen import im_pb2

from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)

SENDER_TYPE_MAP = {1: "用户", 2: "Agent", 3: "系统"}
MSG_TYPE_MAP = {1: "文本", 2: "图片", 3: "视频", 4: "分享创作"}

SHANGHAI_TZ = timezone(timedelta(hours=8))


def _format_ts(ts: int) -> str:
    if ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts, tz=SHANGHAI_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


class WindowManager:
    def __init__(
            self,
            redis_svc: Any,
            graph: Any,
            im: Any,
            aigc: Any,
            action: Any,
            window_sec: int = 10,
    ) -> None:
        self.redis = redis_svc
        self.graph = graph
        self.im = im
        self.aigc = aigc
        self.action = action
        self.window_sec = window_sec

    async def on_message(self, evt: im_pb2.MessageEvent) -> None:
        msg = evt.msg_body
        con_short_id = int(msg.con_short_id or 0)
        con_type = int(msg.con_type or 0)
        sender_type = int(msg.sender_type or 0)
        sender_id = int(msg.sender_id or 0)
        msg_type = int(msg.msg_type or 0)

        if con_short_id <= 0:
            return
        if con_type not in (2, 4):
            return
        if msg_type != 1:
            return

        sender_name = ""
        try:
            if sender_type == 1 and sender_id > 0:
                infos = await self.action.get_user_infos([sender_id])
                if sender_id in infos:
                    sender_name = str(getattr(infos[sender_id], "username", "") or "")
            elif sender_type == 2 and sender_id > 0:
                agents = await self.aigc.get_agents_by_ids([sender_id])
                if agents:
                    sender_name = str(getattr(agents[0], "agent_name", "") or "")
        except Exception:
            logger.exception("lookup sender_name failed sender_id=%s sender_type=%s", sender_id, sender_type)

        item = {
            "sender_id": sender_id,
            "sender_type": SENDER_TYPE_MAP.get(sender_type, "未知"),
            "sender_name": sender_name,
            "msg_type": MSG_TYPE_MAP.get(msg_type, "命令"),
            "msg_content": str(msg.msg_content or ""),
            "create_time": _format_ts(int(msg.create_time or 0)),
            "con_index": int(msg.con_index or 0),
        }
        await self.redis.append_window_item(con_short_id, item)

        if con_type == 4 and sender_type == 2:
            return

        await self.redis.upsert_window(con_short_id, self.window_sec)
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
        cur_items = await self.redis.get_window_items(con_short_id)
        history_items = await self.redis.get_all_buf_items(con_short_id)
        merged = await self.redis.merge_buf_to_all(con_short_id)
        await self.redis.clear_window(con_short_id)
        if merged > 0:
            await self.redis.incr_merge_count(con_short_id)

        await self.graph.run(con_short_id, cur_items, history_items)
