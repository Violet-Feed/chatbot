from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import redis.asyncio as redis_async

from chatbot.dal.redis import keys
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms


@dataclass(frozen=True)
class WindowUpsertResult:
    con_short_id: int
    deadline_ms: int


class RedisService:
    def __init__(self, redis: "redis_async.Redis", max_all_buf_items: int = 200, all_buf_keep: int = 50) -> None:
        self.r = redis
        self.max_all_buf_items = max_all_buf_items
        self.all_buf_keep = all_buf_keep

    async def upsert_window(
            self,
            con_short_id: int,
            window_sec: int,
    ) -> WindowUpsertResult:
        now = now_ms()
        deadline_ms = now + int(window_sec) * 1000
        ttl_sec = int(window_sec) + 60

        await self.r.set(keys.win_deadline(con_short_id), deadline_ms, ex=ttl_sec)
        return WindowUpsertResult(con_short_id=con_short_id, deadline_ms=deadline_ms)

    async def append_window_item(self, con_short_id: int, item: Dict[str, Any]) -> None:
        key = keys.win_buf(con_short_id)
        await self.r.rpush(key, json_dumps(item))

    async def mark_close_scheduled(self, con_short_id: int) -> bool:
        ok = await self.r.set(keys.winclose_scheduled(con_short_id), "1", nx=True, ex=300)
        return bool(ok)

    async def get_deadline_ms(self, con_short_id: int) -> Optional[int]:
        value = await self.r.get(keys.win_deadline(con_short_id))
        if value is None:
            return None
        return int(value)

    async def get_window_items(self, con_short_id: int) -> List[Dict[str, Any]]:
        rows = await self.r.lrange(keys.win_buf(con_short_id), 0, -1)
        items: List[Dict[str, Any]] = []

        for row in rows:
            text = row.decode() if isinstance(row, (bytes, bytearray)) else str(row)
            obj = json.loads(text)
            if isinstance(obj, dict):
                items.append(obj)

        return items

    async def clear_window(self, con_short_id: int) -> None:
        await self.r.delete(
            keys.win_deadline(con_short_id),
            keys.winclose_scheduled(con_short_id),
            keys.win_buf(con_short_id),
        )

    # --- all_buf (accumulated window) ---

    async def merge_buf_to_all(self, con_short_id: int) -> int:
        src = keys.win_buf(con_short_id)
        dst = keys.win_all_buf(con_short_id)
        rows = await self.r.lrange(src, 0, -1)
        if rows:
            pipe = self.r.pipeline()
            for row in rows:
                pipe.rpush(dst, row)
            await pipe.execute()
        return len(rows) if rows else 0

    async def get_all_buf_items(self, con_short_id: int) -> List[Dict[str, Any]]:
        rows = await self.r.lrange(keys.win_all_buf(con_short_id), 0, -1)
        items: List[Dict[str, Any]] = []
        for row in rows:
            text = row.decode() if isinstance(row, (bytes, bytearray)) else str(row)
            obj = json.loads(text)
            if isinstance(obj, dict):
                items.append(obj)
        return items

    async def get_all_buf_length(self, con_short_id: int) -> int:
        return await self.r.llen(keys.win_all_buf(con_short_id))

    async def trim_all_buf(self, con_short_id: int) -> None:
        key = keys.win_all_buf(con_short_id)
        length = await self.r.llen(key)
        if length > self.max_all_buf_items:
            await self.r.ltrim(key, -(self.all_buf_keep), -1)

    # --- merge count (short memory trigger) ---

    async def incr_merge_count(self, con_short_id: int) -> int:
        key = keys.win_merge_count(con_short_id)
        return await self.r.incr(key)

    async def get_merge_count(self, con_short_id: int) -> int:
        val = await self.r.get(keys.win_merge_count(con_short_id))
        if val is None:
            return 0
        return int(val)

    async def reset_merge_count(self, con_short_id: int) -> None:
        await self.r.delete(keys.win_merge_count(con_short_id))

    async def enqueue_send_task(self, task_json: str, send_ts_ms: int) -> None:
        await self.r.zadd("sched:send", {task_json: float(send_ts_ms)})
