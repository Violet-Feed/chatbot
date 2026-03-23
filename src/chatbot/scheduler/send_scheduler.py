from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

import orjson
import redis.asyncio as redis_async

from chatbot.dal.rpc.im import IMClient
from chatbot.settings import Settings
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


def loads_task(s: str) -> Dict[str, Any]:
    obj = orjson.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("task must be json object")
    return obj


def dumps_task(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


class SendScheduler:
    ZSET_KEY = "sched:send"
    POLL_INTERVAL_MS = 200
    BATCH_SIZE = 100
    MAX_RETRY = 2

    def __init__(
        self,
        settings: Settings,
        redis: "redis_async.Redis",
        im: IMClient,
    ) -> None:
        self.settings = settings
        self.r = redis
        self.im = im
        self._stop_evt = asyncio.Event()

    def stop(self) -> None:
        self._stop_evt.set()

    async def run_forever(self) -> None:
        logger.info("send_scheduler started")
        try:
            while not self._stop_evt.is_set():
                due = await self.pop_due_tasks()
                if not due:
                    await asyncio.sleep(self.POLL_INTERVAL_MS / 1000.0)
                    continue

                for task_json in due:
                    claimed = await self.claim_task(task_json)
                    if not claimed:
                        continue
                    await self.handle_one(task_json)
        finally:
            logger.info("send_scheduler stopped")

    async def pop_due_tasks(self) -> List[str]:
        now = now_ms()
        rows = await self.r.zrangebyscore(
            self.ZSET_KEY,
            min=0,
            max=now,
            start=0,
            num=self.BATCH_SIZE,
        )

        result: List[str] = []
        for row in rows:
            if isinstance(row, (bytes, bytearray)):
                result.append(row.decode("utf-8"))
            else:
                result.append(str(row))
        return result

    async def claim_task(self, task_json: str) -> bool:
        removed = await self.r.zrem(self.ZSET_KEY, task_json)
        return int(removed or 0) > 0

    async def handle_one(self, task_json: str) -> None:
        task = loads_task(task_json)

        try:
            await self.send(task)
        except Exception:
            logger.exception("send failed task_id=%s", task.get("task_id"))
            await self.retry(task)

    async def send(self, task: Dict[str, Any]) -> None:
        con_short_id = int(task.get("con_short_id", 0) or 0)
        con_id = str(task.get("con_id", "") or "")
        con_type = int(task.get("con_type", 0) or 0)

        sender_id = int(task.get("sender_id", 0) or 0)
        sender_type = int(task.get("sender_type", 2) or 2)
        msg_type = int(task.get("msg_type", 1) or 1)
        msg_content = str(task.get("msg_content", "") or "")

        if con_short_id <= 0:
            raise ValueError("invalid con_short_id")
        if sender_id <= 0:
            raise ValueError("invalid sender_id")
        if not msg_content:
            raise ValueError("empty msg_content")

        client_msg_id = task.get("client_msg_id")
        client_msg_id_int = int(client_msg_id) if client_msg_id is not None else None

        msg_id = await self.im.send_message(
            sender_id=sender_id,
            sender_type=sender_type,
            con_short_id=con_short_id,
            con_id=con_id,
            con_type=con_type,
            msg_type=msg_type,
            msg_content=msg_content,
            client_msg_id=client_msg_id_int,
        )

        logger.info(
            "send ok task_id=%s msg_id=%s con_short_id=%s sender_id=%s",
            task.get("task_id"),
            msg_id,
            con_short_id,
            sender_id,
        )

    async def retry(self, task: Dict[str, Any]) -> None:
        retry_count = int(task.get("retry_count", 0) or 0) + 1
        task["retry_count"] = retry_count

        if retry_count > self.MAX_RETRY:
            logger.warning("drop task task_id=%s retry_count=%s", task.get("task_id"), retry_count)
            return

        delay_ms = retry_count * 1000
        send_ts_ms = now_ms() + delay_ms
        task["send_ts_ms"] = send_ts_ms

        await self.r.zadd(self.ZSET_KEY, {dumps_task(task): float(send_ts_ms)})

        logger.warning(
            "retry task_id=%s retry_count=%s delay_ms=%s",
            task.get("task_id"),
            retry_count,
            delay_ms,
        )