# src/chatbot/scheduler/send_scheduler.py
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import redis.asyncio as redis_async

from chatbot.dal.redis import keys
from chatbot.dal.rpc.im import IMClient
from chatbot.settings import Settings
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


def _loads_task(s: str) -> Dict[str, Any]:
    obj = orjson.loads(s)
    if not isinstance(obj, dict):
        raise ValueError("task must be json object")
    return obj


def _dumps(obj: Any) -> str:
    return orjson.dumps(obj).decode("utf-8")


def _load_lua_file(filename: str) -> str:
    # 复用 dal/redis/lua
    lua_dir = Path(__file__).resolve().parents[1] / "dal" / "redis" / "lua"
    path = lua_dir / filename
    return path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class SendSchedulerConfig:
    """发送调度配置。"""

    zset_key: str = "sched:send"
    poll_interval_ms: int = 200
    batch_size: int = 64
    max_retry: int = 3


class SendScheduler:
    """
    send_scheduler：
    - 从 Redis ZSET（sched:send）取出到期任务（Lua 原子 pop）
    - 若任务被取消（sched:cancel:{task_id}）则跳过
    - 使用 sched:sent:{task_id} 做幂等去重（防重复发送）
    - 调用 IMClient.send_message 发送
    - 失败则按指数退避重试回 ZSET（最多 max_retry 次）
    """

    def __init__(
        self,
        settings: Settings,
        redis: "redis_async.Redis",
        im: IMClient,
        cfg: Optional[SendSchedulerConfig] = None,
    ) -> None:
        self.s = settings
        self.r = redis
        self.im = im
        self.cfg = cfg or SendSchedulerConfig(
            poll_interval_ms=settings.SCHEDULER_POLL_INTERVAL_MS,
            batch_size=settings.SCHEDULER_BATCH_SIZE,
        )

        self._stop_evt = asyncio.Event()
        self._pop_sha: Optional[str] = None
        self._pop_lua: Optional[str] = None

    def stop(self) -> None:
        self._stop_evt.set()

    async def run_forever(self) -> None:
        if self._pop_lua is None:
            self._pop_lua = _load_lua_file("pop_due.lua")

        try:
            self._pop_sha = await self.r.script_load(self._pop_lua)
        except Exception:
            self._pop_sha = None

        logger.info("send_scheduler started")
        try:
            while not self._stop_evt.is_set():
                due = await self._pop_due()
                if not due:
                    await asyncio.sleep(self.cfg.poll_interval_ms / 1000.0)
                    continue

                for task_json in due:
                    await self._handle_one(task_json)
        finally:
            logger.info("send_scheduler stopped")

    async def _pop_due(self) -> List[str]:
        now = now_ms()
        try:
            if self._pop_sha:
                res = await self.r.evalsha(self._pop_sha, 1, self.cfg.zset_key, now, self.cfg.batch_size)
            else:
                # sha 不可用则回退 eval
                res = await self.r.eval(self._pop_lua, 1, self.cfg.zset_key, now, self.cfg.batch_size)

            out: List[str] = []
            for v in res or []:
                if isinstance(v, (bytes, bytearray)):
                    out.append(v.decode("utf-8", errors="ignore"))
                else:
                    out.append(str(v))
            return out
        except Exception:
            logger.exception("pop_due 失败")
            return []

    def _cancel_key(self, task_id: str) -> str:
        # 你 keys.cancel_task(task_id) 已存在则用它，否则 fallback
        fn = getattr(keys, "cancel_task", None)
        if callable(fn):
            return fn(task_id)
        return f"sched:cancel:{task_id}"

    def _sent_key(self, task_id: str) -> str:
        fn = getattr(keys, "sent_task", None)
        if callable(fn):
            return fn(task_id)
        return f"sched:sent:{task_id}"

    async def _handle_one(self, task_json: str) -> None:
        try:
            task = _loads_task(task_json)
        except Exception:
            logger.exception("任务 JSON 解析失败，丢弃")
            return

        task_id = str(task.get("task_id") or "")
        if not task_id:
            logger.warning("任务缺少 task_id，丢弃")
            return

        # 取消检查
        if await self._is_canceled(task_id):
            return

        # 幂等去重
        if not await self._mark_sent_once(task_id):
            return

        try:
            await self._send(task)
        except Exception:
            # 失败：撤销 sent 标记并重试入队
            await self._unmark_sent(task_id)
            await self._retry(task)

    async def _send(self, task: Dict[str, Any]) -> None:
        con_short_id = int(task.get("con_short_id", 0) or 0)
        con_id = str(task.get("con_id", "") or "")
        con_type = int(task.get("con_type", 2) or 2)

        user_id = int(task.get("agent_id", 0) or 0)
        msg_type = int(task.get("msg_type", 1) or 1)
        msg_content = str(task.get("msg_content", "") or "")

        if con_short_id <= 0 or user_id <= 0 or not msg_content:
            raise ValueError(
                f"invalid task: con_short_id={con_short_id}, user_id={user_id}, content_empty={not msg_content}"
            )

        client_msg_id = task.get("client_msg_id")
        client_msg_id_int = int(client_msg_id) if client_msg_id is not None else None

        msg_id = await self.im.send_message(
            user_id=user_id,
            con_short_id=con_short_id,
            con_id=con_id,
            con_type=con_type,
            msg_type=msg_type,
            msg_content=msg_content,
            client_msg_id=client_msg_id_int,
        )
        logger.info(
            "send ok task_id=%s msg_id=%s con=%s sender=%s",
            task.get("task_id"),
            msg_id,
            con_short_id,
            user_id,
        )

    async def _retry(self, task: Dict[str, Any]) -> None:
        retry = int(task.get("retry_count", 0) or 0) + 1
        task["retry_count"] = retry

        if retry > self.cfg.max_retry:
            logger.warning("drop task exceed max_retry task_id=%s", task.get("task_id"))
            return

        # 指数退避 + 抖动：1s,2s,4s... 上限 30s
        backoff = min(30_000, (2 ** (retry - 1)) * 1000)
        jitter = random.randint(0, 800)
        send_ts = now_ms() + backoff + jitter
        task["send_ts_ms"] = int(send_ts)

        await self.r.zadd(self.cfg.zset_key, {_dumps(task): float(send_ts)})
        logger.warning(
            "retry enqueue task_id=%s retry=%s delay=%sms",
            task.get("task_id"),
            retry,
            backoff + jitter,
        )

    async def _is_canceled(self, task_id: str) -> bool:
        return bool(await self.r.get(self._cancel_key(task_id)))

    async def _mark_sent_once(self, task_id: str) -> bool:
        ok = await self.r.set(self._sent_key(task_id), "1", nx=True, ex=3600)
        return bool(ok)

    async def _unmark_sent(self, task_id: str) -> None:
        await self.r.delete(self._sent_key(task_id))
