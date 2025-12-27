from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import redis.asyncio as redis_async

from chatbot.dal.redis import keys
from chatbot.utils.json import json_dumps
from chatbot.utils.time import now_ms

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowUpsertResult:
    win_id: str
    deadline_ms: int
    count: int


class RedisService:
    """
    Redis 领域服务：统一管理
    - 窗口（win 5s/10s）
    - 频控桶（token bucket）
    - bot 链深、bot2bot 令牌
    - agent 冷却
    - 发送调度队列（ZSET）+ 取消标记 + 第二条待发送标记
    """

    def __init__(
        self,
        redis: "redis_async.Redis",
        window_base_ms: int,
        window_max_ms: int,
        group_bucket_cap: int,
        group_bucket_period_ms: int,
        bot2bot_token_ttl_sec: int,
    ) -> None:
        self.r = redis
        self.window_base_ms = int(window_base_ms)
        self.window_max_ms = int(window_max_ms)
        self.bucket_cap = int(group_bucket_cap)
        self.bucket_period_ms = int(group_bucket_period_ms)
        self.bot2bot_token_ttl_sec = int(bot2bot_token_ttl_sec)

        self._bucket_take_sha: Optional[str] = None
        self._bucket_take_lua: Optional[str] = None
        self._bucket_refund_sha: Optional[str] = None
        self._bucket_refund_lua: Optional[str] = None

    def _load_lua_file(self, filename: str) -> str:
        """
        从 src/chatbot/dal/redis/lua/ 读取 lua 脚本。
        注意：你在源码方式运行时一定可用；若未来打包为 wheel，需要把 lua 文件作为 package data。
        """
        lua_dir = Path(__file__).resolve().parent / "lua"
        path = lua_dir / filename
        return path.read_text(encoding="utf-8")

    # -------------------------
    # 窗口：创建/更新/写入/读取/清理
    # -------------------------
    async def upsert_window(self, g: int, con_index: int, is_hot: bool) -> WindowUpsertResult:
        now = now_ms()

        cur_b = await self.r.get(keys.win_id(g))
        if not cur_b:
            win_id = f"{g}:{now}"
            pipe = self.r.pipeline()
            pipe.set(keys.win_id(g), win_id, ex=15)
            pipe.set(keys.win_begin(g), now, ex=15)
            pipe.set(keys.win_deadline(g), now + self.window_base_ms, ex=15)
            pipe.set(keys.win_count(g), 1, ex=15)
            pipe.set(keys.win_last_ts(g), now, ex=15)
            pipe.set(keys.win_last_idx(g), con_index, ex=15)
            await pipe.execute()
            return WindowUpsertResult(win_id=win_id, deadline_ms=now + self.window_base_ms, count=1)

        win_id = cur_b.decode() if isinstance(cur_b, (bytes, bytearray)) else str(cur_b)

        pipe = self.r.pipeline()
        pipe.incr(keys.win_count(g))
        pipe.set(keys.win_last_ts(g), now, ex=15)
        pipe.get(keys.win_begin(g))
        pipe.get(keys.win_deadline(g))
        pipe.get(keys.win_last_idx(g))
        res = await pipe.execute()

        cnt = int(res[0])
        begin_ts = int(res[2] or now)
        deadline = int(res[3] or (begin_ts + self.window_base_ms))
        last_idx = int(res[4] or 0)

        if con_index > last_idx:
            await self.r.set(keys.win_last_idx(g), con_index, ex=15)

        if is_hot:
            new_deadline = min(begin_ts + self.window_max_ms, now + self.window_max_ms)
            if new_deadline > deadline:
                await self.r.set(keys.win_deadline(g), new_deadline, ex=15)
                deadline = new_deadline

        return WindowUpsertResult(win_id=win_id, deadline_ms=deadline, count=cnt)

    async def append_window_item(self, win_id: str, item: Dict[str, Any]) -> None:
        k = keys.win_buf(win_id)
        pipe = self.r.pipeline()
        pipe.rpush(k, json_dumps(item))
        pipe.expire(k, 30)
        pipe.ltrim(k, -200, -1)
        await pipe.execute()

    async def get_window_items(self, win_id: str) -> List[Dict[str, Any]]:
        raw = await self.r.lrange(keys.win_buf(win_id), 0, -1)
        out: List[Dict[str, Any]] = []
        for b in raw:
            s = b.decode() if isinstance(b, (bytes, bytearray)) else str(b)
            try:
                import orjson

                obj = orjson.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
        return out

    async def get_deadline_ms(self, g: int) -> Optional[int]:
        b = await self.r.get(keys.win_deadline(g))
        return int(b) if b else None

    async def clear_window(self, g: int, win_id: str) -> None:
        pipe = self.r.pipeline()
        pipe.delete(keys.win_buf(win_id))
        pipe.delete(keys.win_sent(win_id))
        pipe.delete(keys.pending_second(win_id))

        cur_b = await self.r.get(keys.win_id(g))
        cur = cur_b.decode() if isinstance(cur_b, (bytes, bytearray)) else (str(cur_b) if cur_b else "")
        if cur == win_id:
            pipe.delete(keys.win_id(g))
            pipe.delete(keys.win_begin(g))
            pipe.delete(keys.win_deadline(g))
            pipe.delete(keys.win_count(g))
            pipe.delete(keys.win_last_ts(g))
            pipe.delete(keys.win_last_idx(g))

        await pipe.execute()

    # -------------------------
    # 窗口 close watcher：去重标记 + 分布式锁
    # -------------------------
    async def mark_close_scheduled(self, win_id: str) -> bool:
        ok = await self.r.set(keys.winclose_scheduled(win_id), "1", nx=True, ex=60)
        return bool(ok)

    async def acquire_close_lock(self, win_id: str, px: int = 20_000) -> bool:
        ok = await self.r.set(keys.winclose_lock(win_id), "1", nx=True, px=px)
        return bool(ok)

    async def release_close_lock(self, win_id: str) -> None:
        await self.r.delete(keys.winclose_lock(win_id))

    # -------------------------
    # bot 行为控制：链深/冷却/bot2bot
    # -------------------------
    async def reset_bot_chain_depth(self, g: int) -> None:
        await self.r.set(keys.bot_chain_depth(g), 0, ex=3600)

    async def set_bot_chain_depth(self, g: int, depth: int) -> None:
        await self.r.set(keys.bot_chain_depth(g), int(depth), ex=3600)

    async def get_bot_chain_depth(self, g: int) -> int:
        b = await self.r.get(keys.bot_chain_depth(g))
        return int(b) if b else 0

    async def take_bot2bot_token(self, g: int) -> bool:
        ok = await self.r.set(keys.bot2bot_token(g), "1", nx=True, ex=self.bot2bot_token_ttl_sec)
        return bool(ok)

    async def is_agent_in_cooldown(self, agent_id: int) -> bool:
        b = await self.r.get(keys.agent_cooldown(agent_id))
        return bool(b and int(b) > now_ms())

    async def set_agent_cooldown_until(self, agent_id: int, until_ms: int, ttl_ms: int) -> None:
        await self.r.set(keys.agent_cooldown(agent_id), int(until_ms), px=int(ttl_ms))

    # -------------------------
    # 频控：群级 token bucket（Lua 文件）
    # -------------------------
    async def bucket_take(self, g: int, cost: int = 1) -> bool:
        """
        从群级令牌桶扣 cost 个 token，失败则表示该窗口不允许 bot 发言。
        使用 lua/bucket_take.lua 保证原子性。
        """
        if self._bucket_take_lua is None:
            self._bucket_take_lua = self._load_lua_file("bucket_take.lua")

        if self._bucket_take_sha is None:
            try:
                self._bucket_take_sha = await self.r.script_load(self._bucket_take_lua)
            except Exception:
                self._bucket_take_sha = None

        now = now_ms()
        try:
            if self._bucket_take_sha:
                ok = await self.r.evalsha(
                    self._bucket_take_sha,
                    1,
                    keys.bucket(g),
                    now,
                    self.bucket_cap,
                    self.bucket_period_ms,
                    int(cost),
                )
            else:
                ok = await self.r.eval(
                    self._bucket_take_lua,
                    1,
                    keys.bucket(g),
                    now,
                    self.bucket_cap,
                    self.bucket_period_ms,
                    int(cost),
                )
            return int(ok) == 1
        except Exception:
            logger.exception("bucket_take 执行失败，默认拒绝")
            return False

    async def bucket_refund(self, g: int, cost: int = 1) -> None:
        """
        将 token 退回群级令牌桶（best-effort）。
        使用 lua/bucket_refund.lua 保证原子性。
        """
        if self._bucket_refund_lua is None:
            self._bucket_refund_lua = self._load_lua_file("bucket_refund.lua")

        if self._bucket_refund_sha is None:
            try:
                self._bucket_refund_sha = await self.r.script_load(self._bucket_refund_lua)
            except Exception:
                self._bucket_refund_sha = None

        now = now_ms()
        try:
            if self._bucket_refund_sha:
                await self.r.evalsha(
                    self._bucket_refund_sha,
                    1,
                    keys.bucket(g),
                    now,
                    self.bucket_cap,
                    self.bucket_period_ms,
                    int(cost),
                )
            else:
                await self.r.eval(
                    self._bucket_refund_lua,
                    1,
                    keys.bucket(g),
                    now,
                    self.bucket_cap,
                    self.bucket_period_ms,
                    int(cost),
                )
        except Exception:
            logger.exception("bucket_refund 执行失败，忽略")

    # -------------------------
    # 调度：ZSET + 取消 + 第二条 pending
    # -------------------------
    async def enqueue_send_task(self, task_json: str, send_ts_ms: int) -> None:
        await self.r.zadd("sched:send", {task_json: float(send_ts_ms)})

    async def set_pending_second(self, win_id: str, task_id: str) -> None:
        await self.r.set(keys.pending_second(win_id), task_id, ex=30)

    async def get_pending_second(self, win_id: str) -> Optional[str]:
        b = await self.r.get(keys.pending_second(win_id))
        if not b:
            return None
        return b.decode() if isinstance(b, (bytes, bytearray)) else str(b)

    async def clear_pending_second(self, win_id: str) -> None:
        await self.r.delete(keys.pending_second(win_id))

    async def cancel_task(self, task_id: str, ttl_sec: int = 60) -> None:
        await self.r.set(keys.cancel_task(task_id), "1", ex=int(ttl_sec))

    async def is_task_canceled(self, task_id: str) -> bool:
        b = await self.r.get(keys.cancel_task(task_id))
        return bool(b)

    async def cancel_pending_second_for_group(self, g: int) -> None:
        win_id_b = await self.r.get(keys.win_id(g))
        if not win_id_b:
            return
        win_id = win_id_b.decode() if isinstance(win_id_b, (bytes, bytearray)) else str(win_id_b)

        task_id = await self.get_pending_second(win_id)
        if not task_id:
            return
        await self.cancel_task(task_id)
        await self.clear_pending_second(win_id)
