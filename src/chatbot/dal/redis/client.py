from __future__ import annotations

import redis.asyncio as redis_async


def new_redis(url: str) -> "redis_async.Redis":
    """
    创建 Redis asyncio 客户端。
    decode_responses=False：保持 bytes，性能更稳；业务处按需 decode。
    """
    return redis_async.from_url(url, decode_responses=False)
