from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from chatbot.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings

    redis: Any
    engine: AsyncEngine
    im: Any

    redis_svc: Any
    agent_svc: Any
    memory_svc: Any
    llm: Any

    planner: Any
    consumer: Any
    scheduler: Any

    async def aclose(self) -> None:
        try:
            close = getattr(self.im, "aclose", None)
            if callable(close):
                await close()
        except Exception:
            logger.exception("close im client failed")

        try:
            await self.engine.dispose()
        except Exception:
            logger.exception("dispose mysql engine failed")

        try:
            aclose = getattr(self.redis, "aclose", None)
            if callable(aclose):
                await aclose()
        except Exception:
            logger.exception("close redis failed")


def _build_mysql_engine_and_sf(dsn: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    try:
        from chatbot.dal.mysql.client import new_mysql_client, new_session_factory

        engine = new_mysql_client(dsn)
        sf = new_session_factory(engine)
        return engine, sf
    except Exception:
        logger.exception("build mysql engine failed")
        raise


def build_container(settings: Optional[Settings] = None) -> Container:
    s = settings or Settings()

    from chatbot.dal.redis.client import new_redis
    redis_client = new_redis(s.REDIS_URL)

    engine, sf = _build_mysql_engine_and_sf(s.MYSQL_DSN)

    from chatbot.dal.mysql.agent_service import AgentService
    from chatbot.dal.mysql.memory_service import MemoryService

    agent_svc = AgentService(sf)
    memory_svc = MemoryService(sf)

    from chatbot.dal.redis.service import RedisService
    redis_svc = RedisService(redis_client)

    from chatbot.dal.rpc.im import IMClient
    im = IMClient(target=s.IM_GRPC_TARGET)

    from chatbot.planner.llm import LLMClient
    llm = LLMClient(s)

    from chatbot.planner.planner import Planner
    planner = Planner(
        redis_svc=redis_svc,
        im=im,
        llm=llm,
        agent_svc=agent_svc,
        memory_svc=memory_svc,
        window_sec=getattr(s, "WINDOW_SEC", 5),
    )

    from chatbot.consumer.message_consumer import MessageConsumer, RocketMQConsumeConfig
    mq_cfg = RocketMQConsumeConfig(
        namesrv_addr=s.ROCKETMQ_NAME_SERVER,
        topic=s.ROCKETMQ_TOPIC,
        consumer_group=s.ROCKETMQ_CONSUMER_GROUP,
        subscription_expression=getattr(s, "ROCKETMQ_SUBSCRIPTION_EXPRESSION", "*"),
    )

    try:
        consumer = MessageConsumer(cfg=mq_cfg, handler=planner.on_message)
    except TypeError:
        consumer = MessageConsumer(handler=planner.on_message, cfg=mq_cfg)

    from chatbot.scheduler.send_scheduler import SendScheduler
    scheduler = SendScheduler(settings=s, redis=redis_client, im=im)

    return Container(
        settings=s,
        redis=redis_client,
        engine=engine,
        im=im,
        redis_svc=redis_svc,
        agent_svc=agent_svc,
        memory_svc=memory_svc,
        llm=llm,
        planner=planner,
        consumer=consumer,
        scheduler=scheduler,
    )