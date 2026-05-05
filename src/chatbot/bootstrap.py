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
    aigc: Any
    action: Any

    redis_svc: Any
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
            close = getattr(self.aigc, "aclose", None)
            if callable(close):
                await close()
        except Exception:
            logger.exception("close aigc client failed")

        try:
            close = getattr(self.action, "aclose", None)
            if callable(close):
                await close()
        except Exception:
            logger.exception("close action client failed")

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

    from chatbot.dal.mysql.memory_service import MemoryService

    memory_svc = MemoryService(sf)

    from chatbot.dal.redis.service import RedisService
    redis_svc = RedisService(redis_client)

    from chatbot.dal.rpc.im import IMClient
    im = IMClient(target=s.IM_GRPC_TARGET)

    from chatbot.dal.rpc.aigc import AigcClient
    aigc = AigcClient(target=s.AIGC_GRPC_TARGET)

    from chatbot.dal.rpc.action import ActionClient
    action = ActionClient(target=s.ACTION_GRPC_TARGET)

    from chatbot.agent.tools import make_web_search_tool
    from chatbot.agent.llm import LLMClient

    web_search_tool = make_web_search_tool(
        base_url=s.WEB_SEARCH_BASE_URL,
        api_key=s.BOCHA_API_KEY,
        timeout_sec=s.WEB_SEARCH_TIMEOUT_SEC,
        max_chars=getattr(s, "WEB_SEARCH_MAX_CHARS", 10000),
        count=getattr(s, "WEB_SEARCH_DEFAULT_COUNT", 10),
        freshness=getattr(s, "WEB_SEARCH_DEFAULT_FRESHNESS", "noLimit"),
        summary=getattr(s, "WEB_SEARCH_DEFAULT_SUMMARY", True),
    )
    llm = LLMClient(s, base_tools=[web_search_tool], im_client=im,
                     max_tool_calls=getattr(s, "AGENT_MAX_TOOL_CALLS", 1))

    from chatbot.agent.graph import ConversationGraph

    graph = ConversationGraph(
        im=im,
        aigc=aigc,
        memory_svc=memory_svc,
        llm=llm,
        redis_svc=redis_svc,
    )

    from chatbot.agent.window import WindowManager
    planner = WindowManager(
        redis_svc=redis_svc,
        graph=graph,
        im=im,
        aigc=aigc,
        action=action,
        window_sec=getattr(s, "WINDOW_SEC", 10),
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
        aigc=aigc,
        action=action,
        redis_svc=redis_svc,
        memory_svc=memory_svc,
        llm=llm,
        planner=planner,
        consumer=consumer,
        scheduler=scheduler,
    )
