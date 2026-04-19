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

    from chatbot.agent.llm import LLMClient
    llm = LLMClient(s)

    from chatbot.agent.tools import make_jina_search_tool
    from chatbot.agent.graph import ConversationGraph

    jina_tool = make_jina_search_tool(
        base_url=s.JINA_SEARCH_BASE_URL,
        api_key=s.JINA_API_KEY,
        timeout_sec=s.JINA_SEARCH_TIMEOUT_SEC,
        max_chars=getattr(s, "JINA_SEARCH_MAX_CHARS", 2000),
    )
    graph = ConversationGraph(
        chat=llm.get_chat_model(),
        base_tools=[jina_tool],  # fetch_context_tool 在 graph 内部创建
        im=im,
        agent_svc=agent_svc,
        memory_svc=memory_svc,
        llm=llm,
        redis_svc=redis_svc,
        max_tool_calls=getattr(s, "AGENT_MAX_TOOL_CALLS", 1),
    )

    from chatbot.agent.window import WindowManager
    planner = WindowManager(
        redis_svc=redis_svc,
        llm=llm,
        memory_svc=memory_svc,
        graph=graph,
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