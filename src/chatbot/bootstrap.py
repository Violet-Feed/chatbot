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

    # Infra clients
    redis: Any
    engine: AsyncEngine
    im: Any

    # DAL services
    summary_svc: Any
    episodic_svc: Any
    style_svc: Any
    redis_svc: Any

    # App components
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
    """
    优先用 dal/mysql/client.py（你当前结构），若不存在则回退 dal/mysql/engine.py。
    """
    try:
        from chatbot.dal.mysql.client import new_mysql_client, new_session_factory  # type: ignore

        engine = new_mysql_client(dsn)
        sf = new_session_factory(engine)
        return engine, sf
    except Exception:
        from chatbot.dal.mysql.engine import new_engine  # type: ignore

        engine = new_engine(dsn)
        sf = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
        return engine, sf


def build_container(settings: Optional[Settings] = None, llm: Optional[Any] = None) -> Container:
    s = settings or Settings()

    # --- Redis client ---
    from chatbot.dal.redis.client import new_redis

    redis_client = new_redis(s.REDIS_URL)

    # --- MySQL engine + session factory ---
    engine, sf = _build_mysql_engine_and_sf(s.MYSQL_DSN)

    # --- MySQL memory services ---
    from chatbot.dal.mysql.memory_service import EpisodicMemoryService, RollingSummaryService, StyleService

    summary_svc = RollingSummaryService(sf)
    episodic_svc = EpisodicMemoryService(sf)
    style_svc = StyleService(sf)

    # --- Redis service（窗口/频控/冷却/调度入队/取消）---
    from chatbot.dal.redis.service import RedisService

    redis_svc = RedisService(
        redis=redis_client,
        window_base_ms=s.WINDOW_BASE_MS,
        window_max_ms=s.WINDOW_MAX_MS,
        group_bucket_cap=s.GROUP_BUCKET_CAP,
        group_bucket_period_ms=s.GROUP_BUCKET_PERIOD_MS,
        bot2bot_token_ttl_sec=s.BOT2BOT_TOKEN_TTL_SEC,
    )

    # --- IM gRPC client ---
    from chatbot.dal.rpc.im import IMClient

    im = IMClient(target=s.IM_GRPC_TARGET)

    # --- LLM client（OpenAI-compatible Qwen）---
    if llm is None:
        try:
            from chatbot.planner.llm import LangChainQwenOpenAICompatibleClient

            llm_client = LangChainQwenOpenAICompatibleClient(s)
        except Exception:
            logger.exception("init llm client failed, fallback to noop")
            from chatbot.planner.llm import NoopLLMClient

            llm_client = NoopLLMClient()
    else:
        llm_client = llm

    # --- Planner（agents 从 IM 拿，memory 用 mysql services）---
    from chatbot.planner.planner import Planner

    planner = Planner(
        settings=s,
        redis_svc=redis_svc,
        im=im,
        summary_svc=summary_svc,
        episodic_svc=episodic_svc,
        style_svc=style_svc,
        llm=llm_client,
        enable_memory=True,
    )

    # --- RocketMQ consumer ---
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

    # --- Send scheduler ---
    from chatbot.scheduler.send_scheduler import SendScheduler

    scheduler = SendScheduler(settings=s, redis=redis_client, im=im)

    return Container(
        settings=s,
        redis=redis_client,
        engine=engine,
        im=im,
        summary_svc=summary_svc,
        episodic_svc=episodic_svc,
        style_svc=style_svc,
        redis_svc=redis_svc,
        planner=planner,
        consumer=consumer,
        scheduler=scheduler,
    )
