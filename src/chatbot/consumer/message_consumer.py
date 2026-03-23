# src/chatbot/consumer/message_consumer.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

import orjson
from rocketmq.client import PushConsumer, ConsumeStatus  # rocketmq-client-python

from chatbot.proto_gen import im_pb2  # 你的 proto_gen 在 chatbot 下

logger = logging.getLogger(__name__)

AsyncHandler = Callable[[im_pb2.MessageEvent], Awaitable[None]]


@dataclass(frozen=True)
class RocketMQConsumeConfig:
    namesrv_addr: str = "127.0.0.1:9876"
    topic: str = "im_conv"
    consumer_group: str = "chatbot_conv_consumer"
    subscription_expression: str = "*"
    handler_timeout_sec: float = 2.0
    drop_bad_json: bool = True

def parse_message_event_json(raw: bytes) -> im_pb2.MessageEvent:
    obj = orjson.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("MQ JSON must be an object")

    body = obj.get("msg_body") or {}
    if not isinstance(body, dict):
        raise ValueError("msg_body must be an object")

    sender_id = int(body.get("sender_id", 0) or 0)
    sender_type = int(body.get("sender_type", 0) or 0)

    msg_body = im_pb2.MessageBody(
        sender_id=sender_id,
        sender_type=sender_type,
        con_id=str(body.get("con_id", "") or ""),
        con_short_id=int(body.get("con_short_id", 0) or 0),
        con_type=int(body.get("con_type", 0) or 0),
        client_msg_id=int(body.get("client_msg_id", 0) or 0),
        msg_id=int(body.get("msg_id", 0) or 0),
        msg_type=int(body.get("msg_type", 0) or 0),
        msg_content=str(body.get("msg_content", "") or ""),
        create_time=int(body.get("create_time", 0) or 0),
        extra=str(body.get("extra", "") or ""),
        con_index=int(body.get("con_index", 0) or 0),
    )

    evt = im_pb2.MessageEvent(
        msg_body=msg_body,
        con_index=int(obj.get("con_index", 0) or 0),
        stored=bool(obj.get("stored", False)),
        user_con_index=int(obj.get("user_con_index", 0) or 0),
        pre_user_con_index=int(obj.get("pre_user_con_index", 0) or 0),
        badge_count=int(obj.get("badge_count", 0) or 0),
        user_cmd_index=int(obj.get("user_cmd_index", 0) or 0),
        retry_count=int(obj.get("retry_count", 0) or 0),
    )
    return evt


class MessageConsumer:
    def __init__(self, handler: AsyncHandler, cfg: Optional[RocketMQConsumeConfig] = None):
        self._handler = handler
        self._cfg = cfg or RocketMQConsumeConfig()
        self._stop_evt = asyncio.Event()

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()

        consumer = PushConsumer(self._cfg.consumer_group)
        consumer.set_name_server_address(self._cfg.namesrv_addr)

        def _callback(msg) -> ConsumeStatus:
            try:
                body = getattr(msg, "body", None)
                if body is None:
                    return ConsumeStatus.CONSUME_SUCCESS

                raw = bytes(body)

                try:
                    evt = parse_message_event_json(raw)
                    logger.info("rocketmq: got message: %s", evt)
                except Exception:
                    logger.exception("rocketmq: bad json payload")
                    return ConsumeStatus.CONSUME_SUCCESS if self._cfg.drop_bad_json else ConsumeStatus.RECONSUME_LATER

                fut = asyncio.run_coroutine_threadsafe(self._handler(evt), loop)
                fut.result(timeout=self._cfg.handler_timeout_sec)
                return ConsumeStatus.CONSUME_SUCCESS
            except Exception:
                logger.exception("rocketmq: handler failed")
                return ConsumeStatus.RECONSUME_LATER

        try:
            consumer.subscribe(self._cfg.topic, _callback, self._cfg.subscription_expression)  # type: ignore
        except TypeError:
            consumer.subscribe(self._cfg.topic, _callback)

        consumer.start()
        logger.info("rocketmq consumer started: %s %s %s",self._cfg.namesrv_addr, self._cfg.topic, self._cfg.consumer_group)

        try:
            await self._stop_evt.wait()
        finally:
            try:
                consumer.shutdown()
            except Exception:
                logger.exception("rocketmq shutdown error")

    def stop(self) -> None:
        self._stop_evt.set()
