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


def _parse_extra(x: Any) -> str:
    """
    proto 的 extra 是 string（你当前 proto 如此），MQ 里 extra 可能是 object 或 string。
    这里统一转成 string（若是 dict/list 就 JSON dump）。
    """
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return orjson.dumps(x).decode("utf-8")
    except Exception:
        return str(x)


def parse_message_event_json(raw: bytes) -> im_pb2.MessageEvent:
    """
    MQ body 是 JSON；构造 proto MessageEvent 并返回。
    兼容两种 sender 字段：
    - 新：sender_id + sender_type
    - 旧：user_id -> sender_id, sender_type=User
    """
    obj = orjson.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("MQ JSON must be an object")

    body = obj.get("msg_body") or obj.get("msgBody") or {}
    if not isinstance(body, dict):
        raise ValueError("msg_body must be an object")

    # sender 兼容
    sender_id = body.get("sender_id")
    sender_type = body.get("sender_type")
    if sender_id is None:
        sender_id = body.get("user_id", 0)
        sender_type = 1  # user (约定：1 user, 2 agent, 3 system；按你最终 enum 映射调整)

    # con_index 兼容（可能在外层/内层）
    con_index = body.get("con_index", obj.get("con_index", 0)) or 0

    msg_body = im_pb2.MessageBody(
        # 注意：这里字段名必须与 proto 一致
        # 你 proto 里仍是 user_id 时：先临时写 user_id，等你改 sender_id/sender_type 再改这里
        user_id=int(sender_id or 0),  # 兼容阶段：用 user_id 先承载 sender_id
        con_id=str(body.get("con_id", "") or ""),
        con_short_id=int(body.get("con_short_id", 0) or 0),
        con_type=int(body.get("con_type", 2) or 2),
        client_msg_id=int(body.get("client_msg_id", 0) or 0),
        msg_id=int(body.get("msg_id", 0) or 0),
        msg_type=int(body.get("msg_type", 0) or 0),
        msg_content=str(body.get("msg_content", "") or ""),
        create_time=int(body.get("create_time", 0) or 0),
        extra=_parse_extra(body.get("extra")),
        con_index=int(con_index),
    )

    evt = im_pb2.MessageEvent(
        msg_body=msg_body,
        con_index=int(obj.get("con_index", msg_body.con_index) or msg_body.con_index),
        stored=bool(obj.get("stored", True)),
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

                if isinstance(body, (bytes, bytearray)):
                    raw = bytes(body)
                elif isinstance(body, str):
                    raw = body.encode("utf-8", errors="ignore")
                else:
                    raw = str(body).encode("utf-8", errors="ignore")

                try:
                    evt = parse_message_event_json(raw)
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
        logger.info("rocketmq consumer started: %s %s %s",
                    self._cfg.namesrv_addr, self._cfg.topic, self._cfg.consumer_group)

        try:
            await self._stop_evt.wait()
        finally:
            try:
                consumer.shutdown()
            except Exception:
                logger.exception("rocketmq shutdown error")

    def stop(self) -> None:
        self._stop_evt.set()
