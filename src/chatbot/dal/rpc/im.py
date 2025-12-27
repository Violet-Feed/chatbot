# src/chatbot/dal/rpc/im.py
from __future__ import annotations

import uuid
from typing import List, Optional

import grpc

from chatbot.proto_gen import common_pb2, im_pb2, im_pb2_grpc


def _new_client_msg_id() -> int:
    """生成 client_msg_id（正 int64，避免冲突）。"""
    return uuid.uuid4().int & 0x7FFF_FFFF_FFFF_FFFF


def _ensure_success(base_resp: common_pb2.BaseResp) -> None:
    """
    严格按 common.BaseResp 判断成功与否：
    - 成功：StatusCode == Success(1000)
    - 失败：抛异常，携带 StatusMessage
    """
    if base_resp.StatusCode != common_pb2.Success:
        raise RuntimeError(
            f"IMService failed: StatusCode={int(base_resp.StatusCode)} "
            f"StatusMessage={base_resp.StatusMessage}"
        )


class IMClient:
    """
    IMService gRPC 客户端封装（grpc.aio）。
    说明：
    - 这里不做任何 BaseResp 字段兼容，只按你 common.proto 的字段名读取。
    """

    def __init__(self, target: str, timeout_sec: float = 3.0) -> None:
        self._target = target
        self._timeout = float(timeout_sec)
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = im_pb2_grpc.IMServiceStub(self._channel)

    async def aclose(self) -> None:
        await self._channel.close()

    async def send_message(
        self,
        sender_id: int,
        sender_type: int,
        con_short_id: int,
        con_id: str,
        con_type: int,
        msg_type: int,
        msg_content: str,
        client_msg_id: Optional[int] = None,
    ) -> int:
        """调用 IMService.SendMessage 发送消息，返回 msg_id。"""
        if client_msg_id is None:
            client_msg_id = _new_client_msg_id()

        req = im_pb2.SendMessageRequest(
            sender_id=int(sender_id),
            sender_type=int(sender_type),
            con_short_id=int(con_short_id),
            con_id=str(con_id),
            con_type=int(con_type),
            client_msg_id=int(client_msg_id),
            msg_type=int(msg_type),
            msg_content=str(msg_content),
        )
        resp: im_pb2.SendMessageResponse = await self._stub.SendMessage(req, timeout=self._timeout)
        _ensure_success(resp.baseResp)
        return int(resp.msg_id)

    async def get_message_by_conversation(
        self,
        user_id: int,
        con_short_id: int,
        con_index: int,
        limit: int,
    ) -> List[im_pb2.MessageBody]:
        """拉取某群会话消息（用于上下文补齐）。"""
        req = im_pb2.GetMessageByConversationRequest(
            user_id=int(user_id),
            con_short_id=int(con_short_id),
            con_index=int(con_index),
            limit=int(limit),
        )
        resp: im_pb2.GetMessageByConversationResponse = await self._stub.GetMessageByConversation(
            req, timeout=self._timeout
        )
        _ensure_success(resp.baseResp)
        return list(resp.msg_bodies)

    async def get_conversation_agents(self, con_short_id: int) -> List[im_pb2.ConversationAgentInfo]:
        """
        获取群内 agent 列表（从 IM 服务取，不走数据库）。
        proto 约定：
          GetConversationAgentsRequest{ con_short_id }
          GetConversationAgentsResponse{ repeated ConversationAgentInfo agents, BaseResp baseResp }
        且 ConversationAgentInfo.agent_info 会携带 AgentInfo（包含 personality/description/avatar 等）。
        """
        req = im_pb2.GetConversationAgentsRequest(con_short_id=int(con_short_id))
        resp: im_pb2.GetConversationAgentsResponse = await self._stub.GetConversationAgents(
            req, timeout=self._timeout
        )
        _ensure_success(resp.baseResp)
        return list(resp.agents)
