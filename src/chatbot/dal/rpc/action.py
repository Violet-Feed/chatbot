from __future__ import annotations

from typing import Dict, List

import grpc
from chatbot.proto_gen import action_pb2, action_pb2_grpc, common_pb2


def _ensure_success(base_resp: common_pb2.BaseResp) -> None:
    if base_resp.StatusCode != common_pb2.Success:
        raise RuntimeError(
            f"ActionService failed: StatusCode={int(base_resp.StatusCode)} "
            f"StatusMessage={base_resp.StatusMessage}"
        )


class ActionClient:
    def __init__(self, target: str, timeout_sec: float = 3.0) -> None:
        self._target = target
        self._timeout = float(timeout_sec)
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = action_pb2_grpc.ActionServiceStub(self._channel)

    async def aclose(self) -> None:
        await self._channel.close()

    async def get_user_infos(self, user_ids: List[int]) -> Dict[int, action_pb2.UserInfo]:
        req = action_pb2.GetUserInfosRequest(
            user_ids=[int(u) for u in user_ids if int(u or 0) > 0]
        )
        resp: action_pb2.GetUserInfosResponse = await self._stub.GetUserInfos(
            req, timeout=self._timeout
        )
        _ensure_success(resp.baseResp)
        return {int(info.user_id): info for info in resp.user_infos}
