from __future__ import annotations

from typing import List

import grpc
from chatbot.proto_gen import aigc_pb2, aigc_pb2_grpc, common_pb2


def _ensure_success(base_resp: common_pb2.BaseResp) -> None:
    if base_resp.StatusCode != common_pb2.Success:
        raise RuntimeError(
            f"AigcService failed: StatusCode={int(base_resp.StatusCode)} "
            f"StatusMessage={base_resp.StatusMessage}"
        )


class AigcClient:
    def __init__(self, target: str, timeout_sec: float = 3.0) -> None:
        self._target = target
        self._timeout = float(timeout_sec)
        self._channel = grpc.aio.insecure_channel(target)
        self._stub = aigc_pb2_grpc.AigcServiceStub(self._channel)

    async def aclose(self) -> None:
        await self._channel.close()

    async def get_agents_by_ids(self, agent_ids: List[int]) -> List[aigc_pb2.AgentInfo]:
        req = aigc_pb2.GetAgentsByIdsRequest(
            agent_ids=[int(a) for a in agent_ids if int(a or 0) > 0]
        )
        resp: aigc_pb2.GetAgentsByIdsResponse = await self._stub.GetAgentsByIds(
            req, timeout=self._timeout
        )
        _ensure_success(resp.baseResp)
        return list(resp.agent_infos)
