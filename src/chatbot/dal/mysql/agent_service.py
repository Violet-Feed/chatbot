from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sqlalchemy import text, bindparam
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@dataclass(frozen=True)
class AgentInfo:
    agent_id: int
    agent_name: str
    personality: str


class AgentService:
    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self.sf = sf

    async def get_agents_by_ids(self, agent_ids: List[int]) -> List[AgentInfo]:
        if not agent_ids:
            return []

        sql = text(
            """
            SELECT agent_id,
                   agent_name,
                   personality
            FROM agent
            WHERE agent_id IN :agent_ids
            """
        ).bindparams(bindparam("agent_ids", expanding=True))

        async with self.sf() as session:
            result = await session.execute(
                sql,
                {"agent_ids": tuple(agent_ids)},
            )
            rows = result.mappings().all()

        items: List[AgentInfo] = []
        for row in rows:
            items.append(
                AgentInfo(
                    agent_id=int(row["agent_id"]),
                    agent_name=str(row.get("agent_name") or ""),
                    personality=str(row.get("personality") or ""),
                )
            )
        return items