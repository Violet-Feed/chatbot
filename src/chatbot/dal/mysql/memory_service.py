from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatbot.utils.time import now_ms


@dataclass(frozen=True)
class GlossaryItem:
    term: str
    meaning: str
    count: int = 1


@dataclass(frozen=True)
class SummaryState:
    short_summary: str
    long_summary: str
    short_version: int
    long_version: int
    short_updated_at: int
    long_updated_at: int
    updated_at: int


class MemoryService:
    def __init__(self, sf: async_sessionmaker[AsyncSession]) -> None:
        self.sf = sf

    async def get_summary_state(self, con_short_id: int) -> SummaryState:
        sql = text(
            """
            SELECT short_summary,
                   long_summary,
                   short_version,
                   long_version,
                   short_updated_at,
                   long_updated_at,
                   updated_at
            FROM chatbot_memory_summary
            WHERE con_short_id = :con_short_id
            """
        )

        async with self.sf() as session:
            result = await session.execute(sql, {"con_short_id": con_short_id})
            row = result.mappings().first()

        if not row:
            return SummaryState(
                short_summary="",
                long_summary="",
                short_version=0,
                long_version=0,
                short_updated_at=0,
                long_updated_at=0,
                updated_at=0,
            )

        return SummaryState(
            short_summary=str(row.get("short_summary") or ""),
            long_summary=str(row.get("long_summary") or ""),
            short_version=int(row.get("short_version") or 0),
            long_version=int(row.get("long_version") or 0),
            short_updated_at=int(row.get("short_updated_at") or 0),
            long_updated_at=int(row.get("long_updated_at") or 0),
            updated_at=int(row.get("updated_at") or 0),
        )

    async def save_short_summary(self, con_short_id: int, short_summary: str) -> None:
        ts = now_ms()
        sql = text(
            """
            INSERT INTO chatbot_memory_summary (con_short_id,
                                                short_summary,
                                                long_summary,
                                                short_version,
                                                long_version,
                                                short_updated_at,
                                                long_updated_at,
                                                updated_at)
            VALUES (:con_short_id,
                    :short_summary,
                    '',
                    1,
                    0,
                    :ts,
                    0,
                    :ts) AS new
            ON DUPLICATE KEY
            UPDATE
                short_summary = new.short_summary,
                short_version = chatbot_memory_summary.short_version + 1,
                short_updated_at = new.short_updated_at,
                updated_at = new.updated_at
            """
        )

        async with self.sf() as session:
            await session.execute(
                sql,
                {
                    "con_short_id": con_short_id,
                    "short_summary": short_summary or "",
                    "ts": ts,
                },
            )
            await session.commit()

    async def save_long_summary(self, con_short_id: int, long_summary: str) -> None:
        ts = now_ms()
        sql = text(
            """
            INSERT INTO chatbot_memory_summary (con_short_id,
                                                short_summary,
                                                long_summary,
                                                short_version,
                                                long_version,
                                                short_updated_at,
                                                long_updated_at,
                                                updated_at)
            VALUES (:con_short_id,
                    '',
                    :long_summary,
                    0,
                    1,
                    0,
                    :ts,
                    :ts) AS new
            ON DUPLICATE KEY
            UPDATE
                long_summary = new.long_summary,
                long_version = chatbot_memory_summary.long_version + 1,
                long_updated_at = new.long_updated_at,
                updated_at = new.updated_at
            """
        )

        async with self.sf() as session:
            await session.execute(
                sql,
                {
                    "con_short_id": con_short_id,
                    "long_summary": long_summary or "",
                    "ts": ts,
                },
            )
            await session.commit()

    async def upsert_glossary_terms(self, con_short_id: int, terms: Iterable[str]) -> None:
        rows = []
        ts = now_ms()

        for term in terms:
            t = str(term or "").strip()
            if not t:
                continue
            rows.append(
                {
                    "con_short_id": con_short_id,
                    "term": t[:64],
                    "updated_at": ts,
                }
            )

        if not rows:
            return

        sql = text(
            """
            INSERT INTO chatbot_memory_glossary (con_short_id,
                                                 term,
                                                 meaning,
                                                 `count`,
                                                 updated_at)
            VALUES (:con_short_id,
                    :term,
                    '',
                    1,
                    :updated_at) AS new
            ON DUPLICATE KEY
            UPDATE
                `count` = chatbot_memory_glossary.`count` + 1,
                updated_at = new.updated_at
            """
        )

        async with self.sf() as session:
            await session.execute(sql, rows)
            await session.commit()

    async def get_terms_need_meaning(
            self,
            con_short_id: int,
            min_count: int = 5,
            limit: int = 20,
    ) -> List[str]:
        sql = text(
            """
            SELECT term
            FROM chatbot_memory_glossary
            WHERE con_short_id = :con_short_id
              AND `count` >= :min_count
              AND meaning = '' LIMIT :limit_n
            """
        )

        async with self.sf() as session:
            result = await session.execute(
                sql,
                {
                    "con_short_id": con_short_id,
                    "min_count": min_count,
                    "limit_n": limit,
                },
            )
            rows = result.mappings().all()

        return [
            str(row.get("term") or "")
            for row in rows
            if str(row.get("term") or "").strip()
        ]

    async def get_terms_need_reinference(
            self,
            con_short_id: int,
            min_count: int = 5,
            step: int = 10,
            limit: int = 20,
    ) -> List[GlossaryItem]:
        sql = text(
            """
            SELECT term, meaning, `count`
            FROM chatbot_memory_glossary
            WHERE con_short_id = :con_short_id
              AND `count` >= :min_count
              AND meaning <> ''
              AND MOD(`count` - :min_count, :step) = 0 LIMIT :limit_n
            """
        )

        async with self.sf() as session:
            result = await session.execute(
                sql,
                {
                    "con_short_id": con_short_id,
                    "min_count": min_count,
                    "step": step,
                    "limit_n": limit,
                },
            )
            rows = result.mappings().all()

        return [
            GlossaryItem(
                term=str(row.get("term") or ""),
                meaning=str(row.get("meaning") or ""),
                count=int(row.get("count") or 0),
            )
            for row in rows
        ]

    async def save_glossary_meanings(
            self,
            con_short_id: int,
            items: Iterable[GlossaryItem],
    ) -> None:
        rows = []
        ts = now_ms()

        for item in items:
            term = str(item.term or "").strip()
            meaning = str(item.meaning or "").strip()
            if not term or not meaning:
                continue
            rows.append(
                {
                    "con_short_id": con_short_id,
                    "term": term[:64],
                    "meaning": meaning[:255],
                    "updated_at": ts,
                }
            )

        if not rows:
            return

        sql = text(
            """
            INSERT INTO chatbot_memory_glossary (con_short_id,
                                                 term,
                                                 meaning,
                                                 `count`,
                                                 updated_at)
            VALUES (:con_short_id,
                    :term,
                    :meaning,
                    1,
                    :updated_at) AS new
            ON DUPLICATE KEY
            UPDATE
                meaning = new.meaning,
                updated_at = new.updated_at
            """
        )

        async with self.sf() as session:
            await session.execute(sql, rows)
            await session.commit()

    async def get_relevant_glossary(
            self,
            con_short_id: int,
            recent_text: str,
            limit: int = 20,
    ) -> List[GlossaryItem]:
        recent_text = str(recent_text or "").strip()

        if recent_text:
            sql = text(
                """
                SELECT term, meaning, `count`
                FROM chatbot_memory_glossary
                WHERE con_short_id = :con_short_id
                  AND meaning <> ''
                  AND INSTR(:recent_text, term) > 0
                ORDER BY term ASC LIMIT :limit_n
                """
            )
            params = {
                "con_short_id": con_short_id,
                "recent_text": recent_text,
                "limit_n": limit,
            }
        else:
            sql = text(
                """
                SELECT term, meaning, `count`
                FROM chatbot_memory_glossary
                WHERE con_short_id = :con_short_id
                  AND meaning <> '' LIMIT :limit_n
                """
            )
            params = {
                "con_short_id": con_short_id,
                "limit_n": limit,
            }

        async with self.sf() as session:
            result = await session.execute(sql, params)
            rows = result.mappings().all()

        return [
            GlossaryItem(
                term=str(row.get("term") or ""),
                meaning=str(row.get("meaning") or ""),
                count=int(row.get("count") or 0),
            )
            for row in rows
        ]
