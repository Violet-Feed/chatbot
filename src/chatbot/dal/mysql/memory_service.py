# src/chatbot/dal/mysql/memory_service.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from chatbot.dal.mysql.models import (
    ConversationEpisodicMemory,
    ConversationRollingSummary,
    ConversationStyleFingerprint,
)
from chatbot.utils.time import now_ms


class RollingSummaryService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def load(self, con_short_id: int) -> Tuple[str, int]:
        """返回 (summary_text, last_con_index)。"""
        async with self._sf() as ses:
            row = await ses.get(ConversationRollingSummary, con_short_id)
            if not row:
                return "", 0
            return row.summary, int(row.last_con_index)

    async def upsert(self, con_short_id: int, summary: str, last_con_index: int) -> None:
        """
        写入/更新滚动摘要。
        约束：last_con_index 只能前进不能回退（避免并发/乱序导致摘要覆盖）。
        """
        async with self._sf() as ses:
            row = await ses.get(ConversationRollingSummary, con_short_id)
            ts = now_ms()
            if not row:
                ses.add(
                    ConversationRollingSummary(
                        con_short_id=con_short_id,
                        summary=summary,
                        last_con_index=int(last_con_index),
                        updated_at=ts,
                    )
                )
            else:
                # 防回退：如果传入更小的 last_con_index，则不更新
                if int(last_con_index) >= int(row.last_con_index):
                    row.summary = summary
                    row.last_con_index = int(last_con_index)
                    row.updated_at = ts
            await ses.commit()


class EpisodicMemoryService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def load_active(self, con_short_id: int, now_ms_value: int) -> List[Dict[str, Any]]:
        """加载未过期的事件记忆，按 importance 降序。"""
        async with self._sf() as ses:
            # expire_at=0 表示不过期
            stmt1 = (
                select(ConversationEpisodicMemory)
                .where(ConversationEpisodicMemory.con_short_id == con_short_id)
                .where(ConversationEpisodicMemory.expire_at == 0)
                .order_by(ConversationEpisodicMemory.importance.desc())
                .limit(50)
            )
            rows1 = (await ses.execute(stmt1)).scalars().all()

            stmt2 = (
                select(ConversationEpisodicMemory)
                .where(ConversationEpisodicMemory.con_short_id == con_short_id)
                .where(ConversationEpisodicMemory.expire_at > now_ms_value)
                .order_by(ConversationEpisodicMemory.importance.desc())
                .limit(50)
            )
            rows2 = (await ses.execute(stmt2)).scalars().all()

        rows = list(rows1) + [r for r in rows2 if r not in rows1]
        return [
            {
                "id": int(r.id),
                "type": r.event_type,
                "content": r.content,
                "importance": float(r.importance),
                "expire_at": int(r.expire_at),
                "created_at": int(r.created_at),
            }
            for r in rows
        ]

    async def add(
        self,
        con_short_id: int,
        event_type: str,
        content: str,
        importance: float = 0.5,
        ttl_ms: int = 0,
    ) -> None:
        """新增事件记忆。ttl_ms=0 表示不过期。"""
        ts = now_ms()
        expire_at = 0 if ttl_ms <= 0 else ts + int(ttl_ms)
        row = ConversationEpisodicMemory(
            con_short_id=con_short_id,
            event_type=event_type,
            content=content,
            importance=float(importance),
            expire_at=expire_at,
            created_at=ts,
        )
        async with self._sf() as ses:
            ses.add(row)
            await ses.commit()

    async def add_many(self, con_short_id: int, events: List[Dict[str, Any]]) -> int:
        """
        批量新增事件记忆（减少 commit）。
        events 每项：
          {"type":..., "content":..., "importance":0~1, "ttl_ms":0|ms}
        返回插入条数。
        """
        if not events:
            return 0

        ts = now_ms()
        rows: List[ConversationEpisodicMemory] = []
        for e in events:
            content = str(e.get("content", "") or "").strip()
            if not content:
                continue
            ttl_ms = int(e.get("ttl_ms", 0) or 0)
            expire_at = 0 if ttl_ms <= 0 else ts + ttl_ms
            rows.append(
                ConversationEpisodicMemory(
                    con_short_id=con_short_id,
                    event_type=str(e.get("type", "generic") or "generic"),
                    content=content,
                    importance=float(e.get("importance", 0.5) or 0.5),
                    expire_at=int(expire_at),
                    created_at=int(ts),
                )
            )

        if not rows:
            return 0

        async with self._sf() as ses:
            ses.add_all(rows)
            await ses.commit()
        return len(rows)

    async def purge_expired(self, con_short_id: int, now_ms_value: int) -> int:
        """清理过期事件记忆，返回删除行数。"""
        async with self._sf() as ses:
            stmt = (
                delete(ConversationEpisodicMemory)
                .where(ConversationEpisodicMemory.con_short_id == con_short_id)
                .where(ConversationEpisodicMemory.expire_at > 0)
                .where(ConversationEpisodicMemory.expire_at <= now_ms_value)
            )
            res = await ses.execute(stmt)
            await ses.commit()
            return int(res.rowcount or 0)

    async def purge_expired_all(self, now_ms_value: int) -> int:
        """可选：全局清理过期事件记忆（可由定时任务调用）。"""
        async with self._sf() as ses:
            stmt = (
                delete(ConversationEpisodicMemory)
                .where(ConversationEpisodicMemory.expire_at > 0)
                .where(ConversationEpisodicMemory.expire_at <= now_ms_value)
            )
            res = await ses.execute(stmt)
            await ses.commit()
            return int(res.rowcount or 0)


class StyleService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    async def load(self, con_short_id: int) -> Dict[str, Any]:
        """返回风格特征（features 字符串 + updated_at）。"""
        async with self._sf() as ses:
            row = await ses.get(ConversationStyleFingerprint, con_short_id)
            if not row:
                return {}
            return self._parse_features(row.features, int(row.updated_at))

    async def upsert(self, con_short_id: int, features: str) -> None:
        """写入/更新风格指纹。"""
        ts = now_ms()
        async with self._sf() as ses:
            row = await ses.get(ConversationStyleFingerprint, con_short_id)
            if not row:
                ses.add(ConversationStyleFingerprint(con_short_id=con_short_id, features=features, updated_at=ts))
            else:
                row.features = features
                row.updated_at = ts
            await ses.commit()

    async def merge_update(
        self,
        con_short_id: int,
        update: Dict[str, Any],
        max_style_rules: int = 20,
        max_hotwords: int = 50,
        max_glossary: int = 50,
    ) -> None:
        """合并更新风格规则与热词（去重、保留最新）。"""
        ts = now_ms()
        async with self._sf() as ses:
            row = await ses.get(ConversationStyleFingerprint, con_short_id)

            base = {}
            if row and row.features:
                base = self._parse_features(row.features, int(row.updated_at or ts))

            merged = self._merge_features(
                base,
                update,
                max_style_rules=max_style_rules,
                max_hotwords=max_hotwords,
                max_glossary=max_glossary,
            )

            if not row:
                ses.add(
                    ConversationStyleFingerprint(
                        con_short_id=con_short_id,
                        features=json.dumps(merged, ensure_ascii=False, separators=(",", ":")),
                        updated_at=ts,
                    )
                )
            else:
                row.features = json.dumps(merged, ensure_ascii=False, separators=(",", ":"))
                row.updated_at = ts
            await ses.commit()

    def _parse_features(self, raw: str, updated_at: int) -> Dict[str, Any]:
        try:
            obj = json.loads(raw or "{}")
        except Exception:
            obj = {}
        if not isinstance(obj, dict):
            obj = {}

        style_rules = obj.get("style_rules")
        hotwords = obj.get("hotwords")
        glossary = obj.get("glossary")

        return {
            "style_rules": style_rules if isinstance(style_rules, list) else [],
            "hotwords": hotwords if isinstance(hotwords, list) else [],
            "glossary": glossary if isinstance(glossary, list) else [],
            "updated_at": int(updated_at),
        }

    def _merge_features(
        self,
        base: Dict[str, Any],
        update: Dict[str, Any],
        max_style_rules: int,
        max_hotwords: int,
        max_glossary: int,
    ) -> Dict[str, Any]:
        def _norm(s: str) -> str:
            return str(s or "").strip().casefold()

        base_rules = [str(x).strip() for x in (base.get("style_rules") or []) if str(x).strip()]
        new_rules = [str(x).strip() for x in (update.get("style_rules") or []) if str(x).strip()]

        rule_seen = set()
        merged_rules: List[str] = []
        for s in new_rules + base_rules:
            key = _norm(s)
            if not key or key in rule_seen:
                continue
            rule_seen.add(key)
            merged_rules.append(s)
            if len(merged_rules) >= max_style_rules:
                break

        base_hot = [str(x).strip() for x in (base.get("hotwords") or []) if str(x).strip()]
        new_hot = [str(x).strip() for x in (update.get("hotwords") or []) if str(x).strip()]

        hot_seen = set()
        merged_hot: List[str] = []
        for s in new_hot + base_hot:
            key = _norm(s)
            if not key or key in hot_seen:
                continue
            hot_seen.add(key)
            merged_hot.append(s)
            if len(merged_hot) >= max_hotwords:
                break

        base_glossary = base.get("glossary") or []
        new_glossary = update.get("glossary") or []
        merged_glossary: List[Dict[str, Any]] = []
        glossary_seen = set()
        for it in (new_glossary + base_glossary):
            if not isinstance(it, dict):
                continue
            term = str(it.get("term", "") or "").strip()
            meaning = str(it.get("meaning", "") or "").strip()
            if not term or not meaning:
                continue
            key = _norm(term)
            if key in glossary_seen:
                continue
            glossary_seen.add(key)
            merged_glossary.append(
                {
                    "term": term,
                    "meaning": meaning,
                    "source": str(it.get("source", "") or "").strip(),
                }
            )
            if len(merged_glossary) >= max_glossary:
                break

        return {
            "style_rules": merged_rules,
            "hotwords": merged_hot,
            "glossary": merged_glossary,
            "updated_at": int(now_ms()),
        }
