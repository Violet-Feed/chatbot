from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EpisodicEvent:
    """
    事件记忆条目（上层可落库，带 TTL）。
    importance: 0~1
    ttl_ms: 0 表示不过期
    """
    event_type: str
    content: str
    importance: float = 0.5
    ttl_ms: int = 0


_TIME_PAT = re.compile(r"(\d{1,2}[:：]\d{2})|((今天|明天|后天|周[一二三四五六日天]|下周|本周))")
_PREF_PAT = re.compile(r"(我(喜欢|不喜欢|讨厌|爱|想要|不要).{0,12})")
_RULE_PAT = re.compile(r"(群里(不要|别|禁止).{0,16})")


def _norm_text(x: Any) -> str:
    return str(x or "").replace("\n", " ").strip()


def extract_episodic_rule_based(window_items: Sequence[Dict[str, Any]], max_events: int = 10) -> List[EpisodicEvent]:
    """
    规则抽取（无 LLM 版本）：
    - 时间约定（今天/明天/xx:xx）
    - 偏好/禁忌（我喜欢/我不喜欢...）
    - 群规则（群里不要...）
    """
    out: List[EpisodicEvent] = []
    for it in window_items[-120:]:
        txt = _norm_text(it.get("msg_content"))
        if not txt:
            continue

        if _RULE_PAT.search(txt):
            out.append(EpisodicEvent("group_rule", txt, importance=0.8, ttl_ms=0))

        if _PREF_PAT.search(txt):
            out.append(EpisodicEvent("preference", txt, importance=0.6, ttl_ms=7 * 24 * 3600 * 1000))

        if _TIME_PAT.search(txt):
            out.append(EpisodicEvent("schedule", txt, importance=0.7, ttl_ms=3 * 24 * 3600 * 1000))

        if len(out) >= max_events:
            break
    return out


async def extract_episodic_llm(
    llm_extract,  # async (messages_text) -> list[dict]
    window_items: Sequence[Dict[str, Any]],
    max_events: int = 10,
) -> List[EpisodicEvent]:
    """
    LLM 抽取：llm_extract 由上层注入（可复用 planner/llm.py）。
    约定 llm_extract 返回结构化 list[dict]，每个 dict 至少有 content/type/importance/ttl_ms。
    """
    messages = []
    for it in window_items[-120:]:
        who = "BOT" if it.get("is_bot") else "USER"
        sid = it.get("sender_id", 0)
        txt = _norm_text(it.get("msg_content"))
        if txt:
            messages.append(f"[{who}:{sid}] {txt}")
    payload = "\n".join(messages)

    raw = await llm_extract(payload)
    out: List[EpisodicEvent] = []
    if isinstance(raw, list):
        for d in raw[:max_events]:
            if not isinstance(d, dict):
                continue
            content = _norm_text(d.get("content"))
            if not content:
                continue
            out.append(
                EpisodicEvent(
                    event_type=_norm_text(d.get("type") or "generic"),
                    content=content,
                    importance=float(d.get("importance", 0.5) or 0.5),
                    ttl_ms=int(d.get("ttl_ms", 0) or 0),
                )
            )
    return out
