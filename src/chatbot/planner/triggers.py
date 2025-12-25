from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


_STRONG_VERBS = ("怎么看", "说说", "解释", "讲讲", "回答", "评价", "建议", "帮我")


def mentioned_agent(agent_id: int, mentions: Any) -> bool:
    """
    mentions 约定：[{type:'agent', id:xxx}, ...]
    """
    if not isinstance(mentions, list):
        return False
    for m in mentions:
        if isinstance(m, dict) and m.get("type") == "agent":
            try:
                if int(m.get("id", -1)) == int(agent_id):
                    return True
            except Exception:
                continue
    return False


def hit_strong(agent_id: int, names: Sequence[str], text: str, mentions: Any) -> bool:
    """
    强触发：
    - 显式 mentions @agent_id
    - 或 @name/@alias
    - 或 点名 + 指令动词（怎么看/说说/解释/建议…）
    """
    if mentioned_agent(agent_id, mentions):
        return True

    for nm in names:
        if nm and f"@{nm}" in text:
            return True

    if any(nm and nm in text for nm in names):
        if any(v in text for v in _STRONG_VERBS):
            return True

    return False


def hit_weak(names: Sequence[str], text: str) -> bool:
    """弱触发：仅名字出现（用于你未来加概率门/主题门）。"""
    return any(nm and nm in text for nm in names)
