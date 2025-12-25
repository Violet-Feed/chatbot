from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chatbot.planner.triggers import hit_strong, mentioned_agent


@dataclass(frozen=True)
class Decision:
    """窗口内 0~2 bot 决策结果。"""
    primary_id: Optional[int]
    secondary_id: Optional[int]
    reason1: str = ""
    reason2: str = ""


@dataclass(frozen=True)
class AgentView:
    """
    决策层只需要的 agent 视图，避免和你 DAL 强绑定。
    你可从 AgentProfile 映射生成。
    """
    agent_id: int
    name: str
    aliases: Sequence[str] = ()
    activeness: float = 0.5


def decide_0_2_agents(
    agents: List[AgentView],
    window_items: List[Dict[str, Any]],
    bot2bot_strict: bool = True,
) -> Decision:
    """
    纯决策（不读 Redis，不做 token/cooldown），只根据消息触发判断 0~2 个 bot：
    - primary：必须强触发
    - secondary：必须强触发 + 与 primary 不同
    - bot↔bot 严格：若最后一条是 bot，则 secondary 必须在最后一条里被 mentions（显式 @）
    """
    if not agents or not window_items:
        return Decision(None, None)

    strong: Dict[int, bool] = {a.agent_id: False for a in agents}

    for it in window_items:
        text = str(it.get("msg_content", "") or "")
        mentions = (it.get("extra") or {}).get("mentions") or []
        for a in agents:
            names = (a.name, *list(a.aliases))
            if hit_strong(a.agent_id, names, text, mentions):
                strong[a.agent_id] = True

    strong_candidates = [a for a in agents if strong.get(a.agent_id)]
    if not strong_candidates:
        return Decision(None, None)

    primary = random.choice(strong_candidates)

    second_candidates = [a for a in strong_candidates if a.agent_id != primary.agent_id]
    if not second_candidates:
        return Decision(primary.agent_id, None, "STRONG", "")

    last = window_items[-1]
    if bot2bot_strict and bool(last.get("is_bot", False)):
        last_mentions = (last.get("extra") or {}).get("mentions") or []
        explicit = [a for a in second_candidates if mentioned_agent(a.agent_id, last_mentions)]
        if not explicit:
            return Decision(primary.agent_id, None, "STRONG", "")
        secondary = random.choice(explicit)
        return Decision(primary.agent_id, secondary.agent_id, "STRONG", "BOT2BOT_STRICT")

    secondary = random.choice(second_candidates)
    return Decision(primary.agent_id, secondary.agent_id, "STRONG", "STRONG")
