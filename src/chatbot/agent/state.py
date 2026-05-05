from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict


@dataclass(frozen=True)
class Agent:
    agent_id: int
    name: str
    personality: str


class ConversationState(TypedDict):
    con_short_id: int
    con_id: str
    con_type: int
    items: List[Dict[str, Any]]
    history_items: List[Dict[str, Any]]
    # load_context node
    con_name: str
    con_description: str
    agents: List[Agent]
    memory_state: Any
    glossary: List[Any]
    # decide node
    should_reply: bool
    primary_agent: Optional[Agent]
    reply_reason: str
    # generate node
    reply_text: str
