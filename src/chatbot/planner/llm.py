from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import orjson
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from chatbot.planner import prompts
from chatbot.settings import Settings


@dataclass(frozen=True)
class ReplyInputs:
    agent_id: int
    agent_name: str
    agent_personality: str
    trigger_reason: str
    recent_messages: str
    group_rules: str = ""


@dataclass(frozen=True)
class DecisionInputs:
    agents_json: str
    recent_messages_json: str
    last_message_json: str
    group_rules: str = ""


@dataclass(frozen=True)
class DecisionOutput:
    respond: bool
    primary_agent_id: Optional[int]
    reason: str = ""


def _render(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)


def _extract_json_object(text: str) -> Optional[str]:
    if not text:
        return None

    s = text.strip()

    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()

    l = s.find("{")
    r = s.rfind("}")
    if l == -1 or r == -1 or r <= l:
        return None
    return s[l : r + 1]


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.s = settings
        api_key = settings.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("缺少 DASHSCOPE_API_KEY")

        kwargs = dict(
            base_url=settings.QWEN_BASE_URL.rstrip("/"),
            api_key=api_key,
            model=settings.QWEN_MODEL,
            temperature=float(settings.QWEN_TEMPERATURE),
            timeout=int(settings.QWEN_TIMEOUT_SEC),
            max_retries=2,
        )

        try:
            self.chat = ChatOpenAI(**kwargs, use_responses_api=False)  # type: ignore[arg-type]
        except TypeError:
            self.chat = ChatOpenAI(**kwargs)

    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput:
        system_text = prompts.DECISION_SYSTEM
        user_text = _render(
            prompts.DECISION_USER,
            agents_json=inputs.agents_json or "[]",
            recent_messages_json=inputs.recent_messages_json or "[]",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]
        )
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return DecisionOutput(respond=False, primary_agent_id=None, reason="bad_json")

        obj = orjson.loads(payload)
        if not isinstance(obj, dict):
            return DecisionOutput(respond=False, primary_agent_id=None, reason="bad_json")

        respond = bool(obj.get("respond", False))

        primary = obj.get("primary_agent_id")
        primary_agent_id = int(primary) if primary is not None else None

        reason = str(obj.get("reason", "") or "")

        return DecisionOutput(
            respond=respond,
            primary_agent_id=primary_agent_id,
            reason=reason,
        )

    async def generate_reply(self, inputs: ReplyInputs) -> str:
        system_text = _render(
            prompts.REPLY_SYSTEM,
            agent_id=inputs.agent_id,
            agent_name=inputs.agent_name,
            personality=inputs.agent_personality or "",
            intent="自然参与对话并直接回答",
            trigger_reason=inputs.trigger_reason or "",
        )
        user_text = _render(
            prompts.REPLY_USER,
            recent_messages=inputs.recent_messages or "",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=system_text),
                HumanMessage(content=user_text),
            ]
        )
        return (getattr(msg, "content", "") or "").strip()