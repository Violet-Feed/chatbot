from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, List

import orjson
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from chatbot.dal.mysql.memory_service import GlossaryItem
from chatbot.agent import prompts
from chatbot.settings import Settings


@dataclass(frozen=True)
class DecisionInputs:
    agents_json: str
    recent_messages_json: str
    short_summary: str
    long_summary: str


@dataclass(frozen=True)
class DecisionOutput:
    respond: bool
    primary_agent_id: Optional[int]
    reason: str = ""


@dataclass(frozen=True)
class ShortMemoryInputs:
    old_short_summary: str
    recent_messages: str


@dataclass(frozen=True)
class LongMemoryInputs:
    old_long_summary: str
    recent_messages: str


@dataclass(frozen=True)
class GlossaryExtractInputs:
    recent_messages: str


@dataclass(frozen=True)
class GlossaryInferInputs:
    terms_json: str
    recent_messages: str
    short_summary: str
    long_summary: str


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


def _extract_json_array(text: str) -> Optional[str]:
    if not text:
        return None

    s = text.strip()

    if s.startswith("```"):
        s = s.strip("`")
        s = s.replace("json", "", 1).strip()

    l = s.find("[")
    r = s.rfind("]")
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

    def get_chat_model(self) -> ChatOpenAI:
        return self.chat

    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput:
        system_text = prompts.DECISION_SYSTEM
        user_text = _render(
            prompts.DECISION_USER,
            agents_json=inputs.agents_json or "[]",
            recent_messages_json=inputs.recent_messages_json or "[]",
            short_summary=inputs.short_summary or "",
            long_summary=inputs.long_summary or "",
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

    async def update_short_memory(self, inputs: ShortMemoryInputs) -> str:
        user_text = _render(
            prompts.SHORT_MEMORY_USER,
            old_short_summary=inputs.old_short_summary or "",
            recent_messages=inputs.recent_messages or "",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=prompts.SHORT_MEMORY_SYSTEM),
                HumanMessage(content=user_text),
            ]
        )
        return (getattr(msg, "content", "") or "").strip()

    async def update_long_memory(self, inputs: LongMemoryInputs) -> str:
        user_text = _render(
            prompts.LONG_MEMORY_USER,
            old_long_summary=inputs.old_long_summary or "",
            recent_messages=inputs.recent_messages or "",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=prompts.LONG_MEMORY_SYSTEM),
                HumanMessage(content=user_text),
            ]
        )
        return (getattr(msg, "content", "") or "").strip()

    async def extract_unknown_terms(self, inputs: GlossaryExtractInputs) -> List[str]:
        user_text = _render(
            prompts.GLOSSARY_EXTRACT_USER,
            recent_messages=inputs.recent_messages or "",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=prompts.GLOSSARY_EXTRACT_SYSTEM),
                HumanMessage(content=user_text),
            ]
        )
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_array(raw)
        if not payload:
            return []

        arr = orjson.loads(payload)
        if not isinstance(arr, list):
            return []

        out: List[str] = []
        for item in arr:
            term = str(item or "").strip()
            if term:
                out.append(term)
        return out

    async def infer_glossary_meanings(self, inputs: GlossaryInferInputs) -> List[GlossaryItem]:
        user_text = _render(
            prompts.GLOSSARY_INFER_USER,
            terms_json=inputs.terms_json or "[]",
            recent_messages=inputs.recent_messages or "",
            short_summary=inputs.short_summary or "",
            long_summary=inputs.long_summary or "",
        )

        msg = await self.chat.ainvoke(
            [
                SystemMessage(content=prompts.GLOSSARY_INFER_SYSTEM),
                HumanMessage(content=user_text),
            ]
        )
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_array(raw)
        if not payload:
            return []

        arr = orjson.loads(payload)
        if not isinstance(arr, list):
            return []

        out: List[GlossaryItem] = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "") or "").strip()
            meaning = str(item.get("meaning", "") or "").strip()
            if not term or not meaning:
                continue
            out.append(GlossaryItem(term=term, meaning=meaning, count=1))
        return out
