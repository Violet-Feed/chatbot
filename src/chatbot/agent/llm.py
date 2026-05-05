from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, List

import orjson
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from chatbot.agent import prompts
from chatbot.agent.tools import make_fetch_context_tool
from chatbot.dal.mysql.memory_service import GlossaryItem
from chatbot.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionInputs:
    agents_json: str
    current_messages_json: str
    history_messages_json: str
    short_summary: str
    long_summary: str
    con_name: str = ""
    con_description: str = ""
    con_type_label: str = ""


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
    con_name: str = ""
    con_description: str = ""


@dataclass(frozen=True)
class GlossaryExtractInputs:
    recent_messages: str


@dataclass(frozen=True)
class GlossaryInferInputs:
    terms_json: str
    recent_messages: str
    short_summary: str
    long_summary: str
    existing_meanings_json: str = "[]"


@dataclass(frozen=True)
class GenerateInputs:
    agent_name: str
    agent_id: int
    personality: str
    trigger_reason: str
    con_type_label: str
    con_name: str
    con_description: str
    current_messages_json: str
    history_messages_json: str
    short_summary: str
    long_summary: str
    glossary_json: str
    con_short_id: int
    min_con_index: int


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
    return s[l: r + 1]


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
    return s[l: r + 1]


class LLMClient:
    def __init__(self, settings: Settings, base_tools: Optional[List[Any]] = None,
                 im_client: Any = None, max_tool_calls: int = 1) -> None:
        self.s = settings
        self._max_tool_calls = max_tool_calls
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

        self._react_app = None
        if base_tools is not None and im_client is not None:
            self._react_app = create_react_agent(
                self.chat,
                list(base_tools) + [make_fetch_context_tool(im_client=im_client)],
            )

    def get_chat_model(self) -> ChatOpenAI:
        return self.chat

    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput:
        system_text = prompts.DECISION_SYSTEM
        user_text = _render(
            prompts.DECISION_USER,
            con_type_label=inputs.con_type_label or "群聊",
            con_name=inputs.con_name or "",
            con_description=inputs.con_description or "",
            agents_json=inputs.agents_json or "[]",
            current_messages_json=inputs.current_messages_json or "[]",
            history_messages_json=inputs.history_messages_json or "[]",
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

    async def generate_reply(self, inputs: GenerateInputs) -> str:
        system_prompt = prompts.AGENT_SYSTEM.format(
            agent_name=inputs.agent_name,
            agent_id=inputs.agent_id,
            personality=inputs.personality or "",
            trigger_reason=inputs.trigger_reason or "",
        )
        user_prompt = prompts.AGENT_USER.format(
            con_type_label=inputs.con_type_label,
            con_name=inputs.con_name or "",
            con_description=inputs.con_description or "",
            current_messages=inputs.current_messages_json or "[]",
            history_messages=inputs.history_messages_json or "[]",
            short_summary=inputs.short_summary or "",
            long_summary=inputs.long_summary or "",
            glossary_json=inputs.glossary_json or "[]",
        )
        logger.info("generate_reply: user_prompt=%s", user_prompt)

        recursion_limit = self._max_tool_calls * 2 + 3
        try:
            result = await self._react_app.ainvoke(
                {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]},
                config={
                    "recursion_limit": recursion_limit,
                    "configurable": {
                        "con_short_id": inputs.con_short_id,
                        "min_con_index": inputs.min_con_index,
                        "sender_id": inputs.agent_id,
                        "sender_type": 2,
                    },
                },
            )
        except Exception:
            logger.exception("generate_reply failed con_short_id=%s", inputs.con_short_id)
            return ""

        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                return str(msg.content).strip()
        return ""

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
            con_name=inputs.con_name or "",
            con_description=inputs.con_description or "",
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
            existing_meanings_json=inputs.existing_meanings_json or "[]",
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
