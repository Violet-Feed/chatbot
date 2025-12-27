# src/chatbot/planner/llm.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import orjson
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from chatbot.settings import Settings
from chatbot.planner import prompts


# -----------------------------
# Inputs/Interface（planner.py 依赖）
# -----------------------------
@dataclass(frozen=True)
class ReplyInputs:
    agent_id: int
    agent_name: str
    agent_personality: str
    agent_description: str

    rank: int  # 1/2
    trigger_reason: str
    intent: str

    recent_messages: str
    rolling_summary: str
    episodic_memory_json: str
    style_hint: str
    glossary_json: str
    web_context: str
    group_rules: str


@dataclass(frozen=True)
class DecisionInputs:
    agents_json: str
    recent_messages_json: str
    last_message_json: str
    bot_chain_depth: int
    group_rules: str


@dataclass(frozen=True)
class DecisionOutput:
    respond: bool
    primary_agent_id: Optional[int]
    secondary_agent_id: Optional[int]
    reason: str = ""


@dataclass(frozen=True)
class StyleLearnInputs:
    recent_messages: str


@dataclass(frozen=True)
class StyleLearnOutput:
    style_rules: List[str]
    hotwords: List[str]


@dataclass(frozen=True)
class SearchDecisionInputs:
    recent_messages: str


@dataclass(frozen=True)
class SearchDecisionOutput:
    need_search: bool
    query: str
    reason: str = ""


@dataclass(frozen=True)
class UnknownTermsInputs:
    recent_messages: str
    known_terms_json: str


@dataclass(frozen=True)
class UnknownTermsOutput:
    terms: List[str]


@dataclass(frozen=True)
class TermMeaningInputs:
    term: str
    recent_messages: str
    web_context: str


@dataclass(frozen=True)
class TermMeaningOutput:
    meaning: str
    confidence: float


@dataclass(frozen=True)
class ClarifyInputs:
    terms: str
    recent_messages: str


class LLMClient(Protocol):
    async def generate_reply(self, inputs: ReplyInputs) -> str: ...
    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput: ...
    async def learn_style(self, inputs: StyleLearnInputs) -> StyleLearnOutput: ...
    async def decide_search(self, inputs: SearchDecisionInputs) -> SearchDecisionOutput: ...
    async def extract_unknown_terms(self, inputs: UnknownTermsInputs) -> UnknownTermsOutput: ...
    async def infer_term_meaning(self, inputs: TermMeaningInputs) -> TermMeaningOutput: ...
    async def generate_clarify(self, inputs: ClarifyInputs) -> str: ...
    async def summarize(self, current_summary: str, new_messages: str) -> str: ...
    async def extract_episodic(self, messages_text: str) -> List[Dict[str, Any]]: ...


@dataclass(frozen=True)
class NoopLLMClient:
    async def generate_reply(self, inputs: ReplyInputs) -> str:
        return ""

    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput:
        return DecisionOutput(respond=False, primary_agent_id=None, secondary_agent_id=None, reason="noop")

    async def learn_style(self, inputs: StyleLearnInputs) -> StyleLearnOutput:
        return StyleLearnOutput(style_rules=[], hotwords=[])

    async def decide_search(self, inputs: SearchDecisionInputs) -> SearchDecisionOutput:
        return SearchDecisionOutput(need_search=False, query="", reason="noop")

    async def extract_unknown_terms(self, inputs: UnknownTermsInputs) -> UnknownTermsOutput:
        return UnknownTermsOutput(terms=[])

    async def infer_term_meaning(self, inputs: TermMeaningInputs) -> TermMeaningOutput:
        return TermMeaningOutput(meaning="", confidence=0.0)

    async def generate_clarify(self, inputs: ClarifyInputs) -> str:
        return ""

    async def summarize(self, current_summary: str, new_messages: str) -> str:
        return current_summary or ""

    async def extract_episodic(self, messages_text: str) -> List[Dict[str, Any]]:
        return []


# -----------------------------
# Qwen (OpenAI-Compatible) via LangChain
# -----------------------------
def _new_chat(s: Settings, max_tokens: int, temperature: Optional[float] = None) -> ChatOpenAI:
    api_key = s.DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY", "")
    if not api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY（.env 或环境变量）")

    kwargs = dict(
        base_url=s.QWEN_BASE_URL.rstrip("/"),
        api_key=api_key,
        model=s.QWEN_MODEL,
        temperature=float(s.QWEN_TEMPERATURE if temperature is None else temperature),
        timeout=int(s.QWEN_TIMEOUT_SEC),
        max_retries=2,
        model_kwargs={"max_tokens": int(max_tokens)},
    )

    # DashScope compatible-mode: /chat/completions
    try:
        return ChatOpenAI(**kwargs, use_responses_api=False)  # type: ignore[arg-type]
    except TypeError:
        return ChatOpenAI(**kwargs)


def _render(template: str, **kwargs: Any) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise RuntimeError(f"提示词模板缺少占位符：{e}") from e
    except Exception as e:
        raise RuntimeError(f"提示词模板渲染失败：{e}") from e


def _clip(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else (text[:n] + "…")


def _extract_json_array(text: str) -> Optional[str]:
    """
    允许模型偶发输出多余前后缀：尽量截取第一个 '[' 到最后一个 ']' 之间内容。
    """
    if not text:
        return None
    s = text.strip()

    # 去掉常见代码块包裹
    if s.startswith("```"):
        s = s.strip("`")
        # 可能是 ```json\n...\n```
        s = s.replace("json", "", 1).strip()

    l = s.find("[")
    r = s.rfind("]")
    if l == -1 or r == -1 or r <= l:
        return None
    return s[l : r + 1]


def _extract_json_object(text: str) -> Optional[str]:
    """
    允许模型偶发输出多余前后缀：尽量截取第一个 '{' 到最后一个 '}' 之间内容。
    """
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


class LangChainQwenOpenAICompatibleClient:
    """
    只使用 prompts.py 中的提示词：
    - prompts.REPLY_SYSTEM / prompts.REPLY_USER
    - prompts.SUMMARY_SYSTEM / prompts.SUMMARY_USER
    - prompts.EPISODIC_SYSTEM / prompts.EPISODIC_USER
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings

    async def decide_reply(self, inputs: DecisionInputs) -> DecisionOutput:
        chat = _new_chat(self.s, max_tokens=200, temperature=0.2)

        sys = prompts.DECISION_SYSTEM
        usr = _render(
            prompts.DECISION_USER,
            agents_json=inputs.agents_json or "[]",
            recent_messages_json=inputs.recent_messages_json or "[]",
            last_message_json=inputs.last_message_json or "{}",
            bot_chain_depth=str(int(inputs.bot_chain_depth)),
            group_rules=inputs.group_rules or "",
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return DecisionOutput(respond=False, primary_agent_id=None, secondary_agent_id=None, reason="bad_json")

        try:
            obj = orjson.loads(payload)
        except Exception:
            return DecisionOutput(respond=False, primary_agent_id=None, secondary_agent_id=None, reason="bad_json")

        if not isinstance(obj, dict):
            return DecisionOutput(respond=False, primary_agent_id=None, secondary_agent_id=None, reason="bad_json")

        respond = bool(obj.get("respond", False))
        primary = obj.get("primary_agent_id")
        secondary = obj.get("secondary_agent_id")
        reason = str(obj.get("reason", "") or "")

        try:
            primary_id = int(primary) if primary is not None else None
        except Exception:
            primary_id = None

        try:
            secondary_id = int(secondary) if secondary is not None else None
        except Exception:
            secondary_id = None

        return DecisionOutput(
            respond=respond,
            primary_agent_id=primary_id,
            secondary_agent_id=secondary_id,
            reason=reason,
        )

    async def learn_style(self, inputs: StyleLearnInputs) -> StyleLearnOutput:
        chat = _new_chat(self.s, max_tokens=260, temperature=0.3)

        sys = prompts.STYLE_SYSTEM
        usr = _render(
            prompts.STYLE_USER,
            recent_messages=_clip(inputs.recent_messages or "", 9000),
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return StyleLearnOutput(style_rules=[], hotwords=[])

        try:
            obj = orjson.loads(payload)
        except Exception:
            return StyleLearnOutput(style_rules=[], hotwords=[])

        if not isinstance(obj, dict):
            return StyleLearnOutput(style_rules=[], hotwords=[])

        style_rules_raw = obj.get("style_rules") or []
        hotwords_raw = obj.get("hotwords") or []

        style_rules: List[str] = []
        if isinstance(style_rules_raw, list):
            for it in style_rules_raw:
                s = str(it or "").strip()
                if s:
                    style_rules.append(s)

        hotwords: List[str] = []
        if isinstance(hotwords_raw, list):
            for it in hotwords_raw:
                s = str(it or "").strip()
                if s:
                    hotwords.append(s)

        return StyleLearnOutput(style_rules=style_rules[:12], hotwords=hotwords[:30])

    async def decide_search(self, inputs: SearchDecisionInputs) -> SearchDecisionOutput:
        chat = _new_chat(self.s, max_tokens=120, temperature=0.2)

        sys = prompts.SEARCH_DECISION_SYSTEM
        usr = _render(prompts.SEARCH_DECISION_USER, recent_messages=_clip(inputs.recent_messages or "", 4000))

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return SearchDecisionOutput(need_search=False, query="", reason="bad_json")

        try:
            obj = orjson.loads(payload)
        except Exception:
            return SearchDecisionOutput(need_search=False, query="", reason="bad_json")

        if not isinstance(obj, dict):
            return SearchDecisionOutput(need_search=False, query="", reason="bad_json")

        need_search = bool(obj.get("need_search", False))
        query = str(obj.get("query", "") or "").strip()
        reason = str(obj.get("reason", "") or "")
        if not need_search:
            query = ""

        return SearchDecisionOutput(need_search=need_search, query=query, reason=reason)

    async def extract_unknown_terms(self, inputs: UnknownTermsInputs) -> UnknownTermsOutput:
        chat = _new_chat(self.s, max_tokens=160, temperature=0.2)

        sys = prompts.UNKNOWN_TERM_SYSTEM
        usr = _render(
            prompts.UNKNOWN_TERM_USER,
            recent_messages=_clip(inputs.recent_messages or "", 6000),
            known_terms_json=inputs.known_terms_json or "[]",
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return UnknownTermsOutput(terms=[])

        try:
            obj = orjson.loads(payload)
        except Exception:
            return UnknownTermsOutput(terms=[])

        if not isinstance(obj, dict):
            return UnknownTermsOutput(terms=[])

        terms_raw = obj.get("terms") or []
        terms: List[str] = []
        if isinstance(terms_raw, list):
            for it in terms_raw:
                s = str(it or "").strip()
                if s:
                    terms.append(s)

        return UnknownTermsOutput(terms=terms[:6])

    async def infer_term_meaning(self, inputs: TermMeaningInputs) -> TermMeaningOutput:
        chat = _new_chat(self.s, max_tokens=200, temperature=0.3)

        sys = prompts.TERM_MEANING_SYSTEM
        usr = _render(
            prompts.TERM_MEANING_USER,
            term=inputs.term or "",
            recent_messages=_clip(inputs.recent_messages or "", 4000),
            web_context=_clip(inputs.web_context or "", 6000),
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_object(raw)
        if not payload:
            return TermMeaningOutput(meaning="", confidence=0.0)

        try:
            obj = orjson.loads(payload)
        except Exception:
            return TermMeaningOutput(meaning="", confidence=0.0)

        if not isinstance(obj, dict):
            return TermMeaningOutput(meaning="", confidence=0.0)

        meaning = str(obj.get("meaning", "") or "").strip()
        try:
            confidence = float(obj.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))

        return TermMeaningOutput(meaning=meaning, confidence=confidence)

    async def generate_clarify(self, inputs: ClarifyInputs) -> str:
        chat = _new_chat(self.s, max_tokens=120, temperature=0.4)

        sys = prompts.CLARIFY_SYSTEM
        usr = _render(
            prompts.CLARIFY_USER,
            terms=inputs.terms or "",
            recent_messages=_clip(inputs.recent_messages or "", 2000),
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        return (getattr(msg, "content", "") or "").strip()

    async def generate_reply(self, inputs: ReplyInputs) -> str:
        max_tokens = 160 if inputs.rank == 1 else 96
        chat = _new_chat(self.s, max_tokens=max_tokens)

        sys = _render(
            prompts.REPLY_SYSTEM,
            personality=inputs.agent_personality or "",
            intent=inputs.intent or "",
            trigger_reason=inputs.trigger_reason or "",
        )
        usr = _render(
            prompts.REPLY_USER,
            recent_messages=inputs.recent_messages or "",
            rolling_summary=inputs.rolling_summary or "",
            episodic_memory=inputs.episodic_memory_json or "[]",
            style_hint=inputs.style_hint or "",
            glossary_json=inputs.glossary_json or "[]",
            web_context=inputs.web_context or "",
            group_rules=inputs.group_rules or "",
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        return (getattr(msg, "content", "") or "").strip()

    async def summarize(self, current_summary: str, new_messages: str) -> str:
        # 摘要要更稳定：温度低
        chat = _new_chat(self.s, max_tokens=260, temperature=0.2)

        sys = prompts.SUMMARY_SYSTEM
        usr = _render(
            prompts.SUMMARY_USER,
            current_summary=current_summary or "",
            new_messages=_clip(new_messages or "", 8000),
        )

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        out = (getattr(msg, "content", "") or "").strip()

        # 防御：避免异常超长污染后续上下文
        return _clip(out, 1800).strip()

    async def extract_episodic(self, messages_text: str) -> List[Dict[str, Any]]:
        # 结构化抽取：温度更低
        chat = _new_chat(self.s, max_tokens=320, temperature=0.1)

        sys = prompts.EPISODIC_SYSTEM
        usr = _render(prompts.EPISODIC_USER, messages=_clip(messages_text or "", 9000))

        msg = await chat.ainvoke([SystemMessage(content=sys), HumanMessage(content=usr)])
        raw = (getattr(msg, "content", "") or "").strip()

        payload = _extract_json_array(raw)
        if not payload:
            return []

        try:
            obj = orjson.loads(payload)
        except Exception:
            return []

        if not isinstance(obj, list):
            return []

        out: List[Dict[str, Any]] = []
        for it in obj[:8]:
            if not isinstance(it, dict):
                continue

            content = str(it.get("content", "") or "").strip()
            if not content:
                continue

            # 规范化字段
            typ = str(it.get("type", "generic") or "generic").strip() or "generic"
            try:
                importance = float(it.get("importance", 0.5) or 0.5)
            except Exception:
                importance = 0.5
            importance = max(0.0, min(1.0, importance))

            try:
                ttl_ms = int(it.get("ttl_ms", 0) or 0)
            except Exception:
                ttl_ms = 0
            ttl_ms = max(0, ttl_ms)

            out.append(
                {
                    "type": typ,
                    "content": content,
                    "importance": importance,
                    "ttl_ms": ttl_ms,
                }
            )
        return out
