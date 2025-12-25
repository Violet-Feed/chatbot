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
    group_rules: str


class LLMClient(Protocol):
    async def generate_reply(self, inputs: ReplyInputs) -> str: ...
    async def summarize(self, current_summary: str, new_messages: str) -> str: ...
    async def extract_episodic(self, messages_text: str) -> List[Dict[str, Any]]: ...


@dataclass(frozen=True)
class NoopLLMClient:
    async def generate_reply(self, inputs: ReplyInputs) -> str:
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


class LangChainQwenOpenAICompatibleClient:
    """
    只使用 prompts.py 中的提示词：
    - prompts.REPLY_SYSTEM / prompts.REPLY_USER
    - prompts.SUMMARY_SYSTEM / prompts.SUMMARY_USER
    - prompts.EPISODIC_SYSTEM / prompts.EPISODIC_USER
    """

    def __init__(self, settings: Settings) -> None:
        self.s = settings

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
