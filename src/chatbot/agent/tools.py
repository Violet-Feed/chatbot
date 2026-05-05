from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _web_search_sync(
        query: str,
        base_url: str,
        api_key: str,
        timeout_sec: int,
        count: int,
        freshness: str,
        summary: bool,
) -> str:
    payload = {
        "query": query,
        "count": max(1, int(count)),
        "freshness": freshness or "noLimit",
        "summary": bool(summary),
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "User-Agent": "chatbot/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(base_url.rstrip("/"), data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def make_web_search_tool(
        base_url: str,
        api_key: str,
        timeout_sec: int,
        max_chars: int,
        count: int = 8,
        freshness: str = "noLimit",
        summary: bool = True,
):
    @tool
    async def web_search(query: str) -> str:
        """联网搜索当前信息。当需要最新时事、价格、天气等实时内容时调用。"""
        logger.info("web_search query=%r", query)
        try:
            raw = await asyncio.to_thread(
                _web_search_sync,
                query,
                base_url,
                api_key,
                timeout_sec,
                count,
                freshness,
                summary,
            )
        except Exception:
            logger.exception("web_search failed query=%r", query)
            return "搜索失败，请根据已有知识回答"
        if not raw:
            return "搜索无结果，请直接根据已有知识回答"
        result = raw[:max_chars]
        logger.info("web_search result=%s", result)
        return result

    return web_search


def make_fetch_context_tool(im_client: Any, max_limit: int = 50):
    """创建一次，con_short_id / min_con_index / sender_id / sender_type 在调用时通过 config['configurable'] 注入。"""

    @tool
    async def fetch_more_context(
            limit: int = 30,
            config: RunnableConfig = None,
    ) -> str:
        """拉取当前会话更早的历史消息。当已有上下文不足以理解对话背景时调用。limit 最多50条。"""
        cfg = (config or {}).get("configurable", {})
        con_short_id = int(cfg.get("con_short_id", 0))
        min_con_index = int(cfg.get("min_con_index", 0) or 0)
        sender_id = int(cfg.get("sender_id", 0) or 0)
        sender_type = int(cfg.get("sender_type", 2) or 2)
        con_index = min_con_index - 1 if min_con_index > 0 else 0
        limit = min(max(limit, 1), max_limit)
        logger.info(
            "fetch_more_context con_short_id=%d con_index=%d limit=%d",
            con_short_id, con_index, limit,
        )
        try:
            messages = await im_client.get_message_by_conversation(
                sender_id=sender_id,
                sender_type=sender_type,
                con_short_id=con_short_id,
                con_index=con_index,
                limit=limit,
            )
        except Exception:
            logger.exception("fetch_more_context failed con_short_id=%d", con_short_id)
            return "获取历史消息失败，请根据已有上下文回答"

        if not messages:
            return "暂无更多历史消息"

        lines: list[str] = []
        for msg in messages:
            msg_sender_id = int(getattr(msg, "sender_id", 0))
            msg_sender_type = int(getattr(msg, "sender_type", 0))
            content = str(getattr(msg, "msg_content", "") or "").replace("\n", " ").strip()
            if not content:
                continue
            if msg_sender_type == 1:
                role = f"用户 {msg_sender_id}"
            elif msg_sender_type == 2:
                role = f"Agent {msg_sender_id}"
            else:
                role = f"未知 {msg_sender_id}"
            lines.append(f"[{role}]\n{content}")

        return "\n\n".join(lines) if lines else "暂无有效历史消息"

    return fetch_more_context
