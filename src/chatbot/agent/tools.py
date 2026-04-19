from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from chatbot.utils.web_search import search_jina

logger = logging.getLogger(__name__)


def make_jina_search_tool(
    base_url: str,
    api_key: str,
    timeout_sec: int,
    max_chars: int,
):
    @tool
    async def jina_search(query: str) -> str:
        """联网搜索当前信息。当需要最新时事、价格、天气等实时内容时调用。"""
        logger.info("jina_search query=%r", query)
        raw = await search_jina(
            query,
            base_url=base_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
        )
        if not raw:
            return "搜索无结果，请直接根据已有知识回答"
        result = raw[:max_chars]
        logger.info("jina_search result_len=%d", len(result))
        return result

    return jina_search


def make_fetch_context_tool(im_client: Any, max_limit: int = 50):
    """创建一次，con_short_id 在调用时通过 config['configurable'] 注入。"""

    @tool
    async def fetch_more_context(
        limit: int = 30,
        config: RunnableConfig = None,
    ) -> str:
        """拉取当前会话更完整的历史消息。当已有上下文不足以理解对话背景时调用。limit 最多50条。"""
        con_short_id = int((config or {}).get("configurable", {}).get("con_short_id", 0))
        limit = min(max(limit, 1), max_limit)
        logger.info("fetch_more_context con_short_id=%d limit=%d", con_short_id, limit)
        try:
            messages = await im_client.get_message_by_conversation(
                user_id=0,
                con_short_id=con_short_id,
                con_index=0,
                limit=limit,
            )
        except Exception:
            logger.exception("fetch_more_context failed con_short_id=%d", con_short_id)
            return "获取历史消息失败，请根据已有上下文回答"

        if not messages:
            return "暂无更多历史消息"

        lines: list[str] = []
        for msg in messages:
            sender_id = int(getattr(msg, "sender_id", 0))
            sender_type = int(getattr(msg, "sender_type", 0))
            content = str(getattr(msg, "msg_content", "") or "").replace("\n", " ").strip()
            if not content:
                continue
            if sender_type == 1:
                role = f"用户 {sender_id}"
            elif sender_type == 2:
                role = f"Agent {sender_id}"
            else:
                role = f"未知 {sender_id}"
            lines.append(f"[{role}]\n{content}")

        return "\n\n".join(lines) if lines else "暂无有效历史消息"

    return fetch_more_context
