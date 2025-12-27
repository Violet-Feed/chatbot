from __future__ import annotations

import asyncio
import urllib.parse
import urllib.request
from typing import Optional


def _search_sync(
    query: str,
    base_url: str,
    api_key: str,
    timeout_sec: int,
) -> str:
    q = urllib.parse.quote_plus(query)
    url = f"{base_url}{q}"

    headers = {
        "User-Agent": "chatbot/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="ignore")


async def search_jina(
    query: str,
    base_url: str,
    api_key: str = "",
    timeout_sec: int = 20,
) -> Optional[str]:
    if not query:
        return None
    try:
        return await asyncio.to_thread(_search_sync, query, base_url, api_key, timeout_sec)
    except Exception:
        return None
