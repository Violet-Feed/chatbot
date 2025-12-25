from __future__ import annotations

from typing import Any, Dict

import orjson


def safe_obj_loads(s: str) -> Dict[str, Any]:
    """
    尝试将字符串解析为 JSON 对象（dict）。
    解析失败或非 dict 时返回空 dict（或用 {"_": value} 兜底）。
    """
    if not s:
        return {}
    try:
        obj = orjson.loads(s)
        if isinstance(obj, dict):
            return obj
        return {"_": obj}
    except Exception:
        return {}


def json_dumps(obj: Any) -> str:
    """高性能 JSON 序列化，返回 str。"""
    return orjson.dumps(obj).decode("utf-8")
