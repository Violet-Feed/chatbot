from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


_EMOJI_PAT = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)


def _norm_text(x: Any) -> str:
    return str(x or "").replace("\r", "").strip()


def compute_style_features(window_items: Sequence[Dict[str, Any]], limit: int = 200) -> Dict[str, Any]:
    """
    风格指纹：轻量统计特征（不依赖模型）
    - 平均句长
    - 标点密度
    - 问号/感叹号比例
    - emoji 出现率
    - 常用句末标点偏好
    """
    items = list(window_items)[-limit:]
    texts: List[str] = []
    for it in items:
        if it.get("is_bot"):
            continue  # 群风格更多取人类
        t = _norm_text(it.get("msg_content"))
        if t:
            texts.append(t)

    if not texts:
        return {}

    total_chars = sum(len(t) for t in texts)
    total_msgs = len(texts)

    punct = "。！？!?；;，,、"
    punct_cnt = sum(sum(1 for ch in t if ch in punct) for t in texts)

    q_cnt = sum(t.count("?") + t.count("？") for t in texts)
    e_cnt = sum(t.count("!") + t.count("！") for t in texts)

    emoji_cnt = sum(len(_EMOJI_PAT.findall(t)) for t in texts)

    end_punct = {"。": 0, "？": 0, "！": 0, "": 0}
    for t in texts:
        if t.endswith("。"):
            end_punct["。"] += 1
        elif t.endswith("？") or t.endswith("?"):
            end_punct["？"] += 1
        elif t.endswith("！") or t.endswith("!"):
            end_punct["！"] += 1
        else:
            end_punct[""] += 1

    return {
        "avg_len": total_chars / max(1, total_msgs),
        "punct_per_msg": punct_cnt / max(1, total_msgs),
        "q_rate": q_cnt / max(1, total_msgs),
        "e_rate": e_cnt / max(1, total_msgs),
        "emoji_per_msg": emoji_cnt / max(1, total_msgs),
        "end_punct": end_punct,
        "sample_size": total_msgs,
    }


def dumps_features(features: Dict[str, Any]) -> str:
    """落库前序列化（稳定 JSON）。"""
    return json.dumps(features or {}, ensure_ascii=False, separators=(",", ":"))
