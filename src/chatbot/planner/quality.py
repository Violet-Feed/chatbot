from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def enforce_short(text: str, strict: bool) -> str:
    """
    输出长度闸门：
    - strict=True（第二 bot）更短
    - 最多两行；每行最多 1~2 句
    """
    t = (text or "").replace("\r", "").strip()
    lines = [x.strip() for x in t.split("\n") if x.strip()][:2]
    if not lines:
        return ""

    out: List[str] = []
    limit = 1 if strict else 2

    for ln in lines:
        segs: List[str] = []
        buf = ""
        for ch in ln:
            buf += ch
            if ch in "。！？!?；;":
                segs.append(buf.strip())
                buf = ""
                if len(segs) >= limit:
                    break
        if buf and len(segs) < limit:
            segs.append(buf.strip())
        out.append("".join(segs).strip())

    return "\n".join(out).strip()


def split_human_like(text: str, max_segments: int = 3) -> List[str]:
    """
    尽量拟人：长句按标点切成多条短消息。
    """
    t = (text or "").replace("\r", " ").strip()
    if not t:
        return []
    segs: List[str] = []
    buf = ""
    for ch in t:
        buf += ch
        if ch in "。！？!?；;":
            segs.append(buf.strip())
            buf = ""
            if len(segs) >= max_segments:
                break
    if buf and len(segs) < max_segments:
        segs.append(buf.strip())
    return [s for s in segs if s][:max_segments]


def _tokenize_cn(s: str) -> List[str]:
    """
    极简 token：按字符与英文单词混合，足够做重复度闸门。
    """
    s = (s or "").strip()
    if not s:
        return []
    out: List[str] = []
    buf = ""
    for ch in s:
        if ch.isascii() and (ch.isalnum() or ch in "_-"):
            buf += ch
        else:
            if buf:
                out.append(buf)
                buf = ""
            if not ch.isspace():
                out.append(ch)
    if buf:
        out.append(buf)
    return out


def jaccard_similarity(a: str, b: str) -> float:
    sa = set(_tokenize_cn(a))
    sb = set(_tokenize_cn(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def too_similar(new_text: str, recent_bot_texts: Sequence[str], threshold: float = 0.75) -> bool:
    """
    防复读：与最近 bot 发言相似度过高则丢弃。
    """
    for t in recent_bot_texts[-5:]:
        if jaccard_similarity(new_text, t) >= threshold:
            return True
    return False
