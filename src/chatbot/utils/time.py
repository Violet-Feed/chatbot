from __future__ import annotations

import time


def now_ms() -> int:
    """返回当前时间戳（毫秒）。"""
    return int(time.time() * 1000)
