from __future__ import annotations


def win_deadline(con_short_id: int) -> str:
    return f"grp:{con_short_id}:win:deadline_ts"


def win_buf(con_short_id: int) -> str:
    return f"grp:{con_short_id}:win:buf"


def winclose_scheduled(con_short_id: int) -> str:
    return f"grp:{con_short_id}:win:close_scheduled"