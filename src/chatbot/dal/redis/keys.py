from __future__ import annotations


def win_id(g: int) -> str:
    return f"grp:{g}:win:id"


def win_begin(g: int) -> str:
    return f"grp:{g}:win:begin_ts"


def win_deadline(g: int) -> str:
    return f"grp:{g}:win:deadline_ts"


def win_count(g: int) -> str:
    return f"grp:{g}:win:count"


def win_last_ts(g: int) -> str:
    return f"grp:{g}:win:last_msg_ts"


def win_last_idx(g: int) -> str:
    return f"grp:{g}:win:last_con_index"


def win_buf(win_id: str) -> str:
    return f"winbuf:{win_id}"


def win_sent(win_id: str) -> str:
    return f"winsent:{win_id}"


def pending_second(win_id: str) -> str:
    return f"pending_second:{win_id}"


def sent_task(task_id: str) -> str:
    return f"sched:sent:{task_id}"


def cancel_task(task_id: str) -> str:
    return f"sched:cancel:{task_id}"


def winclose_scheduled(win_id: str) -> str:
    return f"winclose_scheduled:{win_id}"


def winclose_lock(win_id: str) -> str:
    return f"winclose_lock:{win_id}"


def bot_chain_depth(g: int) -> str:
    return f"grp:{g}:bot_chain_depth"


def bot2bot_token(g: int) -> str:
    return f"grp:{g}:bot2bot_token"


def bucket(g: int) -> str:
    return f"bucket:grp:{g}"


def agent_cooldown(a: int) -> str:
    return f"cooldown:agt:{a}"
