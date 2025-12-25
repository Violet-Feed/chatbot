from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from chatbot.utils.time import now_ms


@dataclass(frozen=True)
class SummaryUpdateDecision:
    """是否需要更新滚动摘要的判定结果。"""
    should_update: bool
    reason: str
    # 用于增量摘要：本次摘要覆盖到的 con_index
    new_last_con_index: int


def decide_update_summary(
    window_items: Sequence[Dict[str, Any]],
    current_summary: str,
    last_con_index: int,
    min_new_msgs: int = 20,
) -> SummaryUpdateDecision:
    """
    触发式滚动摘要（保守策略）：
    - 只有当“新增消息数 >= min_new_msgs”才触发
    - 且只用窗口里 con_index > last_con_index 的部分做增量摘要
    """
    if not window_items:
        return SummaryUpdateDecision(False, "EMPTY_WINDOW", last_con_index)

    newest_idx = int(window_items[-1].get("con_index", 0) or 0)
    if newest_idx <= last_con_index:
        return SummaryUpdateDecision(False, "NO_NEW_INDEX", last_con_index)

    new_msgs = [it for it in window_items if int(it.get("con_index", 0) or 0) > last_con_index]
    if len(new_msgs) < min_new_msgs:
        return SummaryUpdateDecision(False, f"NEW_MSGS_LT_{min_new_msgs}", last_con_index)

    return SummaryUpdateDecision(True, "THRESHOLD_REACHED", newest_idx)


def format_messages_for_summary(window_items: Sequence[Dict[str, Any]], limit: int = 80) -> str:
    """
    将窗口消息格式化为摘要输入文本：尽量短、可压缩。
    """
    items = list(window_items)[-limit:]
    lines: List[str] = []
    for it in items:
        who = "BOT" if it.get("is_bot") else "USER"
        sid = it.get("sender_id", 0)
        txt = str(it.get("msg_content", "") or "").replace("\n", " ").strip()
        if not txt:
            continue
        lines.append(f"[{who}:{sid}] {txt}")
    return "\n".join(lines)


def rule_based_rollup(current_summary: str, new_messages_text: str, max_chars: int = 800) -> str:
    """
    无 LLM 场景的兜底摘要（工程联调用）：
    - 只保留“旧摘要（截断） + 新消息（截断）”
    - 不追求语义质量，但保证系统可运行
    """
    cur = (current_summary or "").strip()
    new = (new_messages_text or "").strip()
    merged = (cur + "\n" + new).strip() if cur else new
    if len(merged) <= max_chars:
        return merged

    # 取尾部，避免无限增长
    return merged[-max_chars:]


async def llm_rollup(
    llm_summarize,  # async (current_summary, new_messages_text) -> str
    current_summary: str,
    new_messages_text: str,
) -> str:
    """
    用 LLM 做增量摘要。llm_summarize 由上层注入（可复用 planner/llm.py）。
    """
    return (await llm_summarize(current_summary, new_messages_text)).strip()


@dataclass(frozen=True)
class RollingSummaryState:
    summary: str
    last_con_index: int
    updated_at_ms: int


async def update_rolling_summary(
    window_items: Sequence[Dict[str, Any]],
    current_state: RollingSummaryState,
    llm_summarize=None,
    min_new_msgs: int = 20,
) -> RollingSummaryState:
    """
    入口：根据窗口与当前摘要状态，决定是否更新摘要并返回新状态。
    - 上层可以把结果写回 MySQL（rolling_summary 表）
    """
    decision = decide_update_summary(
        window_items=window_items,
        current_summary=current_state.summary,
        last_con_index=current_state.last_con_index,
        min_new_msgs=min_new_msgs,
    )
    if not decision.should_update:
        return current_state

    new_text = format_messages_for_summary(
        [it for it in window_items if int(it.get("con_index", 0) or 0) > current_state.last_con_index]
    )

    if llm_summarize is None:
        new_summary = rule_based_rollup(current_state.summary, new_text)
    else:
        new_summary = await llm_rollup(llm_summarize, current_state.summary, new_text)

    return RollingSummaryState(
        summary=new_summary,
        last_con_index=decision.new_last_con_index,
        updated_at_ms=now_ms(),
    )
