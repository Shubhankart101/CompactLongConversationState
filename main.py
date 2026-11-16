"""Compact Long Conversation State.

Scenario
--------
A study assistant has a growing conversation history. It must summarize older
messages while retaining recent turns and structured learner state exactly.

Complete only these TODOs:
1. select_messages_to_compact
2. build_summary_input
3. compact_memory

The state records, summary-input type, validation errors, limits, and a
Groq-compatible summarizer are supplied. Pytest uses a mocked summarizer, so it
requires no API key or network access.

Required behavior
-----------------
- threshold and keep_recent must be positive non-boolean integers, with
  keep_recent no greater than threshold. Otherwise raise InvalidLimitError.
- Compact only when the message count is greater than threshold.
- Select every message except the exact latest keep_recent messages.
- SummaryInput must keep an existing nonblank summary separately from the
  ordered older messages. A blank existing summary becomes None.
- compact_memory calls the summarizer once only when compaction is needed.
- The returned state contains the new single SummaryRecord and exact recent
  message tail. Preferences and unresolved tasks remain unchanged.
- Do not mutate the input state, its message list, preferences, or tasks.
- Raise InvalidSummaryError if the summarizer does not return nonblank text.

Not required: writing a summarization prompt, judging summary prose, counting
tokens, implementing persistence, calling a real model during evaluation, or
building an interface.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from utils.models import (
    InvalidLimitError,
    InvalidSummaryError,
    MemoryState,
    MessageRecord,
    SummaryInput,
    SummaryRecord,
)


def select_messages_to_compact(
    state: MemoryState,
    threshold: int,
    keep_recent: int,
) -> tuple[MessageRecord, ...]:
    """Select the ordered older messages only when compaction is required."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold <= 0:
        raise InvalidLimitError("threshold must be a positive non-boolean integer")
    if isinstance(keep_recent, bool) or not isinstance(keep_recent, int) or keep_recent <= 0:
        raise InvalidLimitError("keep_recent must be a positive non-boolean integer")
    if keep_recent > threshold:
        raise InvalidLimitError("keep_recent must not be greater than threshold")

    all_messages = tuple(state.messages)
    if len(all_messages) <= threshold:
        return ()

    return all_messages[:-keep_recent]


def build_summary_input(
    existing_summary: SummaryRecord | None,
    older_messages: Iterable[MessageRecord],
) -> SummaryInput:
    """Build one non-nested input for the supplied summarizer."""
    summary_text: str | None = None
    if existing_summary is not None:
        if isinstance(existing_summary, SummaryRecord):
            raw = existing_summary.content
        elif isinstance(existing_summary, str):
            raw = existing_summary
        elif hasattr(existing_summary, "content"):
            raw = getattr(existing_summary, "content")
        else:
            raw = None

        if isinstance(raw, str) and raw.strip():
            summary_text = raw

    return SummaryInput(
        existing_summary=summary_text,
        older_messages=tuple(older_messages),
    )


def compact_memory(
    state: MemoryState,
    summarizer,
    threshold: int,
    keep_recent: int,
) -> MemoryState:
    """Compact older messages without changing caller-owned state."""
    older_messages = select_messages_to_compact(state, threshold, keep_recent)
    if not older_messages:
        return state

    summary_input = build_summary_input(state.summary, older_messages)
    raw_summary = summarizer(summary_input)

    if not isinstance(raw_summary, str) or not raw_summary.strip():
        raise InvalidSummaryError("Summarizer output must be a nonblank string")

    recent_tail = tuple(state.messages[-keep_recent:])
    return MemoryState(
        messages=recent_tail,
        summary=SummaryRecord(content=raw_summary),
        preferences=copy.deepcopy(state.preferences),
        unresolved_tasks=tuple(copy.deepcopy(state.unresolved_tasks)),
    )
