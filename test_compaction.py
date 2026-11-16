import pytest
from utils.models import (
    InvalidLimitError,
    InvalidSummaryError,
    MemoryState,
    MessageRecord,
    SummaryInput,
    SummaryRecord,
)
from main import (
    select_messages_to_compact,
    build_summary_input,
    compact_memory,
)


def make_message(role: str, content: str) -> MessageRecord:
    return MessageRecord(role=role, content=content)


def make_state(
    n_messages: int,
    summary: SummaryRecord | None = None,
    preferences: dict | None = None,
    tasks: tuple[str, ...] | None = None,
) -> MemoryState:
    messages = tuple(make_message("user" if i % 2 == 0 else "assistant", f"message {i + 1}") for i in range(n_messages))
    return MemoryState(
        messages=messages,
        summary=summary,
        preferences={} if preferences is None else preferences,
        unresolved_tasks=() if tasks is None else tasks,
    )


class TestSelectMessagesToCompact:
    def test_invalid_threshold_types(self):
        state = make_state(5)
        for val in [True, False, 0, -1, -10, 1.5, "5", None, [], {}]:
            with pytest.raises(InvalidLimitError):
                select_messages_to_compact(state, val, 2)

    def test_invalid_keep_recent_types(self):
        state = make_state(5)
        for val in [True, False, 0, -1, -10, 1.5, "2", None, [], {}]:
            with pytest.raises(InvalidLimitError):
                select_messages_to_compact(state, 5, val)

    def test_keep_recent_greater_than_threshold(self):
        state = make_state(5)
        with pytest.raises(InvalidLimitError):
            select_messages_to_compact(state, threshold=3, keep_recent=4)

    def test_keep_recent_equal_threshold(self):
        state = make_state(5)
        # Should not raise
        result = select_messages_to_compact(state, threshold=3, keep_recent=3)
        assert len(result) == 2
        assert [m.content for m in result] == ["message 1", "message 2"]

    def test_at_threshold_no_compaction(self):
        state = make_state(5)
        result = select_messages_to_compact(state, threshold=5, keep_recent=2)
        assert result == ()

    def test_below_threshold_no_compaction(self):
        state = make_state(3)
        result = select_messages_to_compact(state, threshold=5, keep_recent=2)
        assert result == ()

    def test_above_threshold_compaction(self):
        state = make_state(7)
        result = select_messages_to_compact(state, threshold=5, keep_recent=2)
        assert len(result) == 5
        assert [m.content for m in result] == [
            "message 1",
            "message 2",
            "message 3",
            "message 4",
            "message 5",
        ]

    def test_empty_messages(self):
        state = make_state(0)
        result = select_messages_to_compact(state, threshold=3, keep_recent=1)
        assert result == ()


class TestBuildSummaryInput:
    def test_no_existing_summary(self):
        msgs = [make_message("user", "hi"), make_message("assistant", "hello")]
        inp = build_summary_input(None, msgs)
        assert inp.existing_summary is None
        assert inp.older_messages == tuple(msgs)

    def test_empty_existing_summary(self):
        msgs = [make_message("user", "hi")]
        inp = build_summary_input(SummaryRecord(content=""), msgs)
        assert inp.existing_summary is None
        assert inp.older_messages == tuple(msgs)

    def test_whitespace_existing_summary(self):
        msgs = [make_message("user", "hi")]
        inp = build_summary_input(SummaryRecord(content="   \n\t  "), msgs)
        assert inp.existing_summary is None
        assert inp.older_messages == tuple(msgs)

    def test_valid_existing_summary(self):
        msgs = [make_message("user", "hi")]
        inp = build_summary_input(SummaryRecord(content="Prior summary"), msgs)
        assert inp.existing_summary == "Prior summary"
        assert inp.older_messages == tuple(msgs)

    def test_generator_messages(self):
        msgs = [make_message("user", f"msg {i}") for i in range(3)]
        gen = (m for m in msgs)
        inp = build_summary_input(None, gen)
        assert isinstance(inp.older_messages, tuple)
        assert inp.older_messages == tuple(msgs)

    def test_summary_not_in_older_messages(self):
        msgs = [make_message("user", "hi")]
        inp = build_summary_input(SummaryRecord(content="Prior summary"), msgs)
        assert len(inp.older_messages) == 1
        assert all(isinstance(m, MessageRecord) for m in inp.older_messages)
        assert all(m.content != "Prior summary" for m in inp.older_messages)


class TestCompactMemory:
    def test_no_compaction_at_threshold(self):
        state = make_state(5, preferences={"theme": "dark"}, tasks=("task1",))
        called = False

        def fake_summarizer(inp):
            nonlocal called
            called = True
            return "summary"

        new_state = compact_memory(state, fake_summarizer, threshold=5, keep_recent=2)
        assert not called
        assert len(new_state.messages) == 5
        assert new_state.messages == state.messages
        assert new_state.summary == state.summary
        assert new_state.preferences == state.preferences
        assert new_state.unresolved_tasks == state.unresolved_tasks

    def test_no_compaction_below_threshold(self):
        state = make_state(3)
        called = False

        def fake_summarizer(inp):
            nonlocal called
            called = True
            return "summary"

        new_state = compact_memory(state, fake_summarizer, threshold=5, keep_recent=2)
        assert not called
        assert len(new_state.messages) == 3

    def test_compaction_called_exactly_once(self):
        state = make_state(7, preferences={"subject": "math"}, tasks=("task A",))
        call_count = 0
        captured_input = None

        def fake_summarizer(inp: SummaryInput):
            nonlocal call_count, captured_input
            call_count += 1
            captured_input = inp
            return "Summary of 1 to 5"

        new_state = compact_memory(state, fake_summarizer, threshold=5, keep_recent=2)
        assert call_count == 1
        assert captured_input is not None
        assert captured_input.existing_summary is None
        assert len(captured_input.older_messages) == 5
        assert [m.content for m in captured_input.older_messages] == [
            "message 1", "message 2", "message 3", "message 4", "message 5"
        ]

        assert len(new_state.messages) == 2
        assert [m.content for m in new_state.messages] == ["message 6", "message 7"]
        assert new_state.summary == SummaryRecord(content="Summary of 1 to 5")
        assert new_state.preferences == {"subject": "math"}
        assert new_state.unresolved_tasks == ("task A",)

    def test_immutability_caller_state_preserved(self):
        initial_prefs = {"nested": {"key": "val"}}
        initial_tasks = ("task 1",)
        state = make_state(7, preferences=initial_prefs, tasks=initial_tasks)

        new_state = compact_memory(state, lambda inp: "summary", threshold=5, keep_recent=2)

        # Mutate new_state preferences
        new_state.preferences["nested"]["key"] = "mutated"
        assert state.preferences["nested"]["key"] == "val"

        # Mutate state preferences
        state.preferences["nested"]["key"] = "another_mutation"
        assert new_state.preferences["nested"]["key"] == "mutated"

    def test_immutability_on_no_compaction(self):
        initial_prefs = {"nested": {"key": "val"}}
        state = make_state(3, preferences=initial_prefs)

        new_state = compact_memory(state, lambda inp: "summary", threshold=5, keep_recent=2)

        new_state.preferences["nested"]["key"] = "mutated"
        assert state.preferences["nested"]["key"] == "val"

    def test_invalid_summary_outputs(self):
        state = make_state(7)

        for invalid_output in ["", "   ", " \n\t ", None, 123, ["summary"], {}]:
            with pytest.raises(InvalidSummaryError):
                compact_memory(state, lambda inp: invalid_output, threshold=5, keep_recent=2)

    def test_repeated_compaction(self):
        state = make_state(7)
        history = []

        def summarizer(inp: SummaryInput):
            history.append(inp)
            prior = f"({inp.existing_summary}) + " if inp.existing_summary else ""
            return f"{prior}compacted({len(inp.older_messages)} msgs)"

        # Round 1
        state1 = compact_memory(state, summarizer, threshold=5, keep_recent=2)
        assert len(state1.messages) == 2
        assert [m.content for m in state1.messages] == ["message 6", "message 7"]
        assert state1.summary == SummaryRecord(content="compacted(5 msgs)")
        assert len(history) == 1
        assert history[0].existing_summary is None

        # Add more messages to state1
        new_msgs = state1.messages + (
            make_message("user", "message 8"),
            make_message("assistant", "message 9"),
            make_message("user", "message 10"),
            make_message("assistant", "message 11"),
        )
        state1_extended = MemoryState(
            messages=new_msgs,
            summary=state1.summary,
            preferences=state1.preferences,
            unresolved_tasks=state1.unresolved_tasks,
        )
        assert len(state1_extended.messages) == 6

        # Round 2
        state2 = compact_memory(state1_extended, summarizer, threshold=5, keep_recent=2)
        assert len(history) == 2
        assert history[1].existing_summary == "compacted(5 msgs)"
        assert len(history[1].older_messages) == 4
        assert [m.content for m in history[1].older_messages] == [
            "message 6", "message 7", "message 8", "message 9"
        ]
        assert len(state2.messages) == 2
        assert [m.content for m in state2.messages] == ["message 10", "message 11"]
        assert state2.summary == SummaryRecord(content="(compacted(5 msgs)) + compacted(4 msgs)")
