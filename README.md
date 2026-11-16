# Compact Long Conversation State

A robust, production-grade memory compaction module for conversational AI assistants. This system bounds conversation history growth by progressively summarizing older message turns while preserving recent messages, structured learner preferences, and pending tasks without state mutation.

---

## 📌 Problem Overview

As a conversational AI assistant interacts with a learner over time, its history grows monotonically. Feeding unconstrained history back into an LLM causes:
1. **Context Window Exhaustion:** Exceeding model token limits.
2. **Elevated Latency & Costs:** Quadratic/linear cost increases with every prompt turn.
3. **Loss of Focus:** Degradation in response quality ("needle in a haystack" problem).

To keep memory bounded, older turns must be iteratively summarized. However, standard naive truncation or recursive concatenation can lose critical context, nest summaries into unreadable structures, or accidentally mutate state.

### Design Goals
- **Bounded State:** Compacts only when the message count strictly exceeds a defined `threshold`.
- **Sliding History Window:** Retains the exact `keep_recent` messages in verbatim format.
- **Single Running Summary:** Progressively updates a single `SummaryRecord` across compaction rounds rather than producing nested or lost summaries.
- **Strict Immutability:** Preserves caller-owned state, deeply isolating structured user preferences and unresolved tasks against accidental in-place mutations.
- **Pluggable Summarizer:** Decoupled design compatible with Groq, OpenAI, Anthropic, or mock test drivers.

---

## 🏗️ Architecture & Data Structures

All data structures are defined in `utils/models.py` as immutable (frozen) dataclasses:

| Model | Description |
|---|---|
| `MessageRecord` | An individual conversation turn containing `role: str` and `content: str`. |
| `SummaryRecord` | Represents the cumulative conversation summary (`content: str`). |
| `SummaryInput` | Decoupled payload passed to summarizers with `existing_summary: str \| None` and `older_messages: tuple[MessageRecord, ...]`. |
| `MemoryState` | Complete assistant memory holding `messages`, `summary`, `preferences: dict[str, Any]`, and `unresolved_tasks: tuple[str, ...]`. |

---

## ⚙️ Core Functions

### 1. `select_messages_to_compact(state, threshold, keep_recent)`
- Validates that `threshold` and `keep_recent` are positive non-boolean integers (`int > 0`) and requires `keep_recent <= threshold`.
- Raises `InvalidLimitError` on any invalid argument (e.g. `bool`, `float`, non-positive, or `keep_recent > threshold`).
- **No-op:** Returns an empty tuple `()` when `len(messages) <= threshold`.
- **Compaction:** Returns an ordered tuple containing every message before the most recent `keep_recent` items (`state.messages[:-keep_recent]`).

### 2. `build_summary_input(existing_summary, older_messages)`
- Converts any missing, empty, or whitespace-only existing summary into `None`.
- Preserves clean nonblank existing summaries in the separate `existing_summary` field.
- Packages older messages as an ordered tuple `older_messages`.
- Ensures the prior summary is **never** nested or placed into the raw message turn sequence.

### 3. `compact_memory(state, summarizer, threshold, keep_recent)`
- Calls `select_messages_to_compact()` first.
- **Boundary / No-op:** If message count does not exceed `threshold`, skips the summarizer entirely and returns an equivalent safe copy with isolated structured fields.
- **Compaction:** Calls the supplied `summarizer` callable **exactly once** with a single `SummaryInput`.
- **Validation:** Verifies the summarizer returned a nonblank string; raises `InvalidSummaryError` if empty, whitespace-only, or not a string.
- **State Assembly:** Constructs a new `MemoryState` containing:
  - Exact recent message tail of length `keep_recent`.
  - Updated single `SummaryRecord`.
  - Deep-copied `preferences` and `unresolved_tasks`.

---

## 📊 Example Compaction Flow

Consider a state with 7 messages, `threshold = 5`, and `keep_recent = 2`:

```
Initial Messages (7): [M1, M2, M3, M4, M5, M6, M7]
Threshold: 5 (Compaction triggered because 7 > 5)
Keep Recent: 2

Prefix for Summarizer (5): [M1, M2, M3, M4, M5]
Retained Tail (2):         [M6, M7]

Summarizer Output: "Summary of M1-M5"
New MemoryState:
  - messages: [M6, M7]
  - summary: SummaryRecord("Summary of M1-M5")
  - preferences & tasks: Preserved & Isolated
```

### Multi-Round Compaction
When more messages arrive (e.g. M8, M9, M10, M11):
- Total messages = 6 (`[M6, M7, M8, M9, M10, M11]`).
- Threshold = 5 -> compaction triggered.
- Older messages: `[M6, M7, M8, M9]`.
- Summarizer input: `existing_summary="Summary of M1-M5"`, `older_messages=[M6, M7, M8, M9]`.
- Retained tail: `[M10, M11]`.
- New summary updates cleanly without nesting.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Shubhankart101/CompactLongConversationState.git
   cd CompactLongConversationState
   ```

2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Testing

The repository includes a comprehensive test suite in `test_compaction.py` with 21 unit and integration tests covering:
- **Limit Validation:** Types, booleans, zero, negatives, and relative constraints (`keep_recent <= threshold`).
- **Threshold Boundaries:** Testing below threshold, at threshold, and above threshold behavior.
- **Immutability & Isolation:** Verifying caller-owned dictionaries and tasks cannot be mutated via the returned state.
- **Summarizer Constraints:** Skipped when below/at threshold, called exactly once during compaction, rejection of invalid/blank outputs (`InvalidSummaryError`).
- **Sequential Multi-Round Compaction:** Context preservation without recursive summary nesting.

Run tests using pytest:
```bash
pytest -v
```

Output:
```text
============================= test session starts =============================
collected 21 items

test_compaction.py::TestSelectMessagesToCompact::test_invalid_threshold_types PASSED
test_compaction.py::TestSelectMessagesToCompact::test_invalid_keep_recent_types PASSED
test_compaction.py::TestSelectMessagesToCompact::test_keep_recent_greater_than_threshold PASSED
test_compaction.py::TestSelectMessagesToCompact::test_keep_recent_equal_threshold PASSED
test_compaction.py::TestSelectMessagesToCompact::test_at_threshold_no_compaction PASSED
test_compaction.py::TestSelectMessagesToCompact::test_below_threshold_no_compaction PASSED
test_compaction.py::TestSelectMessagesToCompact::test_above_threshold_compaction PASSED
test_compaction.py::TestSelectMessagesToCompact::test_empty_messages PASSED
test_compaction.py::TestBuildSummaryInput::test_no_existing_summary PASSED
test_compaction.py::TestBuildSummaryInput::test_empty_existing_summary PASSED
test_compaction.py::TestBuildSummaryInput::test_whitespace_existing_summary PASSED
test_compaction.py::TestBuildSummaryInput::test_valid_existing_summary PASSED
test_compaction.py::TestBuildSummaryInput::test_generator_messages PASSED
test_compaction.py::TestBuildSummaryInput::test_summary_not_in_older_messages PASSED
test_compaction.py::TestCompactMemory::test_no_compaction_at_threshold PASSED
test_compaction.py::TestCompactMemory::test_no_compaction_below_threshold PASSED
test_compaction.py::TestCompactMemory::test_compaction_called_exactly_once PASSED
test_compaction.py::TestCompactMemory::test_immutability_caller_state_preserved PASSED
test_compaction.py::TestCompactMemory::test_immutability_on_no_compaction PASSED
test_compaction.py::TestCompactMemory::test_invalid_summary_outputs PASSED
test_compaction.py::TestCompactMemory::test_repeated_compaction PASSED

============================= 21 passed in 0.05s ==============================
```

---

## 📁 Repository Structure

```
CompactLongConversationState/
├── main.py              # Core logic: select_messages_to_compact, build_summary_input, compact_memory
├── requirements.txt     # Dependencies (pytest, groq)
├── test_compaction.py   # Full integration test suite
├── utils/
│   ├── __init__.py
│   ├── models.py        # Frozen dataclasses (MemoryState, MessageRecord, SummaryRecord, etc.)
│   └── summarizer.py    # Production Groq LLM summarizer client
├── .gitignore
└── README.md
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
