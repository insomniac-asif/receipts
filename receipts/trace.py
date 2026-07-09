"""Normalized trace data model + loaders.

A ``Trace`` is a flat list of ``Turn`` objects. Assistant turns carry natural
language ``text`` and zero or more ``ToolCall`` objects. Tool turns carry
``ToolResult`` objects that link back to a call via ``tool_call_id``.

Two loaders are provided:

* :func:`load_trace` — reads the simple native JSON schema (see ``README``).
* :func:`from_openai_messages` — adapts an OpenAI-Chat-style ``messages`` list.

Everything here is pure data wrangling: no network, no model calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool/function invocation requested by the assistant."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """The outcome of a :class:`ToolCall`, keyed by ``tool_call_id``.

    ``error`` is a non-empty string when the tool raised/returned an error.
    ``output`` is the (possibly empty) textual result.
    """

    tool_call_id: str
    output: str = ""
    error: str | None = None

    @property
    def failed(self) -> bool:
        """True when the tool explicitly reported an error."""
        return bool(self.error and str(self.error).strip())


@dataclass
class Turn:
    """One turn in a conversation trace."""

    role: str  # "assistant" | "tool" | "user" | "system"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    results: list[ToolResult] = field(default_factory=list)


@dataclass
class Trace:
    """An ordered list of turns plus optional metadata."""

    turns: list[Turn] = field(default_factory=list)
    name: str | None = None

    # -- convenience accessors -------------------------------------------------

    def assistant_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def tool_calls(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for turn in self.turns:
            calls.extend(turn.tool_calls)
        return calls

    def result_by_call_id(self) -> dict[str, ToolResult]:
        """Map every ``tool_call_id`` to its result (last one wins)."""
        index: dict[str, ToolResult] = {}
        for turn in self.turns:
            for res in turn.results:
                index[res.tool_call_id] = res
        return index

    def assistant_text(self) -> str:
        """All assistant prose concatenated, in order."""
        return "\n".join(t.text for t in self.assistant_turns() if t.text)


# ---------------------------------------------------------------------------
# Native JSON loader
# ---------------------------------------------------------------------------


def _coerce_tool_call(raw: dict[str, Any], idx: int) -> ToolCall:
    call_id = str(raw.get("id") or raw.get("tool_call_id") or f"call_{idx}")
    name = str(raw.get("name") or raw.get("tool") or "")
    args = raw.get("arguments", raw.get("args", {}))
    if isinstance(args, str):
        args = _loads_or_empty(args)
    if not isinstance(args, dict):
        args = {"_value": args}
    return ToolCall(id=call_id, name=name, arguments=args)


def _coerce_tool_result(raw: dict[str, Any], idx: int) -> ToolResult:
    call_id = str(raw.get("tool_call_id") or raw.get("id") or f"call_{idx}")
    output = raw.get("output", raw.get("content", ""))
    if output is None:
        output = ""
    if not isinstance(output, str):
        output = json.dumps(output, ensure_ascii=False)
    error = raw.get("error")
    if error is not None and not isinstance(error, str):
        error = json.dumps(error, ensure_ascii=False)
    return ToolResult(tool_call_id=call_id, output=output, error=error)


def load_trace(data: dict[str, Any] | list[Any]) -> Trace:
    """Build a :class:`Trace` from the native schema.

    Accepts either ``{"name": ..., "turns": [...]}`` or a bare list of turns.
    Each turn is ``{"role", "text"?, "tool_calls"?, "results"?}``.
    """

    if isinstance(data, list):
        raw_turns = data
        name = None
    else:
        raw_turns = data.get("turns", [])
        name = data.get("name")

    turns: list[Turn] = []
    for turn in raw_turns:
        role = str(turn.get("role", "assistant"))
        text = str(turn.get("text") or turn.get("content") or "")
        calls = [
            _coerce_tool_call(c, i)
            for i, c in enumerate(turn.get("tool_calls", []) or [])
        ]
        results = [
            _coerce_tool_result(r, i)
            for i, r in enumerate(turn.get("results", []) or [])
        ]
        turns.append(Turn(role=role, text=text, tool_calls=calls, results=results))
    return Trace(turns=turns, name=name)


def load_trace_file(path: str) -> Trace:
    """Load a native-schema trace from a JSON file on disk."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    trace = load_trace(data)
    if trace.name is None:
        trace.name = path
    return trace


# ---------------------------------------------------------------------------
# OpenAI-Chat adapter
# ---------------------------------------------------------------------------


def _loads_or_empty(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    except (json.JSONDecodeError, TypeError):
        return {}


def from_openai_messages(
    messages: list[dict[str, Any]], name: str | None = None
) -> Trace:
    """Adapt an OpenAI-Chat ``messages`` list into a :class:`Trace`.

    Handles the standard shape where assistant messages carry a ``tool_calls``
    array (``{"id", "function": {"name", "arguments": <json-string>}}``) and
    tool outputs arrive as separate ``{"role": "tool", "tool_call_id", ...}``
    messages. Tool results are attached to the assistant turn that made the
    call, so downstream reconciliation sees calls and results together.
    """

    turns: list[Turn] = []
    # Remember which assistant Turn owns each call id, so tool messages can be
    # routed back to it.
    owner_by_call_id: dict[str, Turn] = {}

    for msg in messages:
        role = msg.get("role", "")
        if role == "assistant":
            text = msg.get("content") or ""
            if not isinstance(text, str):
                text = _content_to_text(text)
            calls: list[ToolCall] = []
            for i, raw in enumerate(msg.get("tool_calls", []) or []):
                fn = raw.get("function", {}) if isinstance(raw, dict) else {}
                call_id = str(raw.get("id") or f"call_{i}")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = _loads_or_empty(args)
                if not isinstance(args, dict):
                    args = {"_value": args}
                call = ToolCall(id=call_id, name=str(fn.get("name", "")), arguments=args)
                calls.append(call)
            turn = Turn(role="assistant", text=text, tool_calls=calls)
            for call in calls:
                owner_by_call_id[call.id] = turn
            turns.append(turn)
        elif role == "tool":
            call_id = str(msg.get("tool_call_id") or "")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = _content_to_text(content)
            error = _detect_error_field(msg)
            result = ToolResult(tool_call_id=call_id, output=content, error=error)
            owner = owner_by_call_id.get(call_id)
            if owner is not None:
                owner.results.append(result)
            else:
                turns.append(Turn(role="tool", results=[result]))
        else:
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = _content_to_text(content)
            turns.append(Turn(role=str(role), text=content))

    return Trace(turns=turns, name=name)


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI structured content blocks into plain text."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
            else:
                parts.append(str(block))
        return " ".join(p for p in parts if p)
    return str(content)


def _detect_error_field(msg: dict[str, Any]) -> str | None:
    """Some adapters carry an explicit error flag on the tool message."""
    if msg.get("is_error") or msg.get("error"):
        err = msg.get("error")
        if err and isinstance(err, str):
            return err
        return "tool reported error"
    return None
