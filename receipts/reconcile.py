"""Match each action claim to a tool call and classify the outcome.

Three verdicts:

* ``VERIFIED``    — a matching tool call exists and its result looks successful.
* ``PHANTOM``     — the assistant narrated the action but no tool call matches.
* ``SILENT_FAIL`` — a matching call exists but it errored or returned nothing,
  yet the assistant reported it as done.

Matching is intentionally interpretable: a claim's canonical action maps to a
set of tokens we expect in a tool name; object words break ties. It is a
heuristic — its accuracy is measured in :mod:`receipts.selfeval`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from .claims import Claim, extract_claims
from .trace import ToolCall, ToolResult, Trace


class Verdict(str, Enum):
    VERIFIED = "VERIFIED"
    PHANTOM = "PHANTOM"
    SILENT_FAIL = "SILENT_FAIL"


# Canonical action -> tokens we would expect to see inside a matching tool name.
_ACTION_TOOL_TOKENS: dict[str, frozenset[str]] = {
    "create": frozenset({"create", "add", "new", "make", "generate", "build", "post", "insert", "setup"}),
    "add": frozenset({"add", "create", "insert", "append", "grant", "assign", "attach"}),
    "delete": frozenset({"delete", "remove", "destroy", "drop", "purge", "clear", "del", "wipe"}),
    "update": frozenset({"update", "edit", "modify", "change", "set", "patch", "config", "configure", "rename", "adjust"}),
    "ban": frozenset({"ban"}),
    "kick": frozenset({"kick", "remove"}),
    "mute": frozenset({"mute", "timeout", "silence"}),
    "warn": frozenset({"warn", "flag", "warning"}),
    "send": frozenset({"send", "post", "dispatch", "notify", "message", "email", "dm", "ping", "reply"}),
    "fetch": frozenset({"fetch", "get", "read", "retrieve", "pull", "load", "query", "search", "lookup", "find", "list", "check"}),
    "merge": frozenset({"merge"}),
    "commit": frozenset({"commit", "push"}),
    "run": frozenset({"run", "execute", "exec", "invoke", "call", "trigger"}),
    "install": frozenset({"install"}),
    "save": frozenset({"save", "store", "persist", "write", "record"}),
    "assign": frozenset({"assign", "grant", "give", "add", "set"}),
    "revoke": frozenset({"revoke", "deny", "remove"}),
    "close": frozenset({"close", "resolve"}),
}

# Substrings that suggest a tool *result* is actually a failure, even when no
# explicit error field is set. Conservative on purpose; see README limitations.
_FAILURE_MARKERS: tuple[str, ...] = (
    "error", "errored", "failed", "failure", "exception", "traceback",
    "denied", "forbidden", "unauthorized", "not found", "could not",
    "couldn't", "couldnt", "unable to", "cannot ", "no such",
    "does not exist", "doesn't exist", "doesnt exist", "invalid",
    "timed out", "timeout", "rejected", "permission",
)

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _name_tokens(name: str) -> set[str]:
    """Split a tool name into lowercase tokens (snake_case + camelCase)."""
    spaced = _CAMEL_RE.sub(" ", name)
    return {t for t in _TOKEN_SPLIT_RE.split(spaced.lower()) if t}


def _arg_tokens(args: dict) -> set[str]:
    tokens: set[str] = set()
    for key, value in args.items():
        tokens.update(_TOKEN_SPLIT_RE.split(str(key).lower()))
        tokens.update(_TOKEN_SPLIT_RE.split(str(value).lower()))
    return {t for t in tokens if t}


@dataclass
class Finding:
    """The result of reconciling one claim against the trace."""

    claim: Claim
    verdict: Verdict
    matched_call: ToolCall | None = None
    matched_result: ToolResult | None = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "claim": self.claim.text,
            "action": self.claim.action,
            "verdict": self.verdict.value,
            "matched_tool": self.matched_call.name if self.matched_call else None,
            "matched_call_id": self.matched_call.id if self.matched_call else None,
            "reason": self.reason,
            "source": self.claim.source_sentence,
        }


@dataclass
class _Candidate:
    call: ToolCall
    verb_hit: bool
    object_overlap: int


def _result_is_failure(result: ToolResult | None) -> tuple[bool, str]:
    """Return (failed, reason) for a tool result."""
    if result is None:
        return False, "call made; no result recorded"
    if result.failed:
        return True, f"tool errored: {result.error!s}"[:200]
    text = (result.output or "").strip()
    if not text:
        return True, "tool returned empty output"
    low = text.lower()
    for marker in _FAILURE_MARKERS:
        if marker in low:
            return True, f"result looks like a failure (matched '{marker.strip()}')"
    return False, "call succeeded"


def _best_match(claim: Claim, calls: list[ToolCall], used: set[str]) -> ToolCall | None:
    """Pick the best *unused* tool call that matches ``claim``.

    A tool call must be verb-aligned with the claim (its name shares a token
    with the claim's action synonyms). Object-word overlap breaks ties. Each
    call backs at most one claim: one tool invocation == one real action, so a
    call already tied to an earlier claim cannot rescue a later one. This is
    what turns "I deleted the channel *and* the role" into VERIFIED + PHANTOM
    when only the channel was actually deleted.
    """
    expected = _ACTION_TOOL_TOKENS.get(claim.action, frozenset())
    obj = set(claim.object_tokens)
    candidates: list[_Candidate] = []
    for call in calls:
        if call.id in used:
            continue  # already backs another claim
        ntokens = _name_tokens(call.name)
        if not (expected & ntokens):
            continue  # verb must line up with the tool name
        overlap = len(obj & (ntokens | _arg_tokens(call.arguments)))
        candidates.append(_Candidate(call=call, verb_hit=True, object_overlap=overlap))

    if not candidates:
        return None

    # Prefer higher object overlap; stable order otherwise.
    candidates.sort(key=lambda c: c.object_overlap, reverse=True)
    return candidates[0].call


def reconcile(trace: Trace) -> list[Finding]:
    """Reconcile every action claim in ``trace`` against its tool calls."""
    calls = trace.tool_calls()
    results = trace.result_by_call_id()
    findings: list[Finding] = []
    used: set[str] = set()

    for claim in extract_claims(trace.assistant_text()):
        match = _best_match(claim, calls, used)
        if match is None:
            findings.append(
                Finding(
                    claim=claim,
                    verdict=Verdict.PHANTOM,
                    reason="no tool call matches this narrated action",
                )
            )
            continue

        used.add(match.id)
        result = results.get(match.id)
        failed, reason = _result_is_failure(result)
        verdict = Verdict.SILENT_FAIL if failed else Verdict.VERIFIED
        findings.append(
            Finding(
                claim=claim,
                verdict=verdict,
                matched_call=match,
                matched_result=result,
                reason=reason,
            )
        )

    return findings
