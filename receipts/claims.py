"""Extract *action claims* from assistant prose with robust heuristics.

An *action claim* is a statement that the assistant **did something** to the
outside world: "I banned the user", "created the role", "sent them the docs".
We deliberately ignore intentions ("I will ban…"), questions, and negations
("I didn't delete it").

This is a heuristic extractor, not a parser. It uses a curated table of
past-tense action verbs mapped to a canonical action, so reconciliation can
line each claim up against a tool name. The accuracy of this step is measured
and published — see :mod:`receipts.selfeval` and the README.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical action -> the past-tense surface forms that signal it.
# Multi-word forms (e.g. "set up") are matched as bigrams.
_ACTION_FORMS: dict[str, tuple[str, ...]] = {
    "create": ("created", "made", "generated", "built", "set up", "initialized"),
    "add": ("added", "inserted", "appended", "attached"),
    "delete": (
        "deleted", "removed", "destroyed", "dropped", "purged",
        "cleared", "wiped", "deleted",
    ),
    "update": (
        "updated", "edited", "modified", "changed", "patched",
        "configured", "adjusted", "renamed", "set",
    ),
    "ban": ("banned",),
    "kick": ("kicked",),
    "mute": ("muted", "timed out", "silenced"),
    "warn": ("warned", "flagged"),
    "send": (
        "sent", "posted", "dispatched", "notified", "messaged",
        "emailed", "dmed", "pinged", "replied",
    ),
    "fetch": (
        "fetched", "retrieved", "pulled", "loaded", "queried",
        "searched", "looked up", "checked",
    ),
    "merge": ("merged",),
    "commit": ("committed", "pushed"),
    "run": ("ran", "executed", "invoked", "triggered"),
    "install": ("installed",),
    "save": ("saved", "stored", "persisted", "recorded"),
    "assign": ("assigned", "granted", "gave"),
    "revoke": ("revoked", "denied"),
    "close": ("closed", "resolved"),
}

# Build reverse lookups for single- and two-word surface forms.
_SURFACE_UNIGRAM: dict[str, str] = {}
_SURFACE_BIGRAM: dict[str, str] = {}
for _canon, _forms in _ACTION_FORMS.items():
    for _form in _forms:
        if " " in _form:
            _SURFACE_BIGRAM[_form] = _canon
        else:
            _SURFACE_UNIGRAM[_form] = _canon

# Tokens in the 3 words *before* a verb that cancel the claim: it is an
# intention, hypothetical, or negation rather than a completed action.
_BLOCKERS: frozenset[str] = frozenset(
    {
        "will", "ll", "going", "gonna", "would", "could", "should", "shall",
        "can", "cannot", "cant", "may", "might", "must",
        "not", "never", "didnt", "dont", "doesnt", "wont", "cant", "havent",
        "try", "trying", "tried", "attempt", "attempting", "want", "wanted",
        "need", "needs", "needed", "let", "lets", "please", "if", "unless",
        "to", "wanna", "hoping", "hope", "planning", "plan", "about",
        "supposed", "meant",
    }
)

# Words that cancel a claim only when they sit *immediately* before the verb.
# These catch idioms like "all set" / "everything sorted" without nuking a real
# verb a few tokens later (e.g. "all set - i updated the config").
_IMMEDIATE_BLOCKERS: frozenset[str] = frozenset({"all", "everything", "thats"})

# Articles / determiners stripped from the front of an object phrase.
_LEADING_STOP: frozenset[str] = frozenset(
    {"the", "a", "an", "my", "our", "your", "their", "his", "her", "its",
     "this", "that", "these", "those", "all", "some", "any", "them", "it",
     "up", "out", "in", "on", "for", "to", "of", "back", "you", "him"}
)

# Words that never count as object content (kept small on purpose).
_OBJECT_STOP: frozenset[str] = _LEADING_STOP | frozenset(
    {"and", "with", "from", "into", "over", "just", "now", "already",
     "successfully", "then", "so", "as", "well", "also"}
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"[.!?\n]+|\s+(?:and then|then|also)\s+|[,;]\s+")


@dataclass
class Claim:
    """A single extracted action claim."""

    action: str  # canonical action, e.g. "ban"
    verb: str  # surface verb as written, e.g. "banned"
    object_phrase: str  # short noun phrase, e.g. "the user"
    text: str  # readable reconstruction, e.g. "banned the user"
    source_sentence: str  # the clause it came from
    object_tokens: list[str] = field(default_factory=list)  # content words

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.text


def _clauses(text: str) -> list[str]:
    """Split prose into small clauses so multiple actions get separated."""
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _object_from(tokens: list[str], start: int) -> tuple[str, list[str]]:
    """Return (readable object phrase, content tokens) starting at ``start``."""
    phrase_words: list[str] = []
    content: list[str] = []
    # Skip leading determiners for the content set, but keep them in the phrase.
    for tok in tokens[start : start + 7]:
        phrase_words.append(tok)
        if tok not in _OBJECT_STOP and len(tok) > 1:
            content.append(tok)
        # Stop the phrase at a natural boundary once we have some content.
        if content and tok in {"and", "then"}:
            phrase_words.pop()
            break
    return " ".join(phrase_words).strip(), content


def extract_claims(text: str) -> list[Claim]:
    """Extract action claims from a block of assistant text."""
    claims: list[Claim] = []
    if not text:
        return claims

    for clause in _clauses(text):
        tokens = _WORD_RE.findall(clause.lower())
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            bigram = f"{tok} {tokens[i + 1]}" if i + 1 < len(tokens) else None

            canon: str | None = None
            span = 1
            if bigram and bigram in _SURFACE_BIGRAM:
                canon = _SURFACE_BIGRAM[bigram]
                span = 2
            elif tok in _SURFACE_UNIGRAM:
                canon = _SURFACE_UNIGRAM[tok]
                span = 1

            if canon is None:
                i += 1
                continue

            window = tokens[max(0, i - 3) : i]
            if any(w in _BLOCKERS for w in window):
                i += span
                continue
            if window and window[-1] in _IMMEDIATE_BLOCKERS:
                i += span
                continue

            obj_phrase, obj_tokens = _object_from(tokens, i + span)
            verb_surface = " ".join(tokens[i : i + span])
            # Require *some* object; a bare "I updated." is too vague to check.
            if not obj_tokens:
                i += span
                continue

            readable = f"{verb_surface} {obj_phrase}".strip()
            claims.append(
                Claim(
                    action=canon,
                    verb=verb_surface,
                    object_phrase=obj_phrase,
                    text=readable,
                    source_sentence=clause.strip(),
                    object_tokens=obj_tokens,
                )
            )
            i += span

    return claims
