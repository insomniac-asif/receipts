"""receipts — check an agent's story against its tool calls.

Public API:

    from receipts import check_trace, load_trace, from_openai_messages

    trace = load_trace(json_data)
    report = check_trace(trace)
    print(report.render())
    print(report.phantom_rate)
"""

from __future__ import annotations

from .claims import Claim, extract_claims
from .reconcile import Finding, Verdict, reconcile
from .report import CorpusReport, RunReport, check_corpus, check_trace
from .trace import (
    ToolCall,
    ToolResult,
    Trace,
    Turn,
    from_openai_messages,
    load_trace,
    load_trace_file,
)

__version__ = "0.1.0"

__all__ = [
    "Claim",
    "extract_claims",
    "Finding",
    "Verdict",
    "reconcile",
    "RunReport",
    "CorpusReport",
    "check_trace",
    "check_corpus",
    "Trace",
    "Turn",
    "ToolCall",
    "ToolResult",
    "load_trace",
    "load_trace_file",
    "from_openai_messages",
    "__version__",
]
