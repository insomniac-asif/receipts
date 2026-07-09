"""Turn findings into per-run and corpus-level reports."""

from __future__ import annotations

from dataclasses import dataclass, field

from .reconcile import Finding, Verdict, reconcile
from .trace import Trace

_ICON = {
    Verdict.VERIFIED: "OK  ",
    Verdict.PHANTOM: "??  ",
    Verdict.SILENT_FAIL: "XX  ",
}


@dataclass
class RunReport:
    """A report for a single trace/run."""

    name: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.findings)

    def count(self, verdict: Verdict) -> int:
        return sum(1 for f in self.findings if f.verdict == verdict)

    @property
    def verified(self) -> int:
        return self.count(Verdict.VERIFIED)

    @property
    def phantom(self) -> int:
        return self.count(Verdict.PHANTOM)

    @property
    def silent_fail(self) -> int:
        return self.count(Verdict.SILENT_FAIL)

    @property
    def phantom_rate(self) -> float:
        return self.phantom / self.total if self.total else 0.0

    @property
    def trustworthy(self) -> bool:
        """True when nothing the assistant claimed is unbacked or failed."""
        return self.phantom == 0 and self.silent_fail == 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "total_claims": self.total,
            "verified": self.verified,
            "phantom": self.phantom,
            "silent_fail": self.silent_fail,
            "phantom_rate": round(self.phantom_rate, 4),
            "findings": [f.to_dict() for f in self.findings],
        }

    def render(self) -> str:
        lines = [f"receipts: {self.name}", "=" * (9 + len(self.name))]
        if not self.findings:
            lines.append("no action claims found in assistant text.")
            return "\n".join(lines)
        for f in self.findings:
            lines.append(f"  [{_ICON[f.verdict]}] {f.verdict.value:<11} {f.claim.text}")
            detail = f.reason
            if f.matched_call:
                detail = f"-> {f.matched_call.name}  ({f.reason})"
            lines.append(f"            {detail}")
        lines.append("")
        lines.append(
            f"  {self.total} claim(s): "
            f"{self.verified} verified, {self.phantom} phantom, "
            f"{self.silent_fail} silent-fail  |  phantom_rate={self.phantom_rate:.0%}"
        )
        return "\n".join(lines)


def check_trace(trace: Trace) -> RunReport:
    """Reconcile a trace and wrap the findings in a :class:`RunReport`."""
    findings = reconcile(trace)
    return RunReport(name=trace.name or "trace", findings=findings)


@dataclass
class CorpusReport:
    """Aggregate over many runs."""

    runs: list[RunReport] = field(default_factory=list)

    @property
    def total_claims(self) -> int:
        return sum(r.total for r in self.runs)

    @property
    def phantom(self) -> int:
        return sum(r.phantom for r in self.runs)

    @property
    def silent_fail(self) -> int:
        return sum(r.silent_fail for r in self.runs)

    @property
    def verified(self) -> int:
        return sum(r.verified for r in self.runs)

    @property
    def phantom_rate(self) -> float:
        return self.phantom / self.total_claims if self.total_claims else 0.0

    @property
    def silent_fail_rate(self) -> float:
        return self.silent_fail / self.total_claims if self.total_claims else 0.0

    def to_dict(self) -> dict:
        return {
            "runs": len(self.runs),
            "total_claims": self.total_claims,
            "verified": self.verified,
            "phantom": self.phantom,
            "silent_fail": self.silent_fail,
            "phantom_rate": round(self.phantom_rate, 4),
            "silent_fail_rate": round(self.silent_fail_rate, 4),
            "per_run": [r.to_dict() for r in self.runs],
        }

    def render(self) -> str:
        lines = [r.render() for r in self.runs]
        lines.append("")
        lines.append("=" * 40)
        lines.append(
            f"corpus: {len(self.runs)} run(s), {self.total_claims} claim(s)"
        )
        lines.append(
            f"  verified={self.verified}  phantom={self.phantom}  "
            f"silent_fail={self.silent_fail}"
        )
        lines.append(
            f"  phantom_rate={self.phantom_rate:.1%}  "
            f"silent_fail_rate={self.silent_fail_rate:.1%}"
        )
        return "\n".join(lines)


def check_corpus(traces: list[Trace]) -> CorpusReport:
    return CorpusReport(runs=[check_trace(t) for t in traces])
