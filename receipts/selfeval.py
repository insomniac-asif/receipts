"""Measure the checker's own accuracy against a hand-labeled fixture set.

On-brand honesty: instead of claiming the heuristics are perfect, we ship a
small gold set and *measure* how well ``reconcile`` reproduces it. Each gold
item describes an action the labeler knows is present, by keyword(s) plus the
true verdict. Evaluation reports:

* **extraction precision/recall** — did we find the right claims (and only
  those)?
* **classification accuracy** — of the claims we matched to gold, did we assign
  the right verdict?
* **per-verdict precision/recall** — treating PHANTOM / SILENT_FAIL detection as
  the thing that actually matters.

No model, no network. Run ``python -m receipts.selfeval`` to print the scorecard.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from .reconcile import Finding, Verdict, reconcile
from .trace import load_trace

FIXTURE_GLOB = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "fixtures", "labeled", "*.json"
)


@dataclass
class GoldClaim:
    keywords: list[str]
    verdict: str

    def matches(self, finding: Finding) -> bool:
        hay = f"{finding.claim.text} {finding.claim.source_sentence}".lower()
        return all(k.lower() in hay for k in self.keywords)


@dataclass
class Scorecard:
    extraction_tp: int = 0
    extraction_fp: int = 0
    extraction_fn: int = 0
    classification_correct: int = 0
    classification_total: int = 0
    # confusion for the two "problem" verdicts we care most about
    per_verdict_tp: dict[str, int] = field(default_factory=dict)
    per_verdict_fp: dict[str, int] = field(default_factory=dict)
    per_verdict_fn: dict[str, int] = field(default_factory=dict)

    @property
    def extraction_precision(self) -> float:
        denom = self.extraction_tp + self.extraction_fp
        return self.extraction_tp / denom if denom else 1.0

    @property
    def extraction_recall(self) -> float:
        denom = self.extraction_tp + self.extraction_fn
        return self.extraction_tp / denom if denom else 1.0

    @property
    def classification_accuracy(self) -> float:
        return (
            self.classification_correct / self.classification_total
            if self.classification_total
            else 1.0
        )

    def verdict_precision(self, v: str) -> float:
        tp = self.per_verdict_tp.get(v, 0)
        fp = self.per_verdict_fp.get(v, 0)
        return tp / (tp + fp) if (tp + fp) else 1.0

    def verdict_recall(self, v: str) -> float:
        tp = self.per_verdict_tp.get(v, 0)
        fn = self.per_verdict_fn.get(v, 0)
        return tp / (tp + fn) if (tp + fn) else 1.0

    def to_dict(self) -> dict:
        return {
            "extraction_precision": round(self.extraction_precision, 4),
            "extraction_recall": round(self.extraction_recall, 4),
            "classification_accuracy": round(self.classification_accuracy, 4),
            "per_verdict": {
                v: {
                    "precision": round(self.verdict_precision(v), 4),
                    "recall": round(self.verdict_recall(v), 4),
                }
                for v in (
                    Verdict.VERIFIED.value,
                    Verdict.PHANTOM.value,
                    Verdict.SILENT_FAIL.value,
                )
            },
        }

    def render(self) -> str:
        d = self.to_dict()
        lines = [
            "receipts self-evaluation (heuristics vs. hand labels)",
            "-" * 52,
            f"  extraction precision : {d['extraction_precision']:.2f}",
            f"  extraction recall    : {d['extraction_recall']:.2f}",
            f"  classification acc.  : {d['classification_accuracy']:.2f}",
            "  per-verdict (precision / recall):",
        ]
        for v, pr in d["per_verdict"].items():
            lines.append(f"    {v:<12} {pr['precision']:.2f} / {pr['recall']:.2f}")
        return "\n".join(lines)


def load_gold(paths: list[str] | None = None) -> list[tuple[dict, list[GoldClaim]]]:
    """Load labeled fixtures as ``(trace_dict, gold_claims)`` pairs."""
    files = paths if paths is not None else sorted(glob.glob(FIXTURE_GLOB))
    dataset: list[tuple[dict, list[GoldClaim]]] = []
    for path in files:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        gold = [
            GoldClaim(keywords=g["keywords"], verdict=g["verdict"])
            for g in raw.get("gold_claims", [])
        ]
        dataset.append((raw["trace"], gold))
    return dataset


def evaluate(dataset: list[tuple[dict, list[GoldClaim]]]) -> Scorecard:
    """Score ``reconcile`` against the gold dataset."""
    sc = Scorecard()
    for trace_dict, gold in dataset:
        trace = load_trace(trace_dict)
        findings = reconcile(trace)

        matched_findings: set[int] = set()
        for g in gold:
            hit = None
            for idx, f in enumerate(findings):
                if idx in matched_findings:
                    continue
                if g.matches(f):
                    hit = (idx, f)
                    break
            if hit is None:
                sc.extraction_fn += 1
                sc.per_verdict_fn[g.verdict] = sc.per_verdict_fn.get(g.verdict, 0) + 1
                continue

            idx, f = hit
            matched_findings.add(idx)
            sc.extraction_tp += 1
            sc.classification_total += 1
            predicted = f.verdict.value
            if predicted == g.verdict:
                sc.classification_correct += 1
                sc.per_verdict_tp[g.verdict] = sc.per_verdict_tp.get(g.verdict, 0) + 1
            else:
                sc.per_verdict_fn[g.verdict] = sc.per_verdict_fn.get(g.verdict, 0) + 1
                sc.per_verdict_fp[predicted] = sc.per_verdict_fp.get(predicted, 0) + 1

        # Any finding not matched to a gold claim is an extraction false positive.
        sc.extraction_fp += len(findings) - len(matched_findings)
    return sc


def run() -> Scorecard:
    return evaluate(load_gold())


if __name__ == "__main__":  # pragma: no cover
    print(run().render())
