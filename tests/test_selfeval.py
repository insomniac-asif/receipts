"""The checker measures and publishes its own accuracy on a gold set.

These thresholds are intentionally conservative. The gold set is small and
hand-labeled; perfect scores on it do NOT mean the heuristics are perfect on
real traces (see the README "Limitations" section). The point of this test is
to keep the checker honest: if a change silently degrades extraction or
classification below these floors, CI fails.
"""

from receipts.selfeval import load_gold, evaluate


def test_gold_set_is_nontrivial():
    dataset = load_gold()
    assert len(dataset) >= 8
    total_gold = sum(len(gold) for _, gold in dataset)
    assert total_gold >= 12


def test_selfeval_meets_published_floors():
    sc = evaluate(load_gold())
    # Floors, not targets. Real measured values are printed by `receipts selfeval`.
    assert sc.extraction_precision >= 0.85
    assert sc.extraction_recall >= 0.85
    assert sc.classification_accuracy >= 0.85


def test_selfeval_detects_phantoms_and_silent_fails():
    sc = evaluate(load_gold())
    # We must actually catch the failure modes we advertise.
    assert sc.verdict_recall("PHANTOM") >= 0.75
    assert sc.verdict_recall("SILENT_FAIL") >= 0.75


def test_scorecard_serializes():
    sc = evaluate(load_gold())
    d = sc.to_dict()
    assert "extraction_precision" in d
    assert set(d["per_verdict"]) == {"VERIFIED", "PHANTOM", "SILENT_FAIL"}
