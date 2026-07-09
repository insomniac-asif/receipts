"""Tests for reconciliation / verdict classification."""

from receipts.reconcile import Verdict, reconcile
from receipts.report import check_trace
from receipts.trace import load_trace


def _verdicts(data):
    return [f.verdict for f in reconcile(load_trace(data))]


def test_verified_when_call_succeeds():
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "ban_user", "arguments": {"user_id": "1"}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": "banned"}]},
            {"role": "assistant", "text": "i banned the user"},
        ]
    }
    assert _verdicts(data) == [Verdict.VERIFIED]


def test_phantom_when_no_call():
    data = {"turns": [{"role": "assistant", "text": "i banned the user"}]}
    assert _verdicts(data) == [Verdict.PHANTOM]


def test_silent_fail_on_error_field():
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "update_config", "arguments": {}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "error": "denied"}]},
            {"role": "assistant", "text": "i updated the config"},
        ]
    }
    assert _verdicts(data) == [Verdict.SILENT_FAIL]


def test_silent_fail_on_empty_output():
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "fetch_logs", "arguments": {}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": ""}]},
            {"role": "assistant", "text": "i fetched the logs"},
        ]
    }
    assert _verdicts(data) == [Verdict.SILENT_FAIL]


def test_silent_fail_on_failure_marker_in_output():
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "delete_channel", "arguments": {"channel": "x"}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": "Error: permission denied"}]},
            {"role": "assistant", "text": "i deleted the channel"},
        ]
    }
    assert _verdicts(data) == [Verdict.SILENT_FAIL]


def test_each_call_backs_at_most_one_claim():
    # Only the channel was deleted; the role claim has no backing call.
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "delete_channel", "arguments": {"channel": "temp"}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": "deleted"}]},
            {"role": "assistant", "text": "i deleted the temp channel. i also removed the temp role."},
        ]
    }
    verdicts = _verdicts(data)
    assert Verdict.VERIFIED in verdicts
    assert Verdict.PHANTOM in verdicts


def test_report_counts_and_phantom_rate():
    data = {
        "name": "r",
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "ban_user", "arguments": {}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": "ok"}]},
            {"role": "assistant", "text": "i banned the user and posted a note in mod-log"},
        ],
    }
    report = check_trace(load_trace(data))
    assert report.total == 2
    assert report.verified == 1
    assert report.phantom == 1
    assert report.phantom_rate == 0.5
    assert report.trustworthy is False
