"""End-to-end CLI tests using the bundled example trace."""

import json
import os

from receipts.cli import main

EXAMPLE = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "examples", "demo_trace.json"
)


def test_check_exits_nonzero_on_problems(capsys):
    # The demo trace has a phantom and a silent-fail, so exit code is 1.
    code = main(["check", EXAMPLE])
    out = capsys.readouterr().out
    assert code == 1
    assert "PHANTOM" in out
    assert "SILENT_FAIL" in out


def test_check_no_fail_flag(capsys):
    code = main(["check", EXAMPLE, "--no-fail"])
    capsys.readouterr()
    assert code == 0


def test_check_json_output(capsys):
    code = main(["check", EXAMPLE, "--json"])
    out = capsys.readouterr().out
    assert code == 1
    payload = json.loads(out)
    assert payload["total_claims"] == 3
    assert payload["phantom"] == 1
    assert payload["silent_fail"] == 1


def test_selfeval_command(capsys):
    code = main(["selfeval", "--json"])
    out = capsys.readouterr().out
    assert code == 0
    payload = json.loads(out)
    assert 0.0 <= payload["extraction_precision"] <= 1.0
