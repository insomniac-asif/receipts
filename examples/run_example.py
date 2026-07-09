"""Minimal end-to-end example.

Run it:  python examples/run_example.py
Or via the CLI:  receipts check examples/demo_trace.json
"""

from receipts import check_trace, load_trace_file

trace = load_trace_file("examples/demo_trace.json")
report = check_trace(trace)
print(report.render())
print()
print(f"phantom_rate for this run: {report.phantom_rate:.0%}")
