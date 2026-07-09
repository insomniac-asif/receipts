# Contributing to receipts

Thanks for helping make agent action-claims verifiable. `receipts` is small,
dependency-free, and fully offline by design — please keep it that way.

## Dev setup

```bash
git clone https://github.com/insomniac-asif/receipts
cd receipts
pip install -e ".[dev]"
python -m pytest -q      # must be green — no network, no model, no GPU
```

## Good first contributions

- **Trace adapters** — add a loader for another agent framework (Anthropic tool-use,
  LangChain/LangGraph, OpenTelemetry spans) next to `from_openai_messages` in `trace.py`.
- **Action verbs** — the claim extractor uses a curated past-tense verb table in
  `claims.py`. Missing a common one (`pinned`, `archived`, `deployed`)? Add it with a test.
- **Harder fixtures** — the honesty of this tool rests on its self-eval. Add tricky
  traces to `fixtures/labeled/` with gold labels; `receipts selfeval` scores against them
  and the suite enforces the floor. A PR that makes the tool *fail* first (then fixes it) is gold.

## The bar

- Tests stay green and offline; every behavior change ships with a test.
- Keep it dependency-free and inspectable — no ML in the core path.
- Be honest in the docs. If a change trades recall for precision (or vice versa), say so —
  and never report a number you can't reproduce with `receipts selfeval`.
