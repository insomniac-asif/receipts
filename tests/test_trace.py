"""Tests for the data model and both loaders."""

from receipts.trace import (
    Trace,
    from_openai_messages,
    load_trace,
)


def test_load_native_schema():
    data = {
        "name": "t",
        "turns": [
            {
                "role": "assistant",
                "text": "banning now",
                "tool_calls": [{"id": "c1", "name": "ban_user", "arguments": {"user_id": "5"}}],
            },
            {"role": "tool", "results": [{"tool_call_id": "c1", "output": "banned"}]},
        ],
    }
    trace = load_trace(data)
    assert isinstance(trace, Trace)
    assert len(trace.tool_calls()) == 1
    assert trace.tool_calls()[0].name == "ban_user"
    assert trace.result_by_call_id()["c1"].output == "banned"
    assert not trace.result_by_call_id()["c1"].failed


def test_native_schema_bare_list():
    trace = load_trace([{"role": "assistant", "text": "hi"}])
    assert len(trace.turns) == 1


def test_native_json_string_arguments():
    data = {
        "turns": [
            {
                "role": "assistant",
                "tool_calls": [{"id": "c1", "name": "f", "arguments": "{\"a\": 1}"}],
            }
        ]
    }
    trace = load_trace(data)
    assert trace.tool_calls()[0].arguments == {"a": 1}


def test_error_field_marks_failure():
    data = {
        "turns": [
            {"role": "tool", "results": [{"tool_call_id": "c1", "error": "boom"}]}
        ]
    }
    trace = load_trace(data)
    assert trace.result_by_call_id()["c1"].failed


def test_openai_adapter_basic():
    messages = [
        {"role": "user", "content": "ban them"},
        {
            "role": "assistant",
            "content": "banning",
            "tool_calls": [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "ban_user", "arguments": "{\"user_id\": \"9\"}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_abc", "content": "User 9 banned."},
        {"role": "assistant", "content": "done, i banned the user"},
    ]
    trace = from_openai_messages(messages, name="oa")
    calls = trace.tool_calls()
    assert len(calls) == 1
    assert calls[0].name == "ban_user"
    assert calls[0].arguments == {"user_id": "9"}
    # The tool result is routed back so it is discoverable by call id.
    assert trace.result_by_call_id()["call_abc"].output == "User 9 banned."
    assert "banned the user" in trace.assistant_text()


def test_openai_adapter_is_error_flag():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c1", "function": {"name": "update_config", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "nope", "is_error": True},
    ]
    trace = from_openai_messages(messages)
    assert trace.result_by_call_id()["c1"].failed


def test_openai_adapter_structured_content():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "i created the role"}],
        }
    ]
    trace = from_openai_messages(messages)
    assert "created the role" in trace.assistant_text()
