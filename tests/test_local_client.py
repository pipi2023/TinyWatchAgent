from tinywatch_grpo.collection.local_client import (
    messages_for_chat_template,
    parse_generated_tool_calls,
)


def test_parse_json_tool_call_block():
    text = '<tool_call>\n{"name":"search_movies","arguments":{"query":"科幻"}}\n</tool_call>'
    calls = parse_generated_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search_movies"
    assert '"query": "科幻"' in calls[0]["function"]["arguments"] or '"query":"科幻"' in calls[0]["function"]["arguments"]


def test_parse_xml_function_tool_call():
    text = (
        "<function=open>\n"
        "<parameter=movie_id>\ntt0111161\n</parameter>\n"
        "</function>"
    )
    calls = parse_generated_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "open"
    assert "tt0111161" in calls[0]["function"]["arguments"]


def test_messages_for_chat_template_parses_argument_strings():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "search_movies",
                        "arguments": '{"query":"x"}',
                    },
                }
            ],
        }
    ]
    rendered = messages_for_chat_template(messages)
    assert isinstance(rendered[0]["tool_calls"][0]["function"]["arguments"], dict)
    assert rendered[0]["tool_calls"][0]["function"]["arguments"]["query"] == "x"
