import json

from tinywatch_grpo.collection.agent_loop import (
    _enforce_serial_tool_call,
    parse_tool_call,
    rollout_task,
)
from tinywatch_grpo.generation.task_gen import generate_task
from tinywatch_grpo.generation.vocab import mini_catalog


def _tool_call(name, arguments, call_id):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


class ScriptedClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.last_usage = {}

    def complete(self, messages, tools):
        del messages, tools
        return self.replies.pop(0)


def _easy_task():
    catalog = mini_catalog()
    state = [3]

    def nxt(n):
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] % n

    return catalog, generate_task(9, catalog, difficulty="easy", nxt=nxt, solvable=True)


def test_enforce_serial_keeps_first_tool_call():
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            _tool_call("search_movies", {"query": "a"}, "call_0"),
            _tool_call("open", {"movie_id": "tt1"}, "call_1"),
            _tool_call("view_crew", {"movie_id": "tt1"}, "call_2"),
        ],
    }
    serial, dropped = _enforce_serial_tool_call(assistant)
    assert [call["id"] for call in serial["tool_calls"]] == ["call_0"]
    assert [call["id"] for call in dropped] == ["call_1", "call_2"]


def test_parse_tool_call_uses_first_call_after_serial():
    parsed, error = parse_tool_call(
        {
            "tool_calls": [
                _tool_call("search_movies", {"query": "科幻"}, "call_0"),
                _tool_call("open", {"movie_id": "tt1"}, "call_1"),
            ]
        }
    )
    assert error is None
    assert parsed["name"] == "search_movies"
    assert parsed["arguments"] == {"query": "科幻"}


def test_rollout_serializes_multiple_tool_calls_before_execution():
    catalog, task = _easy_task()
    client = ScriptedClient(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("search_movies", {"query": f"query{index}"}, f"call_{index}")
                    for index in range(3)
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("search_movies", {"query": "after-obs"}, "call_after")
                ],
            },
        ]
    )
    trajectory = rollout_task(task, catalog, client, max_steps=2)
    assert trajectory["error"] != "multiple_tool_calls"
    assert len(trajectory["steps"]) == 2
    assert [step["tool_name"] for step in trajectory["steps"]] == [
        "search_movies",
        "search_movies",
    ]
    assert trajectory["steps"][0]["parameters"]["query"] == "query0"
    assert trajectory["steps"][1]["parameters"]["query"] == "after-obs"
    assistant = trajectory["messages"][2]
    assert assistant["role"] == "assistant"
    assert len(assistant["tool_calls"]) == 1
    assert assistant["tool_calls"][0]["id"] == "call_0"
    assert trajectory["tool_call_truncations"][0]["kept_tool_call_id"] == "call_0"
    assert [
        call["id"] for call in trajectory["tool_call_truncations"][0]["dropped_tool_calls"]
    ] == ["call_1", "call_2"]


def test_no_tool_call_retries_once_then_continues():
    catalog, task = _easy_task()
    gold_id = task["gold_watchlist"]["movie_ids"][0]
    title = catalog["movies"][0]["title_zh"]
    for movie in catalog["movies"]:
        if movie["movie_id"] == gold_id:
            title = movie.get("title_zh") or movie["title_en"]
            break
    client = ScriptedClient(
        [
            {"role": "assistant", "content": "我先想一下", "tool_calls": []},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("search_movies", {"query": title}, "call_retry")],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("open", {"movie_id": gold_id}, "call_open")],
            },
        ]
    )
    trajectory = rollout_task(task, catalog, client, max_steps=2)
    assert trajectory["error"] != "no_tool_call"
    assert trajectory["steps"][0]["tool_name"] == "search_movies"
    assert any(message.get("role") == "user" and "合法工具" in (message.get("content") or "") for message in trajectory["messages"])
