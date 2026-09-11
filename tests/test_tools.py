from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS, TOOL_NAMES, tool_call_to_action


def test_schema_covers_all_tools():
    names = {item["function"]["name"] for item in TINYWATCH_TOOL_SCHEMAS}
    assert names == set(TOOL_NAMES)


def test_tool_call_to_action():
    action = tool_call_to_action("search_movies", {"query": "科幻"})
    assert action == {"name": "search_movies", "parameters": {"query": "科幻"}}
