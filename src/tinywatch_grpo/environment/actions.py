"""Reject illegal tool calls before they mutate TinyWatch state."""

from __future__ import annotations

from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS, TOOL_NAMES


RUNTIME_GUARD_FIELD = "runtime_action_guard"

TOOL_ARGUMENT_NAMES = {
    tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
    for tool in TINYWATCH_TOOL_SCHEMAS
}


def listed_entity_ids(observation: str) -> list[str]:
    """IDs that appear as `|movie_id|` rows on the current page."""
    ids = []
    seen = set()
    for line in observation.splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        prefix = parts[0].strip()
        if not prefix.isdigit() and not prefix.rstrip(".").isdigit():
            continue
        entity_id = parts[1].strip()
        if entity_id and entity_id not in seen:
            seen.add(entity_id)
            ids.append(entity_id)
    return ids


def action_reject_reason(name, arguments, observation, *, visible_ids=None):
    """Return a reject code, or None if the call may execute."""
    if name not in TOOL_NAMES:
        return "unknown_tool"
    extra = sorted(set(arguments or {}) - TOOL_ARGUMENT_NAMES.get(name, set()))
    if extra:
        return "schema_extra_arguments:" + ",".join(extra)
    arguments = arguments or {}
    if name == "abort":
        if arguments.get("reason") != "no_feasible_watchlist":
            return "invalid_abort_reason"
        return None
    if name in {"view_draft", "finalize_watchlist"}:
        return None
    if name == "search_movies":
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "empty_query"
        return None
    allowed = set(visible_ids or listed_entity_ids(observation or ""))
    if name in {"open", "view_crew", "add_to_watchlist"}:
        movie_id = str(arguments.get("movie_id") or "")
        if movie_id not in allowed:
            return "id_not_in_previous_observation"
        return None
    if name == "compare":
        movie_ids = arguments.get("movie_ids")
        if not isinstance(movie_ids, list) or not 2 <= len(movie_ids) <= 4:
            return "compare_arity"
        if len(set(movie_ids)) != len(movie_ids):
            return "compare_duplicate_ids"
        missing = [item for item in movie_ids if str(item) not in allowed]
        if missing:
            return "id_not_in_previous_observation"
        return None
    return "unknown_or_invalid_tool"


def action_guard_tool_message(tool_call, reason, observation):
    allowed = listed_entity_ids(observation)
    hint = "可打开的影片: " + ", ".join(allowed[:20]) if allowed else "当前页没有列出影片"
    return {
        "role": "tool",
        "tool_call_id": tool_call.get("id"),
        "name": (tool_call.get("function") or {}).get("name"),
        RUNTIME_GUARD_FIELD: True,
        "content": (
            f"上一工具调用被本地动作守卫拒绝（{reason}），未执行。"
            f"下一步只能从当前页面列出的目标中选择。{hint}。"
            "每个 assistant 回合只调用一个合法工具。"
        ),
    }
