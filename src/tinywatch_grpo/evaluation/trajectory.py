"""Normalize TinyWatch rollouts into a Judge-free event stream."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json


NORMALIZED_TRAJECTORY_VERSION = "tinywatch-normalized-trajectory-v1"


def _tool_call_payload(tool_call: object) -> tuple[str, dict, str | None]:
    if not isinstance(tool_call, Mapping):
        return "", {}, "tool_call_not_object"
    function = tool_call.get("function")
    if isinstance(function, Mapping):
        name = str(function.get("name") or "")
        raw = function.get("arguments", {})
    else:
        name = str(tool_call.get("name") or "")
        raw = tool_call.get("arguments", {})
    if isinstance(raw, Mapping):
        return name, deepcopy(dict(raw)), None
    if not isinstance(raw, str):
        return name, {}, "tool_call_arguments_invalid"
    try:
        arguments = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return name, {}, "tool_call_arguments_invalid_json"
    if not isinstance(arguments, dict):
        return name, {}, "tool_call_arguments_not_object"
    return name, arguments, None


def normalize_trajectory(trajectory: object) -> dict:
    if not isinstance(trajectory, Mapping):
        raise TypeError("trajectory must be an object")
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    events = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            continue
        name = str(step.get("tool_name") or "")
        parameters = step.get("parameters") if isinstance(step.get("parameters"), Mapping) else {}
        if not name and step.get("tool_call"):
            name, parameters, _error = _tool_call_payload(step.get("tool_call"))
        event_type = "guard_rejection" if step.get("guard_rejected") else "tool_step"
        events.append(
            {
                "event_id": index,
                "event_type": event_type,
                "tool_name": name,
                "parameters": deepcopy(dict(parameters)),
                "observation": step.get("observation") if event_type == "tool_step" else None,
                "done": bool(step.get("done")),
            }
        )
    terminal = trajectory.get("terminal_result") if isinstance(trajectory.get("terminal_result"), Mapping) else {}
    reward_detail = (
        deepcopy(dict(terminal.get("reward_detail")))
        if isinstance(terminal.get("reward_detail"), Mapping)
        else {}
    )
    watchlist = (
        deepcopy(dict(terminal.get("watchlist")))
        if isinstance(terminal.get("watchlist"), Mapping)
        else {}
    )
    return {
        "schema_version": NORMALIZED_TRAJECTORY_VERSION,
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": trajectory.get("task_id"),
        "status": trajectory.get("status"),
        "done": bool(trajectory.get("done")),
        "query": _query(trajectory),
        "events": events,
        "terminal": {
            "done": bool(terminal.get("done", trajectory.get("done"))),
            "over": bool(terminal.get("over", False)),
            "reward": float(terminal.get("reward", trajectory.get("final_reward") or 0.0) or 0.0),
            "reward_detail": reward_detail,
            "watchlist": watchlist,
        },
        "error": trajectory.get("error"),
    }


def _query(trajectory: Mapping) -> str:
    for message in trajectory.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
    return ""
