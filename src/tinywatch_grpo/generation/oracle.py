"""Programmatic Oracle teacher: shortest legal path to gold_watchlist."""

from __future__ import annotations

import json

from tinywatch_grpo.collection.agent_loop import SYSTEM_PROMPT
from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.environment.tools import ENVIRONMENT_VERSION, REWARD_VERSION, TINYWATCH_TOOL_SCHEMAS


def run_oracle(env: TinyWatchEnv) -> dict:
    """Execute the Oracle policy inside a live environment."""
    result = env.reset()
    steps = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": env.public_instruction()},
    ]

    def emit(name: str, parameters: dict) -> dict:
        nonlocal result
        call_id = f"oracle-{len(steps)}"
        tool_call = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(parameters, ensure_ascii=False)},
        }
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        result = env.step(name, parameters)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": result["observation"],
            }
        )
        steps.append(
            {
                "tool_name": name,
                "parameters": parameters,
                "observation": result["observation"],
                "reward": result["reward"],
                "done": result["done"],
                "guard_rejected": result.get("guard_rejected", False),
            }
        )
        return result

    for name, parameters in _oracle_plan(env):
        if result["done"]:
            break
        emit(name, parameters)

    if not result["done"] and env.task.get("solvable", True):
        for movie_id in list((env.task.get("gold_watchlist") or {}).get("movie_ids") or []):
            if result["done"]:
                break
            _retrieve_and_add(env, movie_id, emit)
        if not result["done"]:
            emit("view_draft", {})
        if not result["done"]:
            emit("finalize_watchlist", {})

    terminal = result.get("terminal_result") or {}
    return {
        "task_id": env.task["task_id"],
        "status": "done" if result["done"] else "incomplete",
        "done": result["done"],
        "final_reward": result["reward"],
        "messages": messages,
        "steps": steps,
        "terminal_result": terminal,
        "teacher": "oracle",
        "environment_version": result.get("environment_version") or ENVIRONMENT_VERSION,
        "reward_version": REWARD_VERSION,
        "oracle": True,
        "tools": TINYWATCH_TOOL_SCHEMAS,
        "error": None,
    }


def oracle_rollout(task: dict, catalog: dict, *, max_steps: int = 14) -> dict:
    """Run the Oracle policy and return a GRPO-compatible trajectory."""
    env = TinyWatchEnv(catalog, task, max_steps=max_steps)
    raw = run_oracle(env)
    terminal = raw.get("terminal_result") or {}
    return {
        "trajectory_id": f"oracle-{task.get('task_id')}",
        "task_id": task.get("task_id"),
        "status": raw.get("status") or "done",
        "done": bool(raw.get("done")),
        "error": raw.get("error"),
        "traceback": None,
        "final_reward": raw.get("final_reward", 0.0),
        "messages": raw.get("messages") or [],
        "steps": raw.get("steps") or [],
        "blocked_tool_calls": [],
        "tool_call_truncations": [],
        "terminal_result": terminal,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "teacher_model": "oracle",
        "oracle": True,
        "tools": TINYWATCH_TOOL_SCHEMAS,
        "thinking": False,
        "collected_at": None,
        "environment_version": raw.get("environment_version") or ENVIRONMENT_VERSION,
        "reward_version": raw.get("reward_version") or REWARD_VERSION,
        "elapsed_s": None,
    }


def _public_search_params(task: dict, movie: dict | None = None) -> dict:
    constraints = task.get("constraints") or {}
    directors = list(constraints.get("directors") or [])
    genres = list(constraints.get("genres") or [])
    query = directors[0] if directors else (genres[0] if genres else None)
    if not query and movie is not None:
        query = movie.get("title_zh") or movie.get("title_en") or movie.get("original_title")
    params = {"query": query or "电影"}
    if genres:
        params["genre"] = genres[0]
    if constraints.get("year_min") is not None:
        params["year_min"] = constraints["year_min"]
    if constraints.get("year_max") is not None:
        params["year_max"] = constraints["year_max"]
    if constraints.get("min_rating") is not None:
        params["min_rating"] = constraints["min_rating"]
    return params


def _retrieve_and_add(env: TinyWatchEnv, movie_id: str, emit) -> None:
    movie = env.catalog.movies[movie_id]
    need_crew = bool((env.task.get("constraints") or {}).get("directors"))
    emit("search_movies", _public_search_params(env.task, movie))
    if movie_id not in env.visible_ids:
        title = movie.get("title_zh") or movie.get("title_en") or movie.get("original_title")
        emit("search_movies", {"query": title})
    emit("open", {"movie_id": movie_id})
    if need_crew:
        emit("view_crew", {"movie_id": movie_id})
    emit("add_to_watchlist", {"movie_id": movie_id})


def _oracle_plan(env: TinyWatchEnv) -> list[tuple[str, dict]]:
    task = env.task
    gold = task.get("gold_watchlist") or {}
    movie_ids = list(gold.get("movie_ids") or [])
    if not movie_ids or not task.get("solvable", True):
        constraints = task.get("constraints") or {}
        fallback = next(iter(env.catalog.movies.values()))
        first = _public_search_params(task, fallback)
        second = {
            "query": (constraints.get("genres") or ["剧情"])[0],
        }
        if constraints.get("year_min") is not None:
            second["year_min"] = constraints["year_min"]
        return [
            ("search_movies", first),
            ("open", {"movie_id": fallback["movie_id"]}),
            ("search_movies", second),
            ("open", {"movie_id": fallback["movie_id"]}),
            ("abort", {"reason": "no_feasible_watchlist"}),
        ]
    return []
