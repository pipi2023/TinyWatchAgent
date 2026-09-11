"""Programmatic Oracle teacher: shortest legal path to gold_watchlist."""

from __future__ import annotations

import json

from tinywatch_grpo.environment.env import TinyWatchEnv


def run_oracle(env: TinyWatchEnv) -> dict:
    """Execute the Oracle policy inside a live environment."""
    result = env.reset()
    steps = []
    messages = [{"role": "user", "content": env.public_instruction()}]
    for index, (name, parameters) in enumerate(_oracle_plan(env)):
        if result["done"]:
            break
        call_id = f"oracle-{index}"
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
        "environment_version": result.get("environment_version"),
    }


def _oracle_plan(env: TinyWatchEnv) -> list[tuple[str, dict]]:
    task = env.task
    gold = task.get("gold_watchlist") or {}
    movie_ids = list(gold.get("movie_ids") or [])
    need_crew = bool((task.get("constraints") or {}).get("directors"))
    if not movie_ids or not task.get("solvable", True):
        fallback = next(iter(env.catalog.movies.values()))
        return [
            ("search_movies", {"query": fallback.get("title_zh") or fallback["title_en"]}),
            ("open", {"movie_id": fallback["movie_id"]}),
            ("search_movies", {"query": fallback.get("title_en") or fallback["movie_id"]}),
            ("open", {"movie_id": fallback["movie_id"]}),
            ("abort", {"reason": "no_feasible_watchlist"}),
        ]
    actions: list[tuple[str, dict]] = []
    for movie_id in movie_ids:
        movie = env.catalog.movies[movie_id]
        query = movie.get("title_zh") or movie.get("title_en") or movie["original_title"]
        actions.append(("search_movies", {"query": query}))
        actions.append(("open", {"movie_id": movie_id}))
        if need_crew:
            actions.append(("view_crew", {"movie_id": movie_id}))
        actions.append(("add_to_watchlist", {"movie_id": movie_id}))
    actions.append(("view_draft", {}))
    actions.append(("finalize_watchlist", {}))
    return actions
