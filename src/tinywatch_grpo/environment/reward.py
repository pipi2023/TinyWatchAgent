"""Reward v1: deterministic terminal scoring for TinyWatch lists."""

from __future__ import annotations

from tinywatch_grpo.environment.tools import REWARD_VERSION


def compile_requirements(task: dict) -> dict:
    constraints = task["constraints"]
    return {
        "n_movies": int(constraints["n_movies"]),
        "max_total_runtime": int(constraints["max_total_runtime"]),
        "genres": list(constraints.get("genres") or []),
        "year_min": constraints.get("year_min"),
        "year_max": constraints.get("year_max"),
        "min_rating": float(constraints.get("min_rating") or 0.0),
        "directors": list(constraints.get("directors") or []),
        "require_zh": bool(constraints.get("require_zh")),
        "solvable": bool(task.get("solvable", True)),
    }


def selected_movies(draft: dict, catalog) -> list[dict]:
    movies = []
    for movie_id in draft.get("movie_ids") or []:
        movie = catalog.movies.get(movie_id)
        if movie is not None:
            movies.append(movie)
    return movies


def hard_gates(task: dict, draft: dict, catalog) -> dict:
    req = compile_requirements(task)
    movies = selected_movies(draft, catalog)
    n_ok = len(movies) == req["n_movies"] and len({movie["movie_id"] for movie in movies}) == len(movies)
    total_runtime = sum(int(movie.get("runtime") or 0) for movie in movies)
    runtime_ok = bool(movies) and total_runtime <= req["max_total_runtime"]
    rating_ok = all(float(movie.get("rating") or 0.0) >= req["min_rating"] for movie in movies) if movies else False
    year_ok = True
    for movie in movies:
        year = int(movie.get("year") or 0)
        if req["year_min"] is not None and year < int(req["year_min"]):
            year_ok = False
        if req["year_max"] is not None and year > int(req["year_max"]):
            year_ok = False
    if not movies:
        year_ok = False
    genre_set = set()
    for movie in movies:
        genre_set.update(movie.get("genres") or [])
    genre_ok = all(genre in genre_set for genre in req["genres"]) if movies else False
    director_ok = True
    if req["directors"]:
        chosen = set()
        for movie in movies:
            chosen.update(movie.get("directors") or [])
        director_ok = all(name in chosen for name in req["directors"]) if movies else False
    zh_ok = True
    if req["require_zh"]:
        zh_ok = all(bool(movie.get("has_zh_aka")) for movie in movies) if movies else False
    has_movies = bool(movies)
    return {
        "n_ok": n_ok,
        "runtime_ok": runtime_ok,
        "rating_ok": rating_ok,
        "year_ok": year_ok,
        "genre_ok": genre_ok,
        "director_ok": director_ok,
        "zh_ok": zh_ok,
        "has_movies": has_movies,
        "total_runtime": total_runtime,
        "all_hard": bool(
            n_ok and runtime_ok and rating_ok and year_ok and genre_ok and director_ok and zh_ok and has_movies
        ),
    }


def evidence_valid(task: dict, draft: dict, inspected_ids: set[str], crew_ids: set[str]) -> bool:
    """finalize must have opened/compared each pick; director hard also needs view_crew."""
    movie_ids = list(draft.get("movie_ids") or [])
    if not movie_ids:
        return False
    if not all(movie_id in inspected_ids for movie_id in movie_ids):
        return False
    directors = list((task.get("constraints") or {}).get("directors") or [])
    if directors and not all(movie_id in crew_ids for movie_id in movie_ids):
        return False
    return True


def gold_match(task: dict, draft: dict) -> bool:
    gold = task.get("gold_watchlist") or {}
    gold_ids = list(gold.get("movie_ids") or [])
    draft_ids = list(draft.get("movie_ids") or [])
    return bool(gold_ids) and gold_ids == draft_ids


def _partial_score(gates: dict) -> float:
    keys = ("n_ok", "runtime_ok", "rating_ok", "year_ok", "genre_ok", "director_ok", "zh_ok", "has_movies")
    hits = sum(1 for key in keys if gates[key])
    return min(0.25, round(0.25 * hits / len(keys), 4))


def score_terminal(
    *,
    task: dict,
    draft: dict,
    catalog,
    inspected_ids: set[str],
    crew_ids: set[str],
    termination: str,
    search_count: int,
    open_count: int,
    loop: bool = False,
) -> dict:
    """Map a finished episode to Reward v1. Invalid evidence never enters SFT."""
    gates = hard_gates(task, draft, catalog)
    req = compile_requirements(task)
    valid_evidence = evidence_valid(task, draft, inspected_ids, crew_ids)
    gold = gold_match(task, draft)

    if loop:
        return _pack("loop", -0.65, True, gates, valid_evidence, gold, termination)
    if termination == "max_steps":
        return _pack("max_steps", -0.50, True, gates, valid_evidence, gold, termination)

    if termination == "abort":
        sufficient = search_count >= 2 and open_count >= 2
        if req["solvable"] is False and sufficient:
            return _pack("correct_abort", 0.50, True, gates, valid_evidence, gold, termination)
        if sufficient and not gates["all_hard"]:
            return _pack("correct_abort", 0.50, True, gates, valid_evidence, gold, termination)
        return _pack("early_abort", -0.35, True, gates, valid_evidence, gold, termination)

    if termination != "finalize_watchlist":
        return _pack("incomplete", -0.50, True, gates, valid_evidence, gold, termination)

    if not valid_evidence:
        return _pack("reward_unverifiable", 0.0, False, gates, valid_evidence, gold, termination)

    if not gates["has_movies"] or not gates["n_ok"]:
        return _pack("incomplete_watchlist", -0.85, True, gates, valid_evidence, gold, termination)
    if not gates["runtime_ok"] or not gates["rating_ok"] or not gates["director_ok"]:
        return _pack("wrong_watchlist", -0.85, True, gates, valid_evidence, gold, termination)

    if gold and gates["all_hard"]:
        return _pack("gold_watchlist", 1.0, True, gates, valid_evidence, gold, termination)
    if gates["all_hard"]:
        return _pack("valid_alternative", 0.60, True, gates, valid_evidence, gold, termination)
    return _pack("partial", _partial_score(gates), True, gates, valid_evidence, gold, termination)


def _pack(reward_type, reward, valid, gates, evidence, gold, termination) -> dict:
    return {
        "reward_version": REWARD_VERSION,
        "reward_type": reward_type,
        "reward": float(reward),
        "reward_valid": bool(valid),
        "termination_reason": reward_type,
        "terminal_action": termination,
        "gold_match": bool(gold),
        "evidence_ok": bool(evidence),
        "hard_gates": {
            key: gates[key]
            for key in (
                "n_ok",
                "runtime_ok",
                "rating_ok",
                "year_ok",
                "genre_ok",
                "director_ok",
                "zh_ok",
                "has_movies",
                "all_hard",
            )
        },
        "total_runtime": gates["total_runtime"],
        "watchlist_success": bool(valid and reward_type == "gold_watchlist"),
    }
