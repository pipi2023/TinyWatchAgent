"""Generate disjoint SFT / GRPO / Eval task pools from a frozen movie catalog."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tinywatch_grpo.environment.catalog import CatalogIndex
from tinywatch_grpo.environment.reward import hard_gates
from tinywatch_grpo.generation.vocab import QUERY_TEMPLATES


def _rng(seed: int):
    value = seed

    def next_int(n: int) -> int:
        nonlocal value
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF
        return value % n

    return next_int


def _pick(nxt, items):
    return items[nxt(len(items))]


def _movie_ok(movie: dict, constraints: dict) -> bool:
    if float(movie.get("rating") or 0.0) < float(constraints.get("min_rating") or 0.0):
        return False
    year = int(movie.get("year") or 0)
    if constraints.get("year_min") is not None and year < int(constraints["year_min"]):
        return False
    if constraints.get("year_max") is not None and year > int(constraints["year_max"]):
        return False
    if constraints.get("require_zh") and not movie.get("has_zh_aka"):
        return False
    return True


def _coverage(movies: list[dict]) -> tuple[set[str], set[str]]:
    genres = set()
    directors = set()
    for movie in movies:
        genres.update(movie.get("genres") or [])
        directors.update(movie.get("directors") or [])
    return genres, directors


def _build_gold(index: CatalogIndex, n_movies: int, genres: list[str], directors: list[str], nxt):
    pool = list(index.movies.values())
    for _ in range(40):
        chosen: list[dict] = []
        used = set()
        missing_genres = list(genres)
        missing_directors = list(directors)
        while (missing_genres or missing_directors) and len(chosen) < n_movies:
            if missing_genres:
                target = missing_genres[0]
                candidates = [
                    movie
                    for movie in index.by_genre.get(target, [])
                    if movie["movie_id"] not in used
                ]
            else:
                target = missing_directors[0]
                candidates = [
                    movie
                    for movie in index.by_director.get(target, [])
                    if movie["movie_id"] not in used
                ]
            if not candidates:
                break
            movie = _pick(nxt, candidates)
            chosen.append(movie)
            used.add(movie["movie_id"])
            have_g, have_d = _coverage(chosen)
            missing_genres = [genre for genre in genres if genre not in have_g]
            missing_directors = [name for name in directors if name not in have_d]
        while len(chosen) < n_movies:
            candidates = [movie for movie in pool if movie["movie_id"] not in used]
            if not candidates:
                break
            movie = _pick(nxt, candidates)
            chosen.append(movie)
            used.add(movie["movie_id"])
        if len(chosen) != n_movies:
            continue
        have_g, have_d = _coverage(chosen)
        if all(genre in have_g for genre in genres) and all(name in have_d for name in directors):
            return chosen
    return None


def _query(difficulty: str, constraints: dict, nxt) -> str:
    template = _pick(nxt, QUERY_TEMPLATES[difficulty])
    year_min = constraints.get("year_min")
    year_max = constraints.get("year_max")
    if year_min and year_max and int(year_min) == int(year_max):
        year_span = f"{year_min}年"
    elif year_min and year_max:
        year_span = f"{year_min}–{year_max}年"
    elif year_min:
        year_span = f"{year_min}年以后"
    elif year_max:
        year_span = f"{year_max}年以前"
    else:
        year_span = "年份不限"
    extras = []
    if constraints.get("directors"):
        extras.append("导演必须包括" + "、".join(constraints["directors"]))
    if constraints.get("require_zh"):
        extras.append("需要有中文译名")
    extra = "，".join(extras) if extras else "先检索再定稿"
    return template.format(
        n=constraints["n_movies"],
        genres="和".join(constraints["genres"]) or "不限类型",
        runtime=constraints["max_total_runtime"],
        rating=constraints["min_rating"],
        year_span=year_span,
        extra=extra,
    )


def generate_task(task_id: int, catalog: dict, *, difficulty: str, nxt, solvable: bool = True) -> dict:
    index = CatalogIndex(catalog)
    movies = list(catalog["movies"])
    if difficulty == "easy":
        n_movies = 1
        n_genres = 1
        with_director = True
        year_window = 0
    elif difficulty == "medium":
        n_movies = 1 + nxt(2)
        n_genres = 1 + nxt(2)
        with_director = nxt(3) != 0
        year_window = 20
    else:
        n_movies = 2
        n_genres = 2
        with_director = True
        year_window = 12
    gold_movies = None
    for _attempt in range(24):
        seed_movie = _pick(nxt, movies)
        if with_director and not seed_movie.get("directors"):
            continue
        genres = []
        available = list(seed_movie.get("genres") or [])
        for _ in range(min(n_genres, len(available) or 1)):
            if not available:
                break
            genres.append(available.pop(nxt(len(available))))
        if not genres:
            continue
        directors = []
        if with_director and seed_movie.get("directors"):
            directors = [seed_movie["directors"][0]]
        candidate = _build_gold(index, n_movies, genres, directors, nxt)
        if candidate is None:
            continue
        years = [int(movie["year"]) for movie in candidate]
        if year_window == 0:
            year_min = min(years)
            year_max = max(years)
        elif year_window:
            year_min = max(1960, min(years) - nxt(3))
            year_max = min(2024, max(years) + nxt(3))
        else:
            year_min = None
            year_max = None
        min_rating = round(min(float(movie["rating"]) for movie in candidate) - 0.1, 1)
        min_rating = max(6.5, min_rating)
        total_runtime = sum(int(movie["runtime"]) for movie in candidate)
        pad = 12 if difficulty == "easy" else 14 if difficulty == "medium" else 8
        max_total_runtime = total_runtime + pad
        require_zh = difficulty != "easy" and nxt(4) == 0
        if require_zh and not all(movie.get("has_zh_aka") for movie in candidate):
            require_zh = False
        constraints = {
            "n_movies": n_movies,
            "max_total_runtime": max_total_runtime,
            "genres": genres,
            "year_min": year_min,
            "year_max": year_max,
            "min_rating": min_rating,
            "directors": directors,
            "require_zh": require_zh,
        }
        if any(not _movie_ok(movie, constraints) for movie in candidate):
            continue
        gates = hard_gates(
            {"constraints": constraints, "solvable": True},
            {"movie_ids": [movie["movie_id"] for movie in candidate]},
            index,
        )
        if not gates["all_hard"]:
            continue
        gold_movies = candidate
        break
    if gold_movies is None:
        solvable = False
        constraints = {
            "n_movies": 2,
            "max_total_runtime": 80,
            "genres": ["纪录片"],
            "year_min": 2022,
            "year_max": 2024,
            "min_rating": 9.5,
            "directors": ["Nobody"],
            "require_zh": True,
        }
    elif not solvable:
        constraints["max_total_runtime"] = max(60, int(sum(int(m["runtime"]) for m in gold_movies) * 0.45))
        gold_movies = None
    return {
        "task_id": int(task_id),
        "difficulty": difficulty,
        "solvable": bool(solvable and gold_movies is not None),
        "query": _query(difficulty, constraints, nxt),
        "constraints": constraints,
        "gold_watchlist": (
            None
            if gold_movies is None
            else {"movie_ids": [movie["movie_id"] for movie in gold_movies]}
        ),
    }


def generate_splits(
    catalog: dict,
    *,
    seed: int = 42,
    sft_pool: int = 1000,
    grpo_train: int = 400,
    eval_holdout: int = 100,
) -> dict[str, list[dict]]:
    nxt = _rng(seed)
    total = sft_pool + grpo_train + eval_holdout
    tasks = []
    task_id = 1
    while len(tasks) < total:
        bucket = task_id % 10
        if bucket < 5:
            difficulty = "easy"
        elif bucket < 8:
            difficulty = "medium"
        else:
            difficulty = "hard"
        solvable = not (difficulty == "hard" and task_id % 17 == 0)
        tasks.append(
            generate_task(task_id, catalog, difficulty=difficulty, nxt=nxt, solvable=solvable)
        )
        task_id += 1
    eval_tasks = tasks[:eval_holdout]
    grpo_tasks = tasks[eval_holdout : eval_holdout + grpo_train]
    pool_tasks = tasks[eval_holdout + grpo_train :]
    return {
        "evaluation": eval_tasks,
        "grpo_train": grpo_tasks,
        "sft_pool": pool_tasks,
    }


def write_jsonl(path: str | Path, rows: list[dict]) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    text = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"path": str(path), "rows": len(rows), "sha256": digest}


def assert_disjoint(splits: dict[str, list[dict]]) -> None:
    seen: dict[int, str] = {}
    for name, rows in splits.items():
        for row in rows:
            task_id = int(row["task_id"])
            if task_id in seen:
                raise ValueError(f"task_id {task_id} in both {seen[task_id]} and {name}")
            seen[task_id] = name
