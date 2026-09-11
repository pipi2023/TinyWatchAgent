"""Observation v1: Actor-visible text, truncated, never leaking gold or reward."""

from __future__ import annotations

from tinywatch_grpo.environment.tools import OBS_CHAR_BUDGET, OBSERVATION_VERSION


def truncate_observation(text: str, budget: int = OBS_CHAR_BUDGET) -> tuple[str, bool]:
    if len(text) <= budget:
        return text, False
    keep = max(budget - 24, 32)
    return text[:keep] + "\n…[obs truncated]", True


def render_movie_line(index: int, movie: dict, *, detail: bool = False) -> str:
    title = movie.get("title_zh") or movie.get("title_en") or movie["original_title"]
    genres = ",".join(movie.get("genres") or [])
    line = (
        f"{index}|{movie['movie_id']}|{title}|{movie.get('year', '?')}|"
        f"{movie.get('rating', '?')}|{genres}"
    )
    if detail:
        line += f"|{movie.get('runtime', '?')}min|{movie.get('votes', '?')}票"
    return line


def render_draft(env_state: dict, catalog) -> str:
    task = env_state["task"]
    constraints = task["constraints"]
    n_movies = int(constraints["n_movies"])
    selected_ids = list(env_state["draft"].get("movie_ids") or [])
    lines = [
        f"目标部数: {n_movies}",
        f"片长预算: {constraints['max_total_runtime']} 分钟",
        f"最低评分: {constraints.get('min_rating')}",
        f"类型要求: {','.join(constraints.get('genres') or []) or '不限'}",
    ]
    year_min = constraints.get("year_min")
    year_max = constraints.get("year_max")
    if year_min or year_max:
        lines.append(f"年份: {year_min or '?'}–{year_max or '?'}")
    directors = constraints.get("directors") or []
    if directors:
        lines.append("导演硬约束: " + "、".join(directors))
    if not selected_ids:
        lines.append("片单: 空")
        lines.append("总片长: 0")
        return "\n".join(lines)
    total_runtime = 0
    genre_set = set()
    for offset, movie_id in enumerate(selected_ids, start=1):
        movie = catalog.movies[movie_id]
        total_runtime += int(movie.get("runtime") or 0)
        genre_set.update(movie.get("genres") or [])
        title = movie.get("title_zh") or movie.get("title_en") or movie["original_title"]
        lines.append(
            f"{offset}|{movie_id}|{title}|{movie.get('year')}|{movie.get('runtime')}min|"
            f"{movie.get('rating')}"
        )
    lines.append(f"总片长: {total_runtime}")
    lines.append("已覆盖类型: " + (",".join(sorted(genre_set)) if genre_set else "无"))
    return "\n".join(lines)


def render_observation(page: dict, env_state: dict, catalog) -> tuple[str, dict]:
    """Build the Actor-visible page plus a small projection audit payload."""
    header = [
        f"observation_version: {OBSERVATION_VERSION}",
        f"步数: {env_state['step']}/{env_state['max_steps']}",
        f"页面: {page.get('page_type', 'unknown')}",
        f"任务: {env_state['task']['query']}",
    ]
    body = page.get("body") or ""
    draft = "当前片单:\n" + render_draft(env_state, catalog)
    footer = (
        "可调用工具: search_movies, open, view_crew, compare, "
        "add_to_watchlist, view_draft, finalize_watchlist, abort"
    )
    text = "\n".join(header) + "\n\n" + body.strip() + "\n\n" + draft + "\n" + footer
    visible, truncated = truncate_observation(text)
    projection = {
        "truncated": truncated,
        "raw_chars": len(text),
        "visible_chars": len(visible),
        "schema_version": OBSERVATION_VERSION,
    }
    return visible, projection
