"""TinyWatch Environment v1: in-process tool-calling state machine."""

from __future__ import annotations

from copy import deepcopy

from tinywatch_grpo.environment.actions import action_reject_reason, listed_entity_ids
from tinywatch_grpo.environment.catalog import CatalogIndex
from tinywatch_grpo.environment.observation import render_movie_line, render_observation
from tinywatch_grpo.environment.reward import score_terminal
from tinywatch_grpo.environment.search import BM25Index, movie_document
from tinywatch_grpo.environment.tools import ENVIRONMENT_VERSION, MAX_STEPS_DEFAULT, REWARD_VERSION, SEARCH_TOP_K
from tinywatch_grpo.generation.vocab import normalize_genre


class TinyWatchEnv:
    """One task, one episode. Observations never include gold or reward."""

    def __init__(self, catalog: dict, task: dict, max_steps: int = MAX_STEPS_DEFAULT):
        self.catalog = CatalogIndex(catalog)
        self.task = deepcopy(task)
        self.max_steps = int(max_steps)
        self._bm25 = BM25Index(
            [(movie["movie_id"], movie_document(movie)) for movie in catalog["movies"]]
        )
        self._title_index = {}
        for movie in catalog["movies"]:
            for key in (movie.get("title_zh"), movie.get("title_en"), movie.get("original_title")):
                if key:
                    self._title_index.setdefault(str(key).casefold(), []).append(movie["movie_id"])
            for aka in movie.get("akas") or []:
                self._title_index.setdefault(str(aka).casefold(), []).append(movie["movie_id"])
        self._reset_state()

    def _reset_state(self):
        self.step_count = 0
        self.done = False
        self.over = False
        self.draft = {"movie_ids": []}
        self.inspected_ids: set[str] = set()
        self.crew_ids: set[str] = set()
        self.visible_ids: set[str] = set()
        self.last_observation = ""
        self.search_count = 0
        self.open_count = 0
        self.compare_count = 0
        self.action_signatures: list[str] = []
        self.no_progress_streak = 0
        self.loop = False
        self.termination = None
        self.last_reward_detail = None
        self.last_projection = None

    def reset(self) -> dict:
        self._reset_state()
        constraints = self.task["constraints"]
        page = {
            "page_type": "start",
            "body": (
                "请使用工具检索影片，核验详情与导演后再写入片单，最后 finalize_watchlist 或 abort。\n"
                f"目标 {constraints['n_movies']} 部，总片长不超过 {constraints['max_total_runtime']} 分钟，"
                f"最低评分 {constraints.get('min_rating')}。"
            ),
        }
        return self._observe(page, reward=0.0, done=False)

    def step(self, name: str, parameters: dict | None = None) -> dict:
        if self.done:
            raise RuntimeError("episode already finished")
        parameters = parameters or {}
        reject = action_reject_reason(
            name,
            parameters,
            self.last_observation,
            visible_ids=self.visible_ids,
        )
        if reject:
            page = {
                "page_type": "guard",
                "body": f"动作被拒绝: {reject}\n请只使用当前页面列出的影片 ID。",
            }
            self.step_count += 1
            self._note_action(name, parameters, progress=False)
            result = self._observe(page, reward=0.0, done=False)
            result["guard_rejected"] = True
            result["guard_reason"] = reject
            if self.step_count >= self.max_steps:
                return self._finish("max_steps")
            return result

        self.step_count += 1
        handler = {
            "search_movies": self._search,
            "open": self._open,
            "view_crew": self._view_crew,
            "compare": self._compare,
            "add_to_watchlist": self._add,
            "view_draft": self._view_draft,
            "finalize_watchlist": self._finalize,
            "abort": self._abort,
        }[name]
        result = handler(parameters)
        if self.done:
            return result
        if self.loop:
            return self._finish("loop")
        if self.step_count >= self.max_steps:
            return self._finish("max_steps")
        return result

    def _search(self, parameters: dict) -> dict:
        parameters = dict(parameters or {})
        if parameters.get("genre"):
            mapped = normalize_genre(parameters.get("genre"))
            if mapped:
                parameters["genre"] = mapped
        query = str(parameters.get("query") or "")
        mapped_query_genre = normalize_genre(query)
        if mapped_query_genre and mapped_query_genre != query:
            query = f"{query} {mapped_query_genre}".strip()
        hits = []
        exact = self._title_index.get(query.casefold()) or []
        for movie_id in exact:
            hits.append((movie_id, 1e8))
        for movie_id, score in self._bm25.search(query, top_k=SEARCH_TOP_K * 4):
            if movie_id not in {item[0] for item in hits}:
                hits.append((movie_id, score))
        movies = []
        for movie_id, _score in hits:
            movie = self.catalog.movies[movie_id]
            if not _movie_matches_filters(movie, parameters):
                continue
            movies.append(movie)
            if len(movies) >= SEARCH_TOP_K:
                break
        self.search_count += 1
        self.visible_ids = {movie["movie_id"] for movie in movies}
        lines = [render_movie_line(index, movie) for index, movie in enumerate(movies, start=1)]
        body = "影片检索结果:\n" + ("\n".join(lines) if lines else "(无结果)")
        self._note_action("search_movies", parameters, progress=bool(movies))
        return self._observe({"page_type": "search", "body": body}, 0.0, False)

    def _open(self, parameters: dict) -> dict:
        movie_id = str(parameters["movie_id"])
        movie = self.catalog.movies[movie_id]
        self.inspected_ids.add(movie_id)
        self.visible_ids = {movie_id}
        self.open_count += 1
        title = movie.get("title_zh") or movie.get("title_en") or movie["original_title"]
        body = (
            f"影片详情\n{render_movie_line(1, movie, detail=True)}\n"
            f"中文片名: {title}\n"
            f"英文/原名: {movie.get('title_en') or movie.get('original_title')}\n"
            f"片长: {movie.get('runtime')} 分钟\n"
            f"评分: {movie.get('rating')}（{movie.get('votes')} 票）\n"
            f"是否有中文译名: {'是' if movie.get('has_zh_aka') else '否'}\n"
            "导演名单请调用 view_crew 核验。"
        )
        self._note_action("open", parameters, progress=True)
        return self._observe({"page_type": "detail", "body": body}, 0.0, False)

    def _view_crew(self, parameters: dict) -> dict:
        movie_id = str(parameters["movie_id"])
        movie = self.catalog.movies[movie_id]
        self.inspected_ids.add(movie_id)
        self.crew_ids.add(movie_id)
        self.visible_ids = {movie_id}
        directors = "、".join(movie.get("directors") or []) or "未知"
        body = (
            f"演职人员\n{render_movie_line(1, movie)}\n"
            f"导演: {directors}"
        )
        self._note_action("view_crew", parameters, progress=True)
        return self._observe({"page_type": "crew", "body": body}, 0.0, False)

    def _compare(self, parameters: dict) -> dict:
        movie_ids = [str(item) for item in parameters["movie_ids"]]
        if any(movie_id not in self.catalog.movies for movie_id in movie_ids):
            page = {"page_type": "compare", "body": "比较失败: 影片不存在。"}
            self._note_action("compare", parameters, progress=False)
            return self._observe(page, 0.0, False)
        self.inspected_ids.update(movie_ids)
        self.visible_ids = set(movie_ids)
        self.compare_count += 1
        lines = [
            render_movie_line(index, self.catalog.movies[movie_id], detail=True)
            for index, movie_id in enumerate(movie_ids, start=1)
        ]
        body = "比较:\n" + "\n".join(lines)
        self._note_action("compare", parameters, progress=True)
        return self._observe({"page_type": "compare", "body": body}, 0.0, False)

    def _add(self, parameters: dict) -> dict:
        movie_id = str(parameters["movie_id"])
        if movie_id not in self.catalog.movies:
            page = {"page_type": "draft", "body": f"未知影片: {movie_id}"}
            self._note_action("add_to_watchlist", parameters, progress=False)
            return self._observe(page, 0.0, False)
        if movie_id not in self.inspected_ids:
            page = {"page_type": "draft", "body": "写入片单前必须先 open 或 compare 该影片。"}
            self._note_action("add_to_watchlist", parameters, progress=False)
            return self._observe(page, 0.0, False)
        n_movies = int(self.task["constraints"]["n_movies"])
        if movie_id not in self.draft["movie_ids"]:
            self.draft["movie_ids"].append(movie_id)
            if len(self.draft["movie_ids"]) > n_movies:
                self.draft["movie_ids"] = self.draft["movie_ids"][-n_movies:]
        self.visible_ids = {movie_id}
        self._note_action("add_to_watchlist", parameters, progress=True)
        page = {"page_type": "draft", "body": f"已将 {movie_id} 加入片单。"}
        return self._observe(page, 0.0, False)

    def _view_draft(self, parameters: dict) -> dict:
        del parameters
        self.visible_ids = set(self.draft["movie_ids"])
        self._note_action("view_draft", {}, progress=False)
        return self._observe({"page_type": "draft", "body": "片单如下。"}, 0.0, False)

    def _finalize(self, parameters: dict) -> dict:
        del parameters
        self._note_action("finalize_watchlist", {}, progress=True)
        return self._finish("finalize_watchlist")

    def _abort(self, parameters: dict) -> dict:
        del parameters
        self._note_action("abort", {"reason": "no_feasible_watchlist"}, progress=True)
        return self._finish("abort")

    def _finish(self, termination: str) -> dict:
        self.done = True
        self.over = True
        self.termination = termination
        detail = score_terminal(
            task=self.task,
            draft=self.draft,
            catalog=self.catalog,
            inspected_ids=self.inspected_ids,
            crew_ids=self.crew_ids,
            termination=termination,
            search_count=self.search_count,
            open_count=self.open_count,
            loop=self.loop or termination == "loop",
        )
        self.last_reward_detail = detail
        page = {
            "page_type": "terminal",
            "body": f"环境已结束: {detail['reward_type']}",
        }
        result = self._observe(page, detail["reward"], True)
        result["reward_detail"] = detail
        result["terminal_result"] = {
            "done": True,
            "over": True,
            "reward": detail["reward"],
            "reward_detail": detail,
            "watchlist": deepcopy(self.draft),
            "goal": {"gold_watchlist": self.task.get("gold_watchlist")},
        }
        return result

    def _observe(self, page: dict, reward: float, done: bool) -> dict:
        visible, projection = render_observation(page, self._env_state(), self.catalog)
        self.last_observation = visible
        self.last_projection = projection
        if page.get("page_type") in {"search", "detail", "crew", "compare"}:
            self.visible_ids |= set(listed_entity_ids(visible))
        return {
            "observation": visible,
            "raw_observation": visible,
            "projection": projection,
            "reward": reward,
            "done": done,
            "over": self.over,
            "environment_version": ENVIRONMENT_VERSION,
            "reward_version": REWARD_VERSION,
        }

    def _env_state(self) -> dict:
        return {
            "task": self.task,
            "draft": self.draft,
            "step": self.step_count,
            "max_steps": self.max_steps,
        }

    def _note_action(self, name: str, parameters: dict, *, progress: bool) -> None:
        signature = f"{name}:{sorted((parameters or {}).items())}"
        if self.action_signatures[-2:] == [signature, signature]:
            self.loop = True
        self.action_signatures.append(signature)
        if progress:
            self.no_progress_streak = 0
        else:
            self.no_progress_streak += 1
            if self.no_progress_streak >= 4:
                self.loop = True

    def public_instruction(self) -> str:
        return self.task["query"]


def _movie_matches_filters(movie: dict, parameters: dict) -> bool:
    genre = normalize_genre(parameters.get("genre"))
    if genre and genre not in (movie.get("genres") or []):
        return False
    year = int(movie.get("year") or 0)
    year_min = parameters.get("year_min")
    if year_min is not None and year < int(year_min):
        return False
    year_max = parameters.get("year_max")
    if year_max is not None and year > int(year_max):
        return False
    min_rating = parameters.get("min_rating")
    if min_rating is not None and float(movie.get("rating") or 0.0) < float(min_rating):
        return False
    return True
