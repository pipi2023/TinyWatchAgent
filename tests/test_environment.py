from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.generation.task_gen import generate_task
from tinywatch_grpo.generation.vocab import mini_catalog


def _task():
    catalog = mini_catalog()
    state = [3]

    def nxt(n):
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] % n

    task = generate_task(9, catalog, difficulty="easy", nxt=nxt, solvable=True)
    return catalog, task


def test_reset_hides_gold():
    catalog, task = _task()
    env = TinyWatchEnv(catalog, task)
    result = env.reset()
    gold_id = task["gold_watchlist"]["movie_ids"][0]
    assert "gold_watchlist" not in result["observation"]
    assert gold_id not in result["observation"]
    assert "tinywatch-observation-v1" in result["observation"]


def test_guard_rejects_unknown_id():
    catalog, task = _task()
    env = TinyWatchEnv(catalog, task)
    env.reset()
    result = env.step("open", {"movie_id": "not_a_real_id"})
    assert result["guard_rejected"] is True
    assert result["done"] is False


def test_search_and_open_roundtrip():
    catalog, task = _task()
    env = TinyWatchEnv(catalog, task)
    env.reset()
    gold_id = task["gold_watchlist"]["movie_ids"][0]
    title = env.catalog.movies[gold_id]["title_zh"]
    result = env.step("search_movies", {"query": title})
    assert result.get("guard_rejected") is not True
    assert "影片检索结果" in result["observation"]
    assert gold_id in result["observation"]
