from tinywatch_grpo.environment.catalog import CatalogIndex
from tinywatch_grpo.environment.reward import evidence_valid, gold_match, score_terminal
from tinywatch_grpo.generation.task_gen import generate_task
from tinywatch_grpo.generation.vocab import mini_catalog


def _sample():
    catalog = mini_catalog()
    state = [11]

    def nxt(n):
        state[0] = (state[0] * 1664525 + 1013904223) & 0xFFFFFFFF
        return state[0] % n

    task = generate_task(4, catalog, difficulty="easy", nxt=nxt, solvable=True)
    return catalog, task, CatalogIndex(catalog)


def test_gold_requires_evidence():
    catalog, task, index = _sample()
    gold = task["gold_watchlist"]
    detail = score_terminal(
        task=task,
        draft=gold,
        catalog=index,
        inspected_ids=set(),
        crew_ids=set(),
        termination="finalize_watchlist",
        search_count=3,
        open_count=0,
    )
    assert detail["reward_valid"] is False
    assert detail["reward_type"] == "reward_unverifiable"


def test_gold_watchlist_score():
    catalog, task, index = _sample()
    gold = task["gold_watchlist"]
    inspected = set(gold["movie_ids"])
    crew = inspected if task["constraints"].get("directors") else set()
    detail = score_terminal(
        task=task,
        draft=gold,
        catalog=index,
        inspected_ids=inspected,
        crew_ids=crew,
        termination="finalize_watchlist",
        search_count=4,
        open_count=4,
    )
    assert gold_match(task, gold)
    assert evidence_valid(task, gold, inspected, crew)
    assert detail["reward_type"] == "gold_watchlist"
    assert detail["reward"] == 1.0
    assert detail["watchlist_success"] is True


def test_gold_match_is_unordered():
    task = {"gold_watchlist": {"movie_ids": ["tt1", "tt2"]}}
    assert gold_match(task, {"movie_ids": ["tt2", "tt1"]})
    assert not gold_match(task, {"movie_ids": ["tt1"]})
