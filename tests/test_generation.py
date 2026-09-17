from tinywatch_grpo.generation.task_gen import assert_disjoint, generate_splits
from tinywatch_grpo.generation.vocab import mini_catalog, normalize_genre


def test_mini_catalog_has_real_imdb_ids():
    catalog = mini_catalog()
    assert catalog["schema_version"] == "tinywatch-catalog-v1"
    assert len(catalog["movies"]) >= 40
    assert all(movie["movie_id"].startswith("tt") for movie in catalog["movies"])
    assert any(movie.get("has_zh_aka") for movie in catalog["movies"])


def test_splits_are_disjoint():
    catalog = mini_catalog()
    splits = generate_splits(catalog, seed=42, sft_pool=20, grpo_train=10, eval_holdout=5)
    assert_disjoint(splits)
    assert len(splits["evaluation"]) == 5
    assert len(splits["grpo_train"]) == 10
    assert len(splits["sft_pool"]) == 20


def test_easy_solvable_tasks_include_director_and_year():
    catalog = mini_catalog()
    splits = generate_splits(catalog, seed=42, sft_pool=20, grpo_train=10, eval_holdout=5)
    easy = [
        task
        for bucket in splits.values()
        for task in bucket
        if task.get("difficulty") == "easy" and task.get("solvable")
    ]
    assert easy
    for task in easy:
        assert task["constraints"].get("directors")
        assert task["constraints"].get("year_min") is not None
        assert task["constraints"].get("year_max") is not None


def test_normalize_genre_aliases():
    assert normalize_genre("War") == "战争"
    assert normalize_genre("comedy") == "喜剧"
    assert normalize_genre("Sci-Fi") == "科幻"
    assert normalize_genre("剧情") == "剧情"
