from tinywatch_grpo.generation.task_gen import assert_disjoint, generate_splits
from tinywatch_grpo.generation.vocab import mini_catalog


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
