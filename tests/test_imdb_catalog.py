from tinywatch_grpo.generation.imdb_catalog import IMDB_BASE, IMDB_FILES, pick_zh_title, zh_title_priority


def test_imdb_source_is_official():
    assert IMDB_BASE.startswith("https://datasets.imdbws.com/")
    assert "title.basics.tsv.gz" in IMDB_FILES
    assert "title.ratings.tsv.gz" in IMDB_FILES
    assert "title.akas.tsv.gz" in IMDB_FILES


def test_cn_title_outranks_taiwan_aka():
    assert zh_title_priority("CN", "", "肖申克的救赎") < zh_title_priority("TW", "", "刺激1995")
    chosen = pick_zh_title(
        [
            (zh_title_priority("TW", "", "刺激1995"), "刺激1995"),
            (zh_title_priority("CN", "", "肖申克的救赎"), "肖申克的救赎"),
        ]
    )
    assert chosen == "肖申克的救赎"
