"""Build a frozen TinyWatch catalog from IMDb non-commercial TSV dumps.

Source: https://developer.imdb.com/non-commercial-datasets/
License: IMDb non-commercial use. Derived catalog is for this research agent only.
"""

from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path
from urllib.request import urlretrieve

from tinywatch_grpo.environment.catalog import CATALOG_SCHEMA_VERSION
from tinywatch_grpo.generation.vocab import GENRE_EN_TO_ZH, mini_catalog


IMDB_FILES = (
    "title.basics.tsv.gz",
    "title.ratings.tsv.gz",
    "title.akas.tsv.gz",
    "title.crew.tsv.gz",
    "name.basics.tsv.gz",
)
IMDB_BASE = "https://datasets.imdbws.com/"
ZH_REGIONS = {"CN", "TW", "HK", "MO"}
ZH_LANGS = {"zh", "cmn", "yue", "zh-cn", "zh-tw"}
ZH_REGION_PRIORITY = {"CN": 0, "HK": 1, "TW": 2, "MO": 3}


def zh_title_priority(region: str, language: str, title: str) -> int | None:
    region = (region or "").upper()
    language = (language or "").casefold()
    if region in ZH_REGION_PRIORITY:
        base = ZH_REGION_PRIORITY[region]
    elif language in ZH_LANGS:
        base = 4
    else:
        return None
    has_han = any("\u4e00" <= char <= "\u9fff" for char in title)
    return base * 10 + (0 if has_han else 5)


def pick_zh_title(candidates: list[tuple[int, str]]) -> str | None:
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (item[0], -len(item[1])))[0][1]


def download_imdb(raw_dir: str | Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in IMDB_FILES:
        dest = raw_dir / name
        if not dest.exists() or dest.stat().st_size < 1024:
            urlretrieve(IMDB_BASE + name, dest)
        paths[name] = dest
    return paths


def _open_tsv(path: Path):
    csv.field_size_limit(sys.maxsize)
    handle = gzip.open(path, mode="rt", encoding="utf-8", newline="")
    return handle, csv.DictReader(handle, delimiter="\t")


def _int(value, default=0):
    if value in (None, "", "\\N"):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value, default=0.0):
    if value in (None, "", "\\N"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_imdb_catalog(
    raw_dir: str | Path,
    *,
    max_movies: int = 5000,
    min_votes: int = 8000,
    year_min: int = 1970,
    year_max: int = 2024,
    runtime_min: int = 70,
    runtime_max: int = 210,
) -> dict:
    raw_dir = Path(raw_dir)
    paths = {name: raw_dir / name for name in IMDB_FILES}
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"IMDb dumps missing: {missing}")

    ratings = {}
    handle, reader = _open_tsv(paths["title.ratings.tsv.gz"])
    with handle:
        for row in reader:
            votes = _int(row.get("numVotes"))
            if votes < min_votes:
                continue
            ratings[row["tconst"]] = {"rating": round(_float(row.get("averageRating")), 1), "votes": votes}

    basics = {}
    handle, reader = _open_tsv(paths["title.basics.tsv.gz"])
    with handle:
        for row in reader:
            tconst = row["tconst"]
            if tconst not in ratings:
                continue
            if row.get("titleType") != "movie" or row.get("isAdult") == "1":
                continue
            year = _int(row.get("startYear"))
            runtime = _int(row.get("runtimeMinutes"))
            if not (year_min <= year <= year_max):
                continue
            if not (runtime_min <= runtime <= runtime_max):
                continue
            genres_en = [part for part in (row.get("genres") or "").split(",") if part and part != "\\N"]
            genres = [GENRE_EN_TO_ZH.get(part, part) for part in genres_en]
            if not genres:
                continue
            basics[tconst] = {
                "movie_id": tconst,
                "title_en": row.get("primaryTitle") or row.get("originalTitle"),
                "original_title": row.get("originalTitle") or row.get("primaryTitle"),
                "year": year,
                "runtime": runtime,
                "rating": ratings[tconst]["rating"],
                "votes": ratings[tconst]["votes"],
                "genres": genres,
                "genres_en": genres_en,
                "directors": [],
                "title_zh": None,
                "akas": [],
                "has_zh_aka": False,
                "_zh_candidates": [],
            }

    akas_handle, akas_reader = _open_tsv(paths["title.akas.tsv.gz"])
    with akas_handle:
        for row in akas_reader:
            tconst = row.get("titleId")
            movie = basics.get(tconst)
            if movie is None:
                continue
            title = row.get("title") or ""
            region = (row.get("region") or "").upper()
            language = (row.get("language") or "").casefold()
            priority = zh_title_priority(region, language, title)
            if priority is None:
                continue
            movie["has_zh_aka"] = True
            if title and title not in movie["akas"]:
                movie["akas"].append(title)
            if title:
                movie["_zh_candidates"].append((priority, title))

    ranked = sorted(
        basics.values(),
        key=lambda movie: (int(movie["has_zh_aka"]), movie["votes"]),
        reverse=True,
    )
    selected = ranked[: max(1, int(max_movies))]
    wanted = {movie["movie_id"]: movie for movie in selected}

    director_ids: dict[str, list[str]] = {}
    nconsts = set()
    crew_handle, crew_reader = _open_tsv(paths["title.crew.tsv.gz"])
    with crew_handle:
        for row in crew_reader:
            tconst = row["tconst"]
            if tconst not in wanted:
                continue
            names = [part for part in (row.get("directors") or "").split(",") if part and part != "\\N"]
            director_ids[tconst] = names[:3]
            nconsts.update(names[:3])

    names = {}
    name_handle, name_reader = _open_tsv(paths["name.basics.tsv.gz"])
    with name_handle:
        for row in name_reader:
            nconst = row["nconst"]
            if nconst in nconsts:
                names[nconst] = row.get("primaryName") or nconst

    movies = []
    for movie in selected:
        movie["directors"] = [names.get(nconst, nconst) for nconst in director_ids.get(movie["movie_id"], [])]
        movie["title_zh"] = pick_zh_title(movie.pop("_zh_candidates", [])) or movie["title_en"]
        if movie["title_en"] and movie["title_en"] not in movie["akas"]:
            movie["akas"].append(movie["title_en"])
        movies.append(movie)

    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source": "imdb-non-commercial-tsv",
        "attribution": "IMDb non-commercial datasets (https://developer.imdb.com/non-commercial-datasets/). Derived subset for research.",
        "filters": {
            "max_movies": max_movies,
            "min_votes": min_votes,
            "year_min": year_min,
            "year_max": year_max,
            "runtime_min": runtime_min,
            "runtime_max": runtime_max,
        },
        "movies": movies,
    }


def write_catalog(path: str | Path, catalog: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "movies": len(catalog.get("movies") or [])}


def load_or_mini(path: str | Path | None = None) -> dict:
    if path and Path(path).exists():
        from tinywatch_grpo.environment.catalog import load_catalog

        return load_catalog(path)
    return mini_catalog()
