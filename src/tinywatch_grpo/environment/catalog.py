"""Catalog loader and movie lookup for TinyWatch Environment v1."""

from __future__ import annotations

import json
from pathlib import Path


CATALOG_SCHEMA_VERSION = "tinywatch-catalog-v1"


def load_catalog(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported catalog schema: {payload.get('schema_version')}"
        )
    return payload


class CatalogIndex:
    """O(1) lookups used by BM25 search and Reward v1."""

    def __init__(self, catalog: dict):
        self.catalog = catalog
        self.movies = {movie["movie_id"]: movie for movie in catalog["movies"]}
        self.by_genre: dict[str, list[dict]] = {}
        self.by_director: dict[str, list[dict]] = {}
        for movie in catalog["movies"]:
            for genre in movie.get("genres") or []:
                self.by_genre.setdefault(genre, []).append(movie)
            for director in movie.get("directors") or []:
                self.by_director.setdefault(director, []).append(movie)

    def get(self, movie_id: str) -> dict | None:
        movie = self.movies.get(movie_id)
        if movie is None:
            return None
        return {"kind": "movie", **movie}

    def entity_kind(self, movie_id: str) -> str | None:
        if movie_id in self.movies:
            return "movie"
        return None
