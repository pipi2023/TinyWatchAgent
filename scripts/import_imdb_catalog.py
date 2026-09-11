#!/usr/bin/env python3
"""Download IMDb non-commercial dumps (if needed) and freeze catalog.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.generation.imdb_catalog import build_imdb_catalog, download_imdb, write_catalog
from tinywatch_grpo.generation.vocab import mini_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("outputs/imdb-raw"))
    parser.add_argument("--output", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--max-movies", type=int, default=5000)
    parser.add_argument("--min-votes", type=int, default=8000)
    parser.add_argument("--mini", action="store_true", help="write the in-repo fixture instead of IMDb")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    if args.mini:
        catalog = mini_catalog()
    else:
        if not args.skip_download:
            download_imdb(args.raw_dir)
        catalog = build_imdb_catalog(args.raw_dir, max_movies=args.max_movies, min_votes=args.min_votes)
    info = write_catalog(args.output, catalog)
    meta = {
        **info,
        "source": catalog.get("source"),
        "movies": len(catalog["movies"]),
        "zh_aka": sum(1 for movie in catalog["movies"] if movie.get("has_zh_aka")),
    }
    Path("data/catalog/metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
