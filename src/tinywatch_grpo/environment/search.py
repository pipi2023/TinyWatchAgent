"""Lightweight BM25 over character bigrams. No extra retrieval dependency."""

from __future__ import annotations

import math
import re
from collections import Counter


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Split ASCII words and emit Chinese unigrams plus adjacent bigrams."""
    parts = _TOKEN_RE.findall(text.casefold())
    tokens: list[str] = []
    han: list[str] = []
    for part in parts:
        if re.fullmatch(r"[A-Za-z0-9]+", part):
            if han:
                tokens.extend(_han_tokens(han))
                han = []
            tokens.append(part)
        else:
            han.append(part)
    if han:
        tokens.extend(_han_tokens(han))
    return tokens


def _han_tokens(chars: list[str]) -> list[str]:
    tokens = list(chars)
    tokens.extend(a + b for a, b in zip(chars, chars[1:]))
    return tokens


class BM25Index:
    def __init__(self, documents: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.doc_tokens = [tokenize(text) for _, text in documents]
        self.doc_len = [len(tokens) or 1 for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        df: Counter[str] = Counter()
        self.tf: list[Counter[str]] = []
        for tokens in self.doc_tokens:
            counts = Counter(tokens)
            self.tf.append(counts)
            df.update(counts.keys())
        n = max(len(self.doc_ids), 1)
        self.idf = {
            term: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        query = (query or "").strip()
        if not query or not self.doc_ids:
            return []
        if query in self.doc_ids:
            return [(query, 1e9)]
        q_tokens = tokenize(query)
        scores = []
        for index, doc_id in enumerate(self.doc_ids):
            score = 0.0
            length = self.doc_len[index]
            tf = self.tf[index]
            for term in q_tokens:
                if term not in tf:
                    continue
                idf = self.idf.get(term, 0.0)
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
                score += idf * (freq * (self.k1 + 1)) / denom
            scores.append((doc_id, score))
        scores.sort(key=lambda item: (-item[1], item[0]))
        return [(doc_id, score) for doc_id, score in scores[:top_k] if score > 0]


def movie_document(movie: dict) -> str:
    genres = " ".join(movie.get("genres") or [])
    directors = " ".join(movie.get("directors") or [])
    aliases = " ".join(movie.get("akas") or [])
    return (
        f"{movie['movie_id']} {movie.get('title_zh') or ''} {movie.get('title_en') or ''} "
        f"{movie.get('original_title') or ''} {aliases} {genres} {directors} "
        f"{movie.get('year', '')}"
    )
