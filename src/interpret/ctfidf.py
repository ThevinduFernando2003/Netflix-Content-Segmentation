"""Class-based TF-IDF.

Each cluster is one pseudo-document. Term scores follow the BERTopic c-TF-IDF
recipe: normalised term frequency within the cluster times ``log(1 + A / f_t)``,
where ``A`` is the average number of tokens per cluster and ``f_t`` is the
token's total count across clusters.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def c_tf_idf_matrix(documents: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(scores, feature_names)`` with one row per document/cluster."""
    if not documents:
        return np.zeros((0, 0), dtype=float), np.array([])
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        counts = vectorizer.fit_transform(documents)
    except ValueError:
        return np.zeros((len(documents), 0), dtype=float), np.array([])
    names = vectorizer.get_feature_names_out()
    return c_tf_idf_from_counts(counts.toarray().astype(float)), names


def c_tf_idf_from_counts(counts: np.ndarray) -> np.ndarray:
    """Score a class-by-term count matrix."""
    if counts.size == 0:
        return counts.astype(float)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    term_frequency = counts / row_sums
    term_totals = counts.sum(axis=0)
    average_tokens = float(counts.sum() / counts.shape[0])
    inverse = np.log((average_tokens / np.clip(term_totals, 1e-12, None)) + 1.0)
    return term_frequency * inverse


def top_terms_per_document(
    documents: list[str],
    top_n: int = 15,
) -> list[list[tuple[str, float]]]:
    scores, names = c_tf_idf_matrix(documents)
    ranked: list[list[tuple[str, float]]] = []
    if scores.size == 0:
        return [[] for _ in documents]
    for row in scores:
        order = np.argsort(row)[::-1][:top_n]
        ranked.append([(str(names[index]), float(row[index])) for index in order if row[index] > 0])
    return ranked
