"""Embedding matching stage.

WHAT GOES IN  -> a query embedding + an enrollment database of embeddings
WHAT COMES OUT -> a MatchResult { identity, similarity, threshold, is_known }

HOW COMPARISON WORKS
--------------------
Every embedding was L2-normalized before storage and before matching, so
cosine similarity equals the plain dot product (see utils.cosine_similarity).
We score the query against EVERY stored embedding, take the best score PER
person (a person is enrolled with several images -> several embeddings), and
then:
    best_score >= threshold  ->  identity = that person  (Known)
    best_score <  threshold  ->  identity = None         (Unknown)

ABOUT THE THRESHOLD
-------------------
The threshold is the acceptance/rejection boundary for cosine similarity.
There is NO universal value — it depends on the model, alignment quality and
your data. It is kept configurable (config.py, CLI --threshold / env
FR_THRESHOLD) and must be validated on YOUR enrolled identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .utils import cosine_similarity


@dataclass
class MatchResult:
    """Result of matching one query embedding against the database."""

    identity: Optional[str]        # person name, or None when unknown
    similarity: float              # best cosine score found
    threshold: float               # the boundary that was applied
    is_known: bool                 # identity is not None and score >= threshold
    scores: Dict[str, float] = field(default_factory=dict)  # best score per person

    @property
    def display_label(self) -> str:
        """Short label used on video frames: `Jordan 0.83` / `Unknown 0.34`."""
        name = self.identity if self.identity is not None else "Unknown"
        return f"{name} {self.similarity:.2f}"

    def __repr__(self) -> str:
        return (
            f"Known: {self.identity}\n"
            f"Similarity: {self.similarity:.4f}"
            if self.is_known
            else f"Unknown\nSimilarity: {self.similarity:.4f}"
        )


class EmbeddingMatcher:
    """Compares a query embedding against a set of enrolled identities.

    `set_enrollment()` injects the database so the matcher is easy to unit-test
    without touching disk or models.
    """

    def __init__(self, threshold: float = 0.4) -> None:
        self.threshold = threshold
        # identity name -> (N, D) array of that person's embeddings
        self._enrolled: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Database management
    # ------------------------------------------------------------------

    @property
    def identities(self) -> List[str]:
        return sorted(self._enrolled.keys())

    def set_enrollment(self, enrollment: Dict[str, np.ndarray]) -> None:
        """Load embeddings keyed by identity name (each an (N, D) array)."""
        self._enrolled = {}
        for name, array in enrollment.items():
            arr = np.asarray(array, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)  # a single embedding for this person
            if arr.ndim != 2:
                raise ValueError(
                    f"Enrollment for '{name}' must be an (N, D) array, "
                    f"got shape {arr.shape}"
                )
            self._enrolled[name] = arr

    def has_enrollments(self) -> bool:
        return len(self._enrolled) > 0

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """Return the best Know/Unknown decision for a query embedding."""
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)

        per_person_best: Dict[str, float] = {}
        for name, stored in self._enrolled.items():
            # score against every embedding of this person, keep the best:
            # "does ANY photo of Jordan look like this?"
            best = max(cosine_similarity(query, row) for row in stored)
            per_person_best[name] = best

        if not per_person_best:
            raise ValueError("matcher.match() called with an empty enrollment database")

        # best overall person
        best_name = max(per_person_best, key=per_person_best.get)
        best_score = per_person_best[best_name]

        is_known = best_score >= self.threshold
        return MatchResult(
            identity=best_name if is_known else None,
            similarity=best_score,
            threshold=self.threshold,
            is_known=is_known,
            scores=per_person_best,
        )