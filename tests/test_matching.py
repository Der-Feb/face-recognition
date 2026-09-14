"""Unit tests: L2 normalization, cosine similarity and threshold matching.

These run WITHOUT any model or webcam — pure numpy checks of the math layer.
Run with:  pytest tests/test_matching.py  (or just `pytest` from the root)
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.matcher import EmbeddingMatcher
from app.utils import cosine_similarity, l2_normalize


class TestL2Normalization:
    def test_unit_norm(self):
        vector = np.array([3.0, 4.0])
        normalized = l2_normalize(vector)
        assert np.isclose(np.linalg.norm(normalized), 1.0)

    def test_direction_unchanged(self):
        vector = np.array([-2.0, 5.0, 1.0])
        normalized = l2_normalize(vector)
        # normalized must be a positive scalar multiple of the original
        ratio = normalized[0] / vector[0]
        assert np.allclose(normalized, ratio * vector)

    def test_zero_vector_guarded(self):
        zero = np.zeros(5)
        normalized = l2_normalize(zero)
        assert np.all(np.isfinite(normalized))
        assert np.allclose(normalized, 0.0)

    def test_random_vectors_unit_norm(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = rng.standard_normal(512)
            assert np.isclose(np.linalg.norm(l2_normalize(v)), 1.0)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([0.5, 0.2, -0.7])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_normalized_dot_product(self):
        """L2-normalized vectors: cosine == dot product (the taught identity)."""
        rng = np.random.default_rng(1)
        a = l2_normalize(rng.standard_normal(64))
        b = l2_normalize(rng.standard_normal(64))
        assert cosine_similarity(a, b) == pytest.approx(float(np.dot(a, b)), abs=1e-6)

    def test_similar_more_similar_than_dissimilar(self):
        rng = np.random.default_rng(2)
        query = rng.standard_normal(64)
        near = query + rng.standard_normal(64) * 0.05      # small perturbation
        far = rng.standard_normal(64)                      # unrelated vector
        assert cosine_similarity(query, near) > cosine_similarity(query, far)


class TestMatcherThreshold:
    def make_db(self, names=("alice", "bob")):
        rng = np.random.default_rng(3)
        return {
            name: l2_normalize(rng.standard_normal((3, 128)))  # 3 embeddings each
            for name in names
        }

    def test_known_identity_above_threshold(self):
        matcher = EmbeddingMatcher(threshold=0.5)
        db = self.make_db()
        matcher.set_enrollment(db)
        # query identical to alice's first embedding -> similarity ~1.0
        q = db["alice"][0]
        result = matcher.match(q)
        assert result.is_known
        assert result.identity == "alice"
        assert result.similarity > 0.9

    def test_threshold_boundary_inclusive(self):
        rng = np.random.default_rng(0)
        base = l2_normalize(rng.standard_normal(16))
        query = l2_normalize(rng.standard_normal(16))  # ~orthogonal to base
        matcher = EmbeddingMatcher(threshold=0.8)
        matcher.set_enrollment({"x": base.reshape(1, -1)})
        result = matcher.match(query)
        assert result.is_known is False
        # When the query is EXACTLY the enrolled vector, similarity == 1.0,
        # which is >= threshold -> Known (boundary is inclusive).
        matcher.set_enrollment({"x": query.reshape(1, -1)})
        result = matcher.match(query)
        assert result.is_known
        assert result.similarity == pytest.approx(1.0, abs=1e-6)

    def test_unknown_below_threshold(self):
        matcher = EmbeddingMatcher(threshold=0.9)  # very strict
        matcher.set_enrollment(self.make_db())
        rng = np.random.default_rng(4)
        q = l2_normalize(rng.standard_normal(128))
        # random vector almost surely far below 0.9 to either identity
        result = matcher.match(q)
        assert result.identity is None
        assert result.is_known is False
        assert result.similarity < 0.9

    def test_empty_database_raises(self):
        matcher = EmbeddingMatcher()
        with pytest.raises(ValueError):
            matcher.match(np.zeros(128))

    def test_threshold_is_configurable(self):
        db = self.make_db()
        strict = EmbeddingMatcher(threshold=0.9999)
        loose = EmbeddingMatcher(threshold=0.0)
        strict.set_enrollment(db)
        loose.set_enrollment(db)
        rng = np.random.default_rng(5)
        q = l2_normalize(rng.standard_normal(128))
        assert not strict.match(q).is_known
        assert loose.match(q).is_known