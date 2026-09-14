"""Tests for the ArcFace embedder.

These tests need the models/ folder populated:
    python -m scripts.download_models

When the model is missing the embedding tests skip with a clear message,
so `pytest` stays green on a fresh clone without a network/model download.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import EMBEDDER_MODEL_PATH
from app.embedder import ArcFaceEmbedder

pytestmark_server = pytest.mark.skipif(
    not os.path.exists(EMBEDDER_MODEL_PATH),
    reason="ArcFace model not downloaded. Run: python -m scripts.download_models",
)


def make_aligned_face():
    """A ridiculously fake but shape-perfect BGR 112x112 'aligned face'."""
    rng = np.random.default_rng(7)
    img = (rng.standard_normal((112, 112, 3)) * 60 + 127).clip(0, 255)
    return img.astype(np.uint8)


@pytestmark_server
class TestEmbedderPreprocessing:
    def test_tensor_shape_after_preprocess(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        tensor = embedder.preprocess(make_aligned_face())
        # (batch, channels, height, width)
        assert tensor.shape == (1, 3, embedder.image_size[0], embedder.image_size[1])
        assert tensor.dtype == np.float32
        # after normalization values must stay finite and reasonable
        assert np.all(np.isfinite(tensor))

    def test_preprocess_is_deterministic(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        face = make_aligned_face()
        assert np.array_equal(embedder.preprocess(face), embedder.preprocess(face))


@pytestmark_server
class TestEmbeddingInference:
    def test_embedding_shape_and_dimension(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        embedding = embedder.get_embedding(make_aligned_face())
        assert embedding.shape == (embedder.expected_dimension,)
        assert embedding.shape[0] == 512

    def test_embedding_is_l2_normalized(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        embedding = embedder.get_embedding(make_aligned_face())
        assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-4)

    def test_same_input_same_embedding(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        face = make_aligned_face()
        e1 = embedder.get_embedding(face)
        e2 = embedder.get_embedding(face)
        assert np.allclose(e1, e2, atol=1e-6)

    def test_different_inputs_different_embeddings(self):
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        e1 = embedder.get_embedding(make_aligned_face())
        e2 = embedder.get_embedding((make_aligned_face() + 30).clip(0, 255))
        assert np.linalg.norm(e1 - e2) > 0.1