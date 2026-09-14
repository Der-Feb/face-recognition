"""Unit tests: 5-point face alignment.

Check that the aligner:
  * produces a (112, 112, 3) standardized crop;
  * places the detected landmarks close to the ArcFace reference landmarks;
  * works for a translated+rotated, scaled face (the whole point of alignment).

No neural networks required. Run with: pytest tests/test_alignment.py
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.aligner import FaceAligner


def make_test_image(size=256):
    """A synthetic image with a tiny 'face' so warping has content to sample."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal((size, size, 3)) * 30 + 128).astype(np.uint8)


def apply_similarity(landmarks, angle_deg, scale, tx, ty):
    """Rotate+scale+translate landmark points (ground truth for the test)."""
    theta = np.deg2rad(angle_deg)
    rot = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    points = scale * (landmarks @ rot.T)
    points[:, 0] += tx
    points[:, 1] += ty
    return points.astype(np.float32)


class TestAlignment:
    def test_output_shape(self):
        aligner = FaceAligner(output_size=112)
        image = make_test_image()
        aligned, matrix = aligner.align(image, FaceAligner.REFERENCE_LANDMARKS)
        assert aligned.shape == (112, 112, 3)
        assert matrix.shape == (2, 3)

    def test_identity_transform_when_landmarks_already_canonical(self):
        """Aligning a face whose landmarks ARE the reference is near-identity."""
        aligner = FaceAligner(output_size=112)
        aligned, matrix = aligner.align(make_test_image(), aligner.REFERENCE_LANDMARKS)
        expected = np.eye(2, 3, dtype=np.float64)
        assert np.allclose(matrix, expected, atol=1e-2)

    def test_transformed_landmarks_land_on_reference(self):
        """Detected eyes/nose/mouth must end up ON the reference template."""
        aligner = FaceAligner(output_size=112)

        # Simulate a face that is rotated, scaled and translated relative to
        # the canonical ArcFace template.
        detected = apply_similarity(
            aligner.REFERENCE_LANDMARKS, angle_deg=25, scale=1.4, tx=-40, ty=95
        )
        image = make_test_image()
        _, matrix = aligner.align(image, detected)

        def warp(point):
            p = np.array([point[0], point[1], 1.0])
            return matrix @ p

        for src, ref in zip(detected, aligner.REFERENCE_LANDMARKS):
            mapped = warp(src)
            assert np.linalg.norm(mapped - ref) < 2.0, (
                f"landmark mapped to {mapped}, expected near {ref}"
            )

    def test_rejects_wrong_landmark_shape(self):
        aligner = FaceAligner()
        # only 3 landmarks instead of 5
        with pytest.raises(ValueError):
            aligner.align(make_test_image(), np.array([[1, 1], [2, 2], [3, 3]]))

    def test_rejects_degenerate_landmarks(self):
        aligner = FaceAligner()
        degenerate = np.tile(np.array([50.0, 50.0]), (5, 1)).astype(np.float32)
        with pytest.raises(ValueError):
            aligner.align(make_test_image(), degenerate)