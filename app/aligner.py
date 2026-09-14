"""5-point face alignment stage.

WHAT GOES IN  -> the original BGR image + the 5 detected landmark points
WHAT COMES OUT -> a standardized, aligned face crop (default 112x112)

WHY ALIGNMENT IS NECESSARY
--------------------------
The ArcFace embedding network has never seen a "raw" face. It was trained on
images that were all pre-aligned so the eyes sit at the same place, the nose
is in the middle, and the mouth is at the bottom. If we fed it raw crops, the
same person photographed from slightly different angles or distances would
produce very different embeddings, and recognition would be fragile.

WHY THE FIVE LANDMARKS MATTER
-----------------------------
* the two EYES define the horizontal axis -> rotation is corrected
* the NOSE marks the vertical centre       -> translation is corrected
* the two MOUTH CORNERS fix the vertical scale -> z-distance / zoom corrected
Two points give rotation+translation; a third breaks scale ambiguity; five
points make the fit robust to any single point being a little noisy.

THE GEOMETRIC TRANSFORMATION
----------------------------
We fit a *similarity transform* (rotation, uniform scale, translation — four
degrees of freedom) that maps the detected landmarks onto a fixed reference:
    detected eyes/nose/mouth  ->  REFERENCE_LANDMARKS on a 112x112 canvas.
cv2.estimateAffinePartial2D computes this 2x3 matrix M in a least-squares
sense; cv2.warpAffine then resamples the image through M.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


class FaceAligner:
    """Aligner that maps detected 5-point landmarks onto a canonical layout.

    The reference below is the *standard ArcFace reference template* used by
    the InsightFace project for a 112x112 output image.
    """

    # Coordinates on the 112x112 output canvas, in order:
    # left-eye, right-eye, nose, left-mouth-corner, right-mouth-corner.
    REFERENCE_LANDMARKS: np.ndarray = np.array(
        [
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose tip
            [41.5493, 92.3655],  # left mouth corner
            [70.7299, 92.2041],  # right mouth corner
        ],
        dtype=np.float32,
    )

    def __init__(self, output_size: int = 112) -> None:
        self.output_size = output_size

    def align(
        self, image: np.ndarray, landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align one face and return (aligned_face, similarity_matrix).

        image      : BGR image containing the face (the FULL frame, not a crop)
        landmarks  : (5, 2) numpy array of the detected landmark points
        Returns
          aligned_face : (output_size, output_size, 3) standardized face crop
          matrix       : the 2x3 similarity transform (useful for debugging,
                         e.g. to draw the canonical points back onto the image)
        """
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.ndim != 2 or landmarks.shape[0] != 5 or landmarks.shape[1] != 2:
            raise ValueError(
                "Expected landmarks of shape (5, 2) (left-eye, right-eye, "
                f"nose, left-mouth, right-mouth); got {landmarks.shape}"
            )

        # Similarity transform mapping *detected* points -> *reference* points.
        # estimateAffinePartial2D restricts the estimated transform to
        # rotation + uniform scale + translation (not a full perspective/affine
        # warp), which is exactly what face alignment is meant to be.
        matrix, inliers = cv2.estimateAffinePartial2D(
            landmarks.reshape(1, -1, 2) if landmarks.ndim == 2 else landmarks,
            self.REFERENCE_LANDMARKS.reshape(1, -1, 2),
        )
        if matrix is None:
            raise ValueError(
                "Similarity-transform estimation failed (degenerate landmarks?)"
            )

        aligned = cv2.warpAffine(
            image,
            matrix,
            (self.output_size, self.output_size),
            flags=cv2.INTER_LINEAR,
            borderValue=(0, 0, 0),
        )
        return aligned, matrix

    @property
    def reference_landmarks(self) -> np.ndarray:
        return self.REFERENCE_LANDMARKS.copy()