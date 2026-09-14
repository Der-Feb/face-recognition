"""Shared helper functions used across the pipeline.

Kept dependency-light on purpose: normalization, similarity and image I/O are
implemented explicitly so the "math behind face recognition" stays visible.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Embedding math (the explicit, teachable part).
# ---------------------------------------------------------------------------

def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit (Euclidean) length, guarding against zero norm.

    Why normalization matters
    -------------------------
    After normalization every embedding lies on the surface of a
    unit hypersphere, so its *direction* is all that matters — its length is
    irrelevant. For a good face model the direction of the embedding encodes
    the identity. Working with unit vectors makes cosine similarity equal to
    the plain dot product and keeps all scores in a fixed, comparable range.

    The epsilon guard is important: an all-zero vector (e.g. a completely
    black aligned face feeding a misbehaving model) would otherwise produce
    NaNs (0 / 0). We return a zero vector and let the caller treat it as a
    "broken" embedding.
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """Cosine similarity between two embeddings.

    cosine = (A . B) / (||A|| * ||B||)

    Reading the score
    -----------------
    * close to +1 : the vectors point in the same direction -> very similar
    * close to  0 : the vectors are nearly orthogonal -> weak similarity
    * close to -1 : opposite directions -> very different

    Because the pipeline L2-normalizes every embedding before storage and
    before matching, both norms equal 1 and cosine similarity reduces to the
    simple dot product:  cosine = A . B   (in [-1, 1]).
    """
    a = l2_normalize(np.asarray(embedding_a, dtype=np.float64).reshape(-1))
    b = l2_normalize(np.asarray(embedding_b, dtype=np.float64).reshape(-1))
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Image I/O.
# ---------------------------------------------------------------------------

def load_image_rgb(image_path: str) -> np.ndarray:
    """Read an image as a BGR numpy array, raising a clear error if it fails.

    Note: OpenCV reads in BGR order by default (that is the byte order the
    SCRFD detector and ArcFace model expect), so we do NOT convert to RGB on
    purpose. The convert happens, explicitly and locally, inside the embedder
    where the network architecture actually requires it.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Could not decode image (unreadable or unsupported format): "
            f"{image_path}"
        )
    return image


def save_image_rgb(image_path: str, image: np.ndarray) -> None:
    """Write a BGR image to disk, creating parent directories as needed."""
    directory = os.path.dirname(os.path.abspath(image_path))
    os.makedirs(directory, exist_ok=True)
    if not cv2.imwrite(image_path, image):
        raise IOError(f"Could not write image: {image_path}")


# ---------------------------------------------------------------------------
# Detection post-processing helpers.
# ---------------------------------------------------------------------------

def distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode SCRFD distance predictions into [x1, y1, x2, y2] boxes.

    SCRFD does NOT regress corners directly; its head predicts the four
    distances (left, top, right, bottom) between an anchor center and the box
    borders. Decoding is therefore:

        x1 = center_x - distance_left
        y1 = center_y - distance_top
        x2 = center_x + distance_right
        y2 = center_y + distance_bottom

    points   : (N, 2) anchor centers (in feature-map pixel coordinates)
    distances: (N, 4) predicted distances (already scaled back by the stride)
    Returns  : (N, 4) decoded boxes.
    """
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Decode SCRFD landmark predictions into (5, 2) point arrays.

    Each of the 5 landmarks (left-eye, right-eye, nose, left-mouth,
    right-mouth) is predicted as an (x, y) offset from the anchor center:

        x_i = center_x + offset_x_i
        y_i = center_y + offset_y_i

    points   : (N, 2) anchor centers
    distances: (N, 10) interleaved (x0, y0, x1, y1, ..., x4, y4)
    Returns  : (N, 5, 2)
    """
    num = distances.shape[0]
    decoded = np.zeros((num, 5, 2), dtype=np.float32)
    for i in range(5):
        decoded[:, i, 0] = points[:, 0] + distances[:, 2 * i]
        decoded[:, i, 1] = points[:, 1] + distances[:, 2 * i + 1]
    return decoded


def non_maximum_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Greedy NMS: keep the highest-scoring box, suppress overlapping ones.

    Face detectors emit many overlapping boxes for the same face. NMS keeps
    the most confident one and removes neighbours whose intersection-over-
    union (IoU) with it exceeds iou_threshold, then repeats on the survivors.

    boxes : (N, 4) in [x1, y1, x2, y2]
    scores: (N,) detector confidences
    Returns indices to keep, sorted by score (highest first).
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1].astype(int)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        # Intersection rectangle over every remaining box.
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1)
        height = np.maximum(0.0, yy2 - yy1)
        intersection = width * height
        union = areas[i] + areas[order[1:]] - intersection
        iou = np.divide(
            intersection, union, out=np.zeros_like(intersection), where=union > 0
        )

        suppress = np.where(iou > iou_threshold)[0]
        keep_idx = np.ones(order.size - 1, dtype=bool)
        keep_idx[suppress] = False
        order = order[1:][keep_idx]

    return np.asarray(keep, dtype=int)


def clip_box_to_image(
    box: np.ndarray, image_height: int, image_width: int
) -> np.ndarray:
    """Clamp a [x1, y1, x2, y2] box to the image bounds."""
    x1, y1, x2, y2 = box
    return np.array(
        [max(0.0, x1), max(0.0, y1), min(image_width - 1.0, x2), min(image_height - 1.0, y2)],
        dtype=np.float32,
    )


def read_bytes_with_retry(url: str) -> bytes:
    raise NotImplementedError("Internal helper; not used by the pipeline.")