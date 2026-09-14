"""Face detection stage: SCRFD on ONNX Runtime.

WHAT GOES IN  -> a BGR image (H, W, 3), uint8            (e.g. an OpenCV frame)
WHAT COMES OUT -> a list of Detection objects, each with:
                  * bbox        [x1, y1, x2, y2]
                  * confidence  probability in [0, 1]
                  * landmarks   (5, 2) -> [left-eye, right-eye, nose,
                                           left-mouth, right-mouth]

WHY THIS STAGE EXISTS
---------------------
Before we can recognize a face we must first find it. SCRFD is a one-stage
CNN that, given an image, simultaneously predicts:
  * per-anchor objectness scores   (is there a face here?)
  * per-anchor box distances       (where is it?)
  * per-anchor 5-point keypoints   (where are the eyes / nose / mouth?)

The class below runs the network with ONNX Runtime and then does the fully
explicit post-processing: anchor generation -> distance decoding -> score
thresholding -> non-maximum suppression -> scale-back to the original image.

The detector implements a small, clean interface (`detect`) so that it can be
swapped for another detector (YuNet, RetinaFace, ...) without touching the
rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .utils import (
    clip_box_to_image,
    distance_to_bbox,
    distance_to_landmarks,
    non_maximum_suppression,
)

# Strides of the three SCRFD feature maps (8x, 16x, 32x downsampled).
_FEATURE_STRIDES_FPN: Sequence[int] = (8, 16, 32)
# SCRFD predicts *two* anchors at every feature-map location.
_NUM_ANCHORS_PER_LOCATION = 2


@dataclass
class Detection:
    """A single detected face.

    bbox_x1y1x2y2 : pixel coordinates of the box corners in the ORIGINAL image
    confidence    : detector score in [0, 1]
    landmarks     : (5, 2) pixel coordinates in the ORIGINAL image
    """

    bbox: np.ndarray
    confidence: float
    landmarks: np.ndarray

    def __repr__(self) -> str:  # readable when printed from scripts
        x1, y1, x2, y2 = self.bbox
        return (
            f"Detection(bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}], "
            f"conf={self.confidence:.3f}, landmarks={self.landmarks.shape})"
        )


class SCRFDDetector:
    """SCRFD face detector wrapped around ONNX Runtime (CPU-friendly)."""

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.input_size = input_size          # (width, height) fed to the net
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        if not os_path_exists(model_path):
            raise FileNotFoundError(
                f"Face-detector model not found: {model_path}\n"
                "Run `python -m scripts.download_models` to fetch it."
            )
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        # === Inspect the model's declared input shape (educational step). ===
        declared = self.session.get_inputs()[0].shape
        # SCRFD in the InsightFace release ships with dynamic dims ("?" / None);
        # if it declares fixed spatial dims, honour them.
        if len(declared) >= 4 and isinstance(declared[2], int) and isinstance(declared[3], int):
            self.input_size = (declared[3], declared[2])

        self._center_cache: Dict[Tuple[int, int, int], np.ndarray] = {}

    # --- public API --------------------------------------------------------

    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a BGR image; return all faces (possibly many)."""
        if image is None or image.size == 0:
            raise ValueError("detect() received an empty image")
        if not (image.ndim == 3 and image.shape[2] in (3, 4)):
            raise ValueError(
                f"detect() expects an HxWx3 image, got shape {image.shape}"
            )

        original_height, original_width = image.shape[:2]
        net_image, scale = self._resize_and_pad(image)

        # --- forward pass --------------------------------------------------
        # blobFromImage: resize already done; normalizes to (x - 127.5)/128,
        # converts BGR -> RGB (swapRB=True), and produces NCHW float32.
        blob = cv2.dnn.blobFromImage(
            net_image,
            scalefactor=1.0 / 128.0,
            size=self.input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_all, boxes_all, landmarks_all = self._decode_outputs(net_outs)

        # --- filter by confidence across every stride ----------------------
        keep_mask = scores_all >= self.confidence_threshold
        scores_all, boxes_all, landmarks_all = (
            scores_all[keep_mask],
            boxes_all[keep_mask],
            landmarks_all[keep_mask],
        )
        if scores_all.shape[0] == 0:
            return []

        # --- NMS to collapse duplicate detections --------------------------
        keep = non_maximum_suppression(
            boxes_all, scores_all, self.nms_threshold
        )
        scores, boxes, landmarks = (
            scores_all[keep],
            boxes_all[keep],
            landmarks_all[keep],
        )

        # --- map network coords back to the original image -----------------
        boxes = boxes / scale
        landmarks = landmarks / scale

        detections: List[Detection] = []
        for score, box, lmk in zip(scores, boxes, landmarks):
            box = clip_box_to_image(box, original_height, original_width)
            # Skip degenerate boxes left after clipping.
            if box[2] - box[0] < 2 or box[3] - box[1] < 2:
                continue
            detections.append(
                Detection(
                    bbox=box.astype(np.float32),
                    confidence=float(score),
                    landmarks=lmk.reshape(5, 2).astype(np.float32),
                )
            )
        # Most confident face first, which simplifies enrollment logic.
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # --- internal helpers ---------------------------------------------------

    def _resize_and_pad(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize the image into a target-width/target-height square-ish canvas.

        The network takes exactly (input_width, input_height). We resize the
        image preserving its aspect ratio, letterbox it into an all-black
        canvas, and remember the scale factor so boxes can be mapped back.
        """
        image_height, image_width = image.shape[:2]
        target_width, target_height = self.input_size

        image_ratio = image_height / float(image_width)
        model_ratio = target_height / float(target_width)
        if image_ratio > model_ratio:
            new_height = target_height
            new_width = max(1, int(new_height / image_ratio))
        else:
            new_width = target_width
            new_height = max(1, int(new_width * image_ratio))

        scale = new_height / float(image_height)
        resized = cv2.resize(image, (new_width, new_height))
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        return canvas, scale

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        """The (K * num_anchors, 2) anchor centers for one feature-map level.

        Every cell of the feature map is an "anchor center" on the original
        image grid (resized image coordinates!). The two-per-location anchors
        share the same center — only their aspect-ratio priors differ — so we
        simply duplicate each center. Results are cached because the grid is
        identical for every frame at a given input size.
        """
        key = (height, width, stride)
        if key in self._center_cache:
            return self._center_cache[key]

        # mgrid gives row/col per pixel; we want (x, y) per pixel.
        yy, xx = np.mgrid[0:height, 0:width]
        centers = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
        centers *= stride  # feature-map cell -> original image coordinates
        # Duplicate each center once for the second anchor.
        centers = np.repeat(centers, _NUM_ANCHORS_PER_LOCATION, axis=0)

        if len(self._center_cache) < 32:
            self._center_cache[key] = centers
        return centers

    def _decode_outputs(
        self, net_outs: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Conv adapter: anchor generation + distance decoding for all levels.

        SCRFD produces 9 tensors (3 feature-map levels x [scores, boxes, kps]).
        For each level we:
          1. multiply the box/kps distances by the level's stride
             (the head regresses distances in feature-map units);
          2. add them to the anchor centers -> real pixel coordinates;
          3. keep scores as the level's confidences.
        Results from every level are concatenated into one big candidate pool.
        """
        image_height, image_width = self.input_size[1], self.input_size[0]

        all_scores: List[np.ndarray] = []
        all_boxes: List[np.ndarray] = []
        all_landmarks: List[np.ndarray] = []

        # NOTE: the InsightFace release of SCRFD lists its 9 outputs grouped
        # BY TYPE, not interleaved per feature-map level:
        #   [score_8, score_16, score_32,
        #    bbox_8,   bbox_16,  bbox_32,
        #    kps_8,    kps_16,   kps_32]
        # So level `i` reads scores[ i ], boxes[ i + 3 ], kps[ i + 6 ].
        for level_index, stride in enumerate(_FEATURE_STRIDES_FPN):
            # --- squeeze raw network outputs for this level ---
            scores_raw = np.asarray(net_outs[level_index]).reshape(-1)
            boxes_raw = np.asarray(net_outs[level_index + 3]).reshape(-1, 4)
            landmarks_raw = np.asarray(net_outs[level_index + 6]).reshape(-1, 10)

            height = image_height // stride
            width = image_width // stride
            boxes_raw = boxes_raw * stride            # scale to image units
            landmarks_raw = landmarks_raw * stride

            centers = self._anchor_centers(height, width, stride)
            if centers.shape[0] == boxes_raw.shape[0]:
                boxes = distance_to_bbox(centers, boxes_raw)
                landmarks = distance_to_landmarks(centers, landmarks_raw)
            else:  # pragma: no cover - safety net for odd model exports
                boxes = boxes_raw
                landmarks = landmarks_raw.reshape(-1, 5, 2)

            all_scores.append(scores_raw)
            all_boxes.append(boxes)
            all_landmarks.append(landmarks.reshape(-1, 5, 2))

        return (
            np.concatenate(all_scores),
            np.concatenate(all_boxes).astype(np.float32),
            np.concatenate(all_landmarks).astype(np.float32),
        )


def os_path_exists(path: str) -> bool:
    import os

    return os.path.exists(path)