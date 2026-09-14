"""Recognition pipeline: compose the whole detection->embedding->matching flow.

This module is deliberately thin: it wires together the detector, aligner,
embedder and matcher so that each stage stays explicit, testable and
replaceable. It is the only place where the stages know about each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .aligner import FaceAligner
from .config import (
    DETECTOR_CONFIDENCE,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS,
    EMBEDDER_MODEL_PATH,
    EMBEDDINGS_NPZ_PATH,
    MATCHING_THRESHOLD,
)
from .detector import SCRFDDetector
from .embedder import ArcFaceEmbedder
from .matcher import EmbeddingMatcher, MatchResult


@dataclass
class FaceRecognitionResult:
    """Everything the pipeline knows about ONE face in an image."""

    bbox: np.ndarray          # [x1, y1, x2, y2]
    confidence: float         # detector score
    landmarks: np.ndarray     # (5, 2)
    aligned_face: np.ndarray  # (112, 112, 3) BGR crop (for inspection/demo)
    embedding: np.ndarray     # (D,) L2-normalized
    match: MatchResult        # identity / similarity / threshold decision


class RecognitionPipeline:
    """Entry point used by the CLI scripts (webcam + single image)."""

    def __init__(
        self,
        detector: Optional[SCRFDDetector] = None,
        aligner: Optional[FaceAligner] = None,
        embedder: Optional[ArcFaceEmbedder] = None,
        matcher: Optional[EmbeddingMatcher] = None,
        threshold: Optional[float] = None,
    ) -> None:
        # Build each stage from defaults unless the caller injects its own
        # (injection is how unit tests replace heavy/real components).
        self.detector = detector or SCRFDDetector(
            model_path=DETECTOR_MODEL_PATH,
            input_size=DETECTOR_INPUT_SIZE,
            confidence_threshold=DETECTOR_CONFIDENCE,
            nms_threshold=DETECTOR_NMS,
        )
        self.aligner = aligner or FaceAligner()
        self.embedder = embedder or ArcFaceEmbedder(
            model_path=EMBEDDER_MODEL_PATH
        )
        self.matcher = matcher or EmbeddingMatcher(
            threshold=threshold if threshold is not None else MATCHING_THRESHOLD
        )
        if matcher is None:
            # Load the enrollment database written by `scripts.enroll`.
            # If it does not exist yet, leave the matcher empty and let
            # recognize_image() raise a friendly, actionable error.
            try:
                with np.load(EMBEDDINGS_NPZ_PATH) as data:
                    self.matcher.set_enrollment(
                        {name: data[name] for name in data.files}
                    )
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # The recognizable flow (explicitly, stage after stage)
    # ------------------------------------------------------------------

    def _recognize_face(self, image: np.ndarray, face) -> FaceRecognitionResult:
        landmarks = np.asarray(face.landmarks, dtype=np.float32)
        # 1) align the detected face into the canonical ArcFace layout
        aligned, _ = self.aligner.align(image, landmarks)
        # 2) embed the aligned face
        embedding = self.embedder.get_embedding(aligned)
        # 3) match against the enrollment database
        match = self.matcher.match(embedding)
        return FaceRecognitionResult(
            bbox=face.bbox,
            confidence=face.confidence,
            landmarks=landmarks,
            aligned_face=aligned,
            embedding=embedding,
            match=match,
        )

    def recognize_image(self, image: np.ndarray) -> List[FaceRecognitionResult]:
        """Recognize every face in one image (BGR). Handles 0..N faces."""
        if not self.matcher.has_enrollments():
            raise ValueError(
                "enrollment database is empty; run `python -m scripts.enroll` first"
            )
        faces = self.detector.detect(image)
        return [self._recognize_face(image, face) for face in faces]

    def recognize_frame(self, frame: np.ndarray) -> List[FaceRecognitionResult]:
        """Recognize every face in a webcam frame (same as recognize_image)."""
        return self.recognize_image(frame)