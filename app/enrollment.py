"""Enrollment: turning raw per-person face photos into an embedding database.

WHAT HAPPENS HERE
-----------------
For each person who needs to be recognized later, we take several photos and
reduce each to a fixed-size embedding using the SAME pipeline used during
recognition (detect -> align -> ArcFace -> L2-normalize). The result is stored
as a compact database — just numeric vectors, no raw images.

WHY ENROLL BEFORE RECOGNITION
-----------------------------
Matching is comparison against a reference. There is nothing to compare to
until identities have been embedded and stored, so enrollment is a mandatory
first step. Only exactly-one-face photos are accepted per image so enrollment
cannot silently enroll the wrong person.

WHY MULTIPLE IMAGES PER PERSON
------------------------------
One face photo captures one pose + one lighting. Several photos capture the
variation the network will encounter later, which makes matching much more
robust. How the multiple embeddings are combined is controlled by the
`strategy` argument (see config.MULTI_EMBEDDING_STRATEGY):

  * "all"   keep every embedding. The matcher compares the query against each
            and reports the best score. Robust -> RECOMMENDED, default.
  * "mean"  average the embeddings and L2-normalize the average into a single
            representative embedding. Compact, but smooths away variation
            (and can forget rare poses).

STORAGE FORMAT
--------------
  data/embeddings/embeddings.npz     numpy archive, one array per identity
  data/embeddings/meta.json          human-readable summary + pipeline info

We deliberately store ONLY embeddings (128 numeric vectors the size of
1024 bytes each) and never the raw photos, keeping biometric data minimal.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

from .config import (
    EMBEDDING_DIMENSION,
    EMBEDDINGS_META_PATH,
    EMBEDDINGS_NPZ_PATH,
    IMAGE_EXTENSIONS,
)
from .utils import l2_normalize, load_image_rgb


class EnrollmentError(Exception):
    """Raised for user-level enrollment problems (bad folder, no faces...)."""


class EnrollmentManager:
    """Builds and persists the per-identity embedding database."""

    def __init__(
        self,
        detector,
        aligner,
        embedder,
        npz_path: str = EMBEDDINGS_NPZ_PATH,
        meta_path: str = EMBEDDINGS_META_PATH,
        strategy: str = "all",
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.npz_path = npz_path
        self.meta_path = meta_path
        self.strategy = strategy.lower()
        if self.strategy not in ("all", "mean"):
            raise ValueError(
                f"Unknown multi-embedding strategy '{strategy}' "
                "(choose 'all' or 'mean')"
            )
        self._embeddings: Dict[str, np.ndarray] = {}  # name -> (N, D)

    # ------------------------------------------------------------------
    # Embedding a single face image
    # ------------------------------------------------------------------

    def embed_face(self, image: np.ndarray) -> np.ndarray:
        """detect -> align -> embed -> L2-normalized vector for ONE face image.

        Raises EnrollmentError if the image contains zero or multiple faces,
        so we never enroll a wrong or ambiguous identity.
        """
        faces = self.detector.detect(image)
        if len(faces) == 0:
            raise EnrollmentError("no face detected in image")
        if len(faces) > 1:
            raise EnrollmentError(
                f"multiple faces ({len(faces)}) detected; "
                "enrollment expects exactly one face per image"
            )
        face = faces[0]
        aligned, _ = self.aligner.align(image, face.landmarks)
        return self.embedder.get_embedding(aligned)

    def embed_image_file(self, image_path: str) -> np.ndarray:
        return self.embed_face(load_image_rgb(image_path))

    # ------------------------------------------------------------------
    # Building the database
    # ------------------------------------------------------------------

    def enroll_identity(
        self, name: str, image_paths: List[str]
    ) -> int:
        """Enroll one person from a list of image files; returns #embeddings.

        After enrolling, embeddings are normalized, then combined according to
        the configured strategy and stored under `name` (replacing any
        previous enrollment of the same name).
        """
        if not image_paths:
            raise EnrollmentError(f"No images given for identity '{name}'")

        embeddings: List[np.ndarray] = []
        for path in image_paths:
            if not os.path.exists(path):
                raise EnrollmentError(f"Image not found: {path}")
            try:
                vec = self.embed_image_file(path)
            except EnrollmentError as exc:
                raise EnrollmentError(f"{os.path.basename(path)}: {exc}") from exc
            embeddings.append(np.asarray(vec, dtype=np.float32))

        array = np.stack(embeddings, axis=0)  # (N, D)
        if self.strategy == "all":
            self._embeddings[name] = array
        else:  # 'mean': one representative vector per person
            mean_vec = l2_normalize(array.mean(axis=0)).astype(np.float32)
            self._embeddings[name] = mean_vec.reshape(1, -1)
        return array.shape[0]

    def scan_faces_dir(self, faces_dir: str) -> Dict[str, List[str]]:
        """Map identity folder names to their ordered image paths.

        Expected layout:
            faces_dir /
                jordan/  photo1.jpg photo2.jpg ...
                alice/   photo1.jpg ...
        """
        if not os.path.isdir(faces_dir):
            raise EnrollmentError(f"faces directory not found: {faces_dir}")

        result: Dict[str, List[str]] = {}
        for entry in sorted(os.listdir(faces_dir)):
            person_dir = os.path.join(faces_dir, entry)
            if not os.path.isdir(person_dir):
                continue
            images = [
                os.path.join(person_dir, f)
                for f in sorted(os.listdir(person_dir))
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]
            if images:
                result[entry] = images
        if not result:
            raise EnrollmentError(
                f"no identity folders with images found under {faces_dir}\n"
                "Make dirs like  data/faces/jordan/  and drop photos inside."
            )
        return result

    def enroll_directory(self, faces_dir: str) -> Dict[str, int]:
        """Enroll every identity folder under faces_dir; return per-person count."""
        plan = self.scan_faces_dir(faces_dir)
        summary: Dict[str, int] = {}
        for name, paths in plan.items():
            summary[name] = self.enroll_identity(name, paths)
        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, npz_path: Optional[str] = None, meta_path: Optional[str] = None) -> None:
        """Write the embedding database (.npz) + a human-readable meta file."""
        if not self._embeddings:
            raise EnrollmentError("nothing to save: no identities enrolled yet")

        npz_path = npz_path or self.npz_path
        meta_path = meta_path or self.meta_path
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)

        np.savez(npz_path, **self._embeddings)

        meta = {
            "model": os.path.basename(self.embedder.model_path),
            "embedding_dimension": EMBEDDING_DIMENSION,
            "strategy": self.strategy,
            "identities": {
                name: int(array.shape[0]) for name, array in self._embeddings.items()
            },
            "total_embeddings": int(
                sum(array.shape[0] for array in self._embeddings.values())
            ),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    def load(self, npz_path: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load the database; raises EnrollmentError when it does not exist."""
        npz_path = npz_path or self.npz_path
        if not os.path.exists(npz_path):
            raise EnrollmentError(
                f"enrollment database not found: {npz_path}\n"
                "Run `python -m scripts.enroll` first."
            )
        with np.load(npz_path) as data:
            self._embeddings = {name: data[name] for name in data.files}
        return dict(self._embeddings)

    def clear(self) -> None:
        self._embeddings = {}