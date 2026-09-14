"""ArcFace embedding stage on ONNX Runtime.

WHAT GOES IN  -> an aligned face crop (from the aligner)
WHAT COMES OUT -> an L2-normalized embedding vector (default 512 floats)

WHY ARCFACE PRODUCES DISCRIMINATIVE EMBEDDINGS
----------------------------------------------
ArcFace is a deep CNN trained with the *additive angular margin* loss. During
training the network is forced to place faces of the same identity close
together on a unit hypersphere while pushing different identities apart by a
set angular margin. The result: a 512-dimensional vector whose direction is
stable for one person and distinct across people.

WHY ONNX / ONNX RUNTIME
-----------------------
The trained weights are frozen into an ONNX graph — a portable interchange
format for neural networks. ONNX Runtime then executes that graph efficiently
on the CPU (and optionally GPU) without needing the training framework
(PyTorch/TensorFlow) at all. That keeps installation tiny and clearly shows
that "inference" is just a few matrix/tensor operations.

PREPROCESSING (explicit!)
--------------------------
The aligner hands us a (112, 112, 3) BGR crop with values 0..255. The network
was trained on RGB, CHW, float32 tensors, so we:
  1. convert BGR -> RGB
  2. transpose HWC -> CHW
  3. cast to float32
  4. apply the model's normalization convention (auto-detected from the graph:
     InsightFace exports bake the (x - 127.5)/128 normalisation INTO the graph,
     in which case we pass raw values; otherwise we normalize here).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .utils import build_session_options, l2_normalize


class ArcFaceEmbedder:
    """ArcFace face-recognition network wrapped around ONNX Runtime."""

    def __init__(
        self,
        model_path: str,
        image_size: Tuple[int, int] = (112, 112),
        expected_dimension: int = 512,
        providers: Optional[List[str]] = None,
    ) -> None:
        import os

        self.model_path = model_path
        self.image_size = image_size            # (height, width)
        self.expected_dimension = expected_dimension

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace model not found: {model_path}\n"
                "Run `python -m scripts.download_models` to fetch it."
            )

        self.session = ort.InferenceSession(
            model_path, sess_options=build_session_options(),
            providers=providers or ["CPUExecutionProvider"],
        )

        # ==== Inspect the model: input name / declared shape / dtype ========
        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        declared = input_meta.shape
        if isinstance(declared, list) and len(declared) >= 4:
            # NCHW; if the export declares concrete spatial dims, use them.
            if isinstance(declared[2], int) and isinstance(declared[3], int):
                self.image_size = (declared[2], declared[3])
        try:
            self.input_dtype = input_meta.type  # e.g. tensor(float)
        except Exception:
            self.input_dtype = "tensor(float)"

        self.output_names = [o.name for o in self.session.get_outputs()]

        # ==== Detect the model's built-in normalization =====================
        # InsightFace's exported ArcFace graphs embed a Sub(127.5)+Mul(1/128)
        # right at the start; when that is present, raw 0..255 pixels are the
        # correctly normalized input and no external math is required.
        self.input_mean, self.input_std = self._detect_graph_normalization()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _detect_graph_normalization(self) -> Tuple[float, float]:
        """Inspect the first ops of the ONNX graph to learn normalization."""
        try:
            import onnx  # optional; only used to inspect the graph

            model = onnx.load(self.model_path, load_external_data=False)
        except Exception:
            # onnx package not installed OR parsing failed: fall back to the
            # InsightFace default for externally-normalized models.
            return 127.5, 127.5

        head = [n.op_type for n in model.graph.node[:12]]
        has_sub = any("Sub" in op or op.startswith("_") for op in head)
        has_mul = any("Mul" in op for op in head)
        if has_sub and has_mul:
            return 0.0, 1.0      # graph normalizes internally -> feed raw
        return 127.5, 127.5      # we must normalize externally

    def preprocess(self, aligned_face: np.ndarray) -> np.ndarray:
        """Turn a BGR aligned face into the network's expected tensor.

        Returns a float32 array of shape (1, 3, H, W) in range approx. [-1, 1].
        """
        face = aligned_face
        if face.dtype != np.float32:
            face = face.astype(np.float32)

        # 1) BGR -> RGB (the network was trained on RGB).
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        # 2) HWC -> CHW.
        face = np.transpose(face, (2, 0, 1))
        # 3) add a batch axis -> (1, C, H, W).
        face = np.expand_dims(face, axis=0)
        # 4) normalization (only if the graph does not do it internally).
        face = (face - self.input_mean) / self.input_std
        return np.ascontiguousarray(face)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Run inference and return the L2-normalized embedding.

        aligned_face : the (112,112,3) BGR crop from the aligner.
        Returns      : float32 vector of shape (embedding_dimension,).
        """
        tensor = self.preprocess(aligned_face)
        outputs = self.session.run(
            self.output_names, {self.input_name: tensor}
        )
        raw = np.asarray(outputs[0]).reshape(-1)

        if raw.shape[0] != self.expected_dimension:
            raise ValueError(
                f"Model returned embedding of dimension {raw.shape[0]}, "
                f"expected {self.expected_dimension}. Check the model file."
            )

        # Post-processing: L2-normalize (see utils.l2_normalize docstring).
        return l2_normalize(raw).astype(np.float32)