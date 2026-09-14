"""Live webcam recognition.

Usage (from the project root):
    python -m scripts.recognize
    python -m scripts.recognize --camera 0 --threshold 0.4

Every frame shows each detected face with a box and a label like:
    Jordan 0.83     (score >= threshold -> Known)
    Unknown 0.34    (score <  threshold -> Unknown)

Controls:  q / ESC  -> quit
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MATCHING_THRESHOLD              # noqa: E402
from app.enrollment import EnrollmentError              # noqa: E402
from app.recognition import RecognitionPipeline         # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces live from a webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam device index (0 = default)")
    parser.add_argument("--threshold", type=float, default=MATCHING_THRESHOLD,
                        help="cosine-similarity threshold for Known/Unknown")
    parser.add_argument("--fps-limit", type=int, default=0,
                        help="if >0, downsample the frame to ~this FPS target")
    return parser


def _frame_is_usable(frame: np.ndarray, min_brightness: float = 25.0) -> bool:
    """A camera is 'usable' if it actually delivers a non-black frame.

    Laptops often report a second dark/black "camera" device (covered lenses,
    virtual devices, ...). On Windows an external USB webcam is frequently
    index 1 rather than 0, so we look for the first device that returns real
    content instead of trusting the index blindly.
    """
    if frame is None:
        return False
    return float(np.mean(frame)) >= min_brightness


def open_usable_camera(preferred: int, max_tries: int = 4):
    """Open the preferred camera, falling back to any working device."""
    for idx in range(preferred, max_tries):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        # warm up a few frames so auto-exposure/auto-white-balance settle
        for _ in range(8):
            ok, frame = cap.read()
        if ok and _frame_is_usable(frame):
            return cap, idx
        cap.release()
    return None, None


def _draw(annotated, result) -> None:
    """Draw box + label + confidence for one recognition result."""
    x1, y1, x2, y2 = (int(v) for v in result.bbox)
    known = result.match.is_known
    color = (0, 200, 0) if known else (0, 0, 255)  # green / red
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    label = result.match.display_label
    (text_w, text_h), _baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    top = max(0, y1 - text_h - 8)
    cv2.rectangle(annotated, (x1, top), (x1 + text_w + 8, top + text_h + 8), color, -1)
    cv2.putText(
        annotated, label, (x1 + 4, top + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA,
    )


def main() -> int:
    args = build_parser().parse_args()

    try:
        pipeline = RecognitionPipeline()
    except (FileNotFoundError, EnrollmentError) as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 1

    cap, used_index = open_usable_camera(args.camera)
    if cap is None:
        print(
            f"ERROR: could not find a working webcam (tried indices {args.camera}.."
            f"{args.camera + 3}). Close other apps using the camera and try again.",
            file=sys.stderr,
        )
        return 1
    if used_index != args.camera:
        print(f"NOTE: using camera index {used_index} "
              f"(index {args.camera} was black or unavailable).")

    print("Recognition started. Press q or ESC to quit.")
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR: lost the webcam stream. Quitting.", file=sys.stderr)
            break

        # Optional crude FPS limit: only run recognition on some frames.
        recognize_this = (args.fps_limit <= 0) or (frame_index % args.fps_limit == 0)
        annotated = frame.copy()
        if recognize_this:
            try:
                results = pipeline.recognize_frame(frame)
                for result in results:
                    _draw(annotated, result)
            except ValueError as exc:  # empty enrollment db
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

        cv2.imshow("Face Recognition (ArcFace + ONNX)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())