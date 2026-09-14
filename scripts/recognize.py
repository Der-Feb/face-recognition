"""Live webcam recognition.

Usage (from the project root):
    python -m scripts.recognize
    python -m scripts.recognize --camera 0 --threshold 0.4
    python -m scripts.recognize --skip 2 --det-size 480   # faster on CPU

Every frame shows each detected face with a box and a label like:
    Jordan 0.83     (score >= threshold -> Known)
    Unknown 0.34    (score <  threshold -> Unknown)

Performance: running detection+embedding on EVERY frame is slow on a CPU.
By default the pipeline is therefore re-run only every 3rd frame -- the live
video keeps streaming smoothly and the last boxes/labels stay on screen in
between. Use --skip 1 to process every frame, or --skip 5+ for a low-end CPU.
--det-size shrinks the SCRFD input image for a further speedup.

Controls:  q / ESC  -> quit
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (                            # noqa: E402
    DETECTOR_CONFIDENCE,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS,
    MATCHING_THRESHOLD,
)
from app.detector import SCRFDDetector               # noqa: E402
from app.enrollment import EnrollmentError           # noqa: E402
from app.recognition import RecognitionPipeline      # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces live from a webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam device index (0 = default)")
    parser.add_argument("--threshold", type=float, default=MATCHING_THRESHOLD,
                        help="cosine-similarity threshold for Known/Unknown")
    parser.add_argument("--skip", type=int, default=3,
                        help="run detection+embedding once every N frames "
                             "(1 = every frame; larger = faster)")
    parser.add_argument("--det-size", type=int, default=-1,
                        help="square size fed to the SCRFD detector "
                             f"(default {DETECTOR_INPUT_SIZE[0]}; smaller is "
                             "much faster, e.g. 480 or 416)")
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

    if args.skip < 1:
        print("ERROR: --skip must be >= 1.", file=sys.stderr)
        return 1

    if args.det_size > 0:
        detector = SCRFDDetector(
            model_path=DETECTOR_MODEL_PATH,
            input_size=(args.det_size, args.det_size),
            confidence_threshold=DETECTOR_CONFIDENCE,
            nms_threshold=DETECTOR_NMS,
        )
    else:
        detector = None

    try:
        pipeline = RecognitionPipeline(detector=detector)
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
    last_results = []
    fps_window = 30
    fps_times = []

    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR: lost the webcam stream. Quitting.", file=sys.stderr)
            break

        annotated = frame.copy()
        # Re-run the (expensive) pipeline only every --skip frames; in between,
        # keep the previous boxes/labels so the feed stays interactive.
        if frame_index % args.skip == 0:
            try:
                last_results = pipeline.recognize_frame(frame)
            except ValueError as exc:  # empty enrollment db
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1

        for result in last_results:
            _draw(annotated, result)

        # small debug overlay: display + recognition rates
        fps_times.append(t0)
        if len(fps_times) > fps_window:
            fps_times.pop(0)
        if len(fps_times) > 1:
            span = fps_times[-1] - fps_times[0]
            disp = len(fps_times) / span if span > 0 else 0
            cv2.putText(annotated, f"display {disp:.0f} fps",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 2, cv2.LINE_AA)

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