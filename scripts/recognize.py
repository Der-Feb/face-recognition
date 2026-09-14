"""Live webcam recognition.

Usage (from the project root):
    python -m scripts.recognize
    python -m scripts.recognize --camera 0 --threshold 0.4
    python -m scripts.recognize --skip 2 --det-size 480   # faster on CPU

Every frame shows each detected face with a box and a label like:
    Jordan 0.83     (score >= threshold -> Known)
    Unknown 0.34    (score <  threshold -> Unknown)

Performance design
------------------
Recognition (SCRFD + ArcFace, ~0.4 s on a laptop CPU) runs on a SEPARATE
thread, so it can never stall the video: the main loop only reads the camera
and displays frames, which runs at the camera's own maximum rate (≈30 fps for
a normal USB webcam — its hardware limit, regardless of software). On every
--skip-th frame the worker grabs the newest frame and refreshes the labels.

Controls:  q / ESC  -> quit
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
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
    parser.add_argument("--res", default="640x480",
                        help="camera resolution WxH (e.g. 1280x720)")
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


def open_usable_camera(preferred: int, max_tries: int = 4, res: tuple = (640, 480)):
    """Open the preferred camera, falling back to any working device.

    Requests the MJPG codec + the given resolution and a high FPS target: many
    webcams deliver more frames with MJPG and a smaller frame than on their
    default raw/RGB pipeline. We then warm the camera up (auto-exposure /
    white-balance need a few frames) and require real (non-black) content.
    """
    for idx in range(preferred, max_tries):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            continue
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except AttributeError:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        cap.set(cv2.CAP_PROP_FPS, 60)  # ask high; the camera gives what it can
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


class _SharedState:
    """Thread-safe hand-off between the camera loop and the recognition worker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame = None        # newest frame from the camera (BGR)
        self.results: list = []  # latest recognition output (may lag the frame)

    def publish_frame(self, frame) -> None:
        with self.lock:
            self.frame = frame

    def snapshot(self):
        """Copy of (frame, current results) without blocking the producer long."""
        with self.lock:
            return self.frame, list(self.results)

    def publish_results(self, results) -> None:
        with self.lock:
            self.results = results


def recognition_worker(pipeline, state: _SharedState, skip: int,
                       stop: threading.Event) -> None:
    """Run the (expensive) pipeline on a background thread.

    The camera loop keeps streaming/displaying at the camera's true max FPS;
    this thread simply updates the labels every --skip-th *new* frame. ONNX
    sessions are thread-safe, so calling recognize_frame() here is safe.
    """
    seen = 0
    while not stop.is_set():
        frame, _ = state.snapshot()
        if frame is None:
            time.sleep(0.005)
            continue
        if seen % skip != 0:
            seen += 1
            continue
        seen += 1
        try:
            results = pipeline.recognize_frame(frame)
        except ValueError as exc:  # empty enrollment db
            print(f"ERROR: {exc}", file=sys.stderr)
            stop.set()
            return
        state.publish_results(results)


def main() -> int:
    args = build_parser().parse_args()

    if args.skip < 1:
        print("ERROR: --skip must be >= 1.", file=sys.stderr)
        return 1

    try:
        width, height = (int(x) for x in args.res.lower().split("x"))
    except ValueError:
        print(f"ERROR: --res must look like 640x480, got {args.res!r}.",
              file=sys.stderr)
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

    cap, used_index = open_usable_camera(args.camera, res=(width, height))
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

    # Background recognition: the display loop never waits for the pipeline.
    state = _SharedState()
    stop = threading.Event()
    worker = threading.Thread(
        target=recognition_worker,
        args=(pipeline, state, args.skip, stop),
        daemon=True,
    )
    worker.start()

    print("Recognition started. Press q or ESC to quit.")
    fps_window = 30
    fps_times = []

    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: lost the webcam stream. Quitting.", file=sys.stderr)
                break

            state.publish_frame(frame)
            _, results = state.snapshot()

            annotated = frame.copy()
            for result in results:
                _draw(annotated, result)

            # display-rate counter so you can SEE the video is not slowed down
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
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()
        worker.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())