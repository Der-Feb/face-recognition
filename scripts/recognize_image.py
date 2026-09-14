"""Recognize faces in a single image (no webcam required).

Usage (from the project root):
    python -m scripts.recognize_image --image path/to/photo.jpg
    python -m scripts.recognize_image --image photo.jpg --save outputs/annotated.jpg
    python -m scripts.recognize_image --image photo.jpg --threshold 0.5

Prints, per detected face, something like:
    Known: Jordan   similarity=0.83 (threshold 0.40)
    Unknown         similarity=0.34 (threshold 0.40)

With --save it also writes an annotated copy of the image.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import MATCHING_THRESHOLD                  # noqa: E402
from app.enrollment import EnrollmentError                  # noqa: E402
from app.recognition import RecognitionPipeline             # noqa: E402
from app.utils import load_image_rgb, save_image_rgb        # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize faces in a single image",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True,
                        help="path to the image to analyze")
    parser.add_argument("--threshold", type=float, default=MATCHING_THRESHOLD,
                        help="cosine-similarity threshold for Known/Unknown")
    parser.add_argument("--save", default=None,
                        help="optional output path for an annotated copy")
    parser.add_argument("--json", action="store_true",
                        help="print machine-readable JSON results")
    parser.add_argument("--show", action="store_true",
                        help="pop up a window with the annotated result "
                             "(press any key to close)")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 1

    try:
        pipeline = RecognitionPipeline()
    except (FileNotFoundError, EnrollmentError) as exc:
        print(f"SETUP ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        image = load_image_rgb(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        results = pipeline.recognize_image(image)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        print("No face detected in the image.")
        return 0 if args.json else 0

    if args.json:
        import json

        payload = [
            {
                "bbox": [int(v) for v in r.bbox],
                "confidence": round(r.confidence, 4),
                "identity": r.match.identity,
                "known": r.match.is_known,
                "similarity": round(r.match.similarity, 4),
            }
            for r in results
        ]
        print(json.dumps({"results": payload}, indent=2))
    else:
        for result in results:
            print(result.match)
            print(f"  bbox={[int(v) for v in result.bbox]} "
                  f"det_conf={result.confidence:.3f}")

    annotated = None
    if args.save or args.show:
        annotated = image.copy()
        for result in results:
            x1, y1, x2, y2 = (int(v) for v in result.bbox)
            color = (0, 200, 0) if result.match.is_known else (0, 0, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = result.match.display_label
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
            )

    if args.save:
        save_image_rgb(args.save, annotated)
        print(f"Annotated image saved to: {args.save}")

    if args.show:
        # Pop a window so you can SEE the result (run from your own terminal,
        # not a headless shell). Wait for any key, then close cleanly.
        cv2.imshow("Face Recognition result", annotated)
        print("Displaying result in a window. Press any key to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())