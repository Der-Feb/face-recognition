"""Command-line enrollment: turn per-person photo folders into embeddings.

Usage (from the project root):
    python -m scripts.enroll
    python -m scripts.enroll --faces-dir data/faces --strategy all
    python -m scripts.enroll --identity jordan --images a.jpg b.jpg c.jpg

Enrollment reads the standard layout data/faces/<person>/*.jpg by default.
Each image must contain exactly one face. The resulting database is written
to data/embeddings/ as embeddings.npz + meta.json.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.aligner import FaceAligner                  # noqa: E402
from app.config import (                             # noqa: E402
    DETECTOR_CONFIDENCE,
    DETECTOR_INPUT_SIZE,
    DETECTOR_MODEL_PATH,
    DETECTOR_NMS,
    EMBEDDER_MODEL_PATH,
    FACES_DIR,
)
from app.detector import SCRFDDetector                # noqa: E402
from app.embedder import ArcFaceEmbedder              # noqa: E402
from app.enrollment import EnrollmentManager, EnrollmentError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll identities from data/faces/<person>/*.jpg",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--faces-dir", default=FACES_DIR,
                        help="directory that holds one sub-folder per person")
    parser.add_argument("--strategy", choices=["all", "mean"], default="all",
                        help="how to combine multiple embeddings per person")
    parser.add_argument("--identity", default=None,
                        help="enroll ONLY this identity using --images")
    parser.add_argument("--images", nargs="+", default=[],
                        help="photo paths used with --identity")
    parser.add_argument("--threshold-det", type=float, default=DETECTOR_CONFIDENCE,
                        help="minimum face-detection confidence")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        detector = SCRFDDetector(
            model_path=DETECTOR_MODEL_PATH,
            input_size=DETECTOR_INPUT_SIZE,
            confidence_threshold=args.threshold_det,
            nms_threshold=DETECTOR_NMS,
        )
        aligner = FaceAligner()
        embedder = ArcFaceEmbedder(model_path=EMBEDDER_MODEL_PATH)
        manager = EnrollmentManager(
            detector=detector,
            aligner=aligner,
            embedder=embedder,
            strategy=args.strategy,
        )

        if args.identity is not None:
            if not args.images:
                raise EnrollmentError(
                    "--identity requires --images img1.jpg img2.jpg ..."
                )
            count = manager.enroll_identity(args.identity, args.images)
            summary = {args.identity: count}
        else:
            summary = manager.enroll_directory(args.faces_dir)

        manager.save()
        total = sum(summary.values())
        print("\n=== ENROLLMENT COMPLETE ===")
        for name, count in summary.items():
            print(f"  {name:<16} {count} embedding(s)")
        print(f"  saved -> {manager.npz_path}")
        print(f"           {manager.meta_path}")
        print(f"  identities: {len(summary)}, total embeddings: {total}")
        return 0
    except (EnrollmentError, FileNotFoundError, ValueError) as exc:
        print(f"ENROLLMENT FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())