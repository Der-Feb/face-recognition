"""Download and prepare the ONNX models for the face-recognition project.

Models fetched (currently, the official InsightFace v0.7 release package):

  1. models/det_10g.onnx        SCRFD face detector (boxes + 5 landmarks), ~16 MB
  2. models/w600k_r50.onnx      ArcFace R50 recognizer (512-dim embedding), ~166 MB

Both are distributed inside the single official archive:

  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

Why this archive?
  * It is the official InsightFace release (the maintainers of SCRFD and ArcFace).
  * It contains the detector *and* the recognizer that InsightFace itself pairs,
    so their inputs/preprocessing are guaranteed to match (RGB, 112x112, etc.).
  * The InsightFace project is MIT licensed.

Usage:
  python -m scripts.download_models

The big model binaries are intentionally NOT committed to Git; run this once
after cloning to populate the models/ directory.
"""

from __future__ import annotations

import os
import shutil
import sys
import zipfile
import urllib.request

# Allow running as `python -m scripts.download_models` from the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.config import DETECTOR_MODEL_PATH, EMBEDDER_MODEL_PATH, MODELS_DIR  # noqa: E402

_BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/"
    "releases/download/v0.7/buffalo_l.zip"
)

# Files we need from inside the archive.
_REQUIRED_MEMBERS = {
    "det_10g.onnx": DETECTOR_MODEL_PATH,
    "w600k_r50.onnx": EMBEDDER_MODEL_PATH,
}


def _human(size: float) -> str:
    """Format a byte count for human-readable progress output."""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def main() -> int:
    if os.path.exists(DETECTOR_MODEL_PATH) and os.path.exists(EMBEDDER_MODEL_PATH):
        print("Both models already exist:")
        print(f"  {DETECTOR_MODEL_PATH}")
        print(f"  {EMBEDDER_MODEL_PATH}")
        return 0

    os.makedirs(MODELS_DIR, exist_ok=True)
    archive_path = os.path.join(MODELS_DIR, "buffalo_l.zip")

    try:
        if not os.path.exists(archive_path) or os.path.getsize(archive_path) == 0:
            print(f"Downloading {_BUFFALO_L_URL}")
            request = urllib.request.Request(
                _BUFFALO_L_URL, headers={"User-Agent": "face-recognition-educational"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                with open(archive_path, "wb") as out:
                    while True:
                        chunk = response.read(1024 * 512)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = min(100.0, downloaded * 100.0 / total)
                            sys.stdout.write(
                                f"\r  {_human(downloaded)} / {_human(total)} ({pct:.0f}%)"
                            )
                        else:
                            sys.stdout.write(f"\r  {_human(downloaded)}")
                        sys.stdout.flush()
            print()
        else:
            print(f"Reusing existing archive: {archive_path}")

        print("Extracting required ONNX files...")
        with zipfile.ZipFile(archive_path) as zf:
            names = set(zf.namelist())
            missing = [name for name in _REQUIRED_MEMBERS if name not in names]
            if missing:
                print(f"ERROR: archive does not contain: {missing}", file=sys.stderr)
                return 1
            for name, dest in _REQUIRED_MEMBERS.items():
                print(f"  {name} -> {dest}")
                with zf.open(name) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        os.remove(archive_path)
        print("Done. Models are ready in models/.")
        return 0
    except Exception as exc:  # network / disk / zip errors -> clean message
        print(f"ERROR: could not download models: {exc}", file=sys.stderr)
        print(
            "Check your internet connection and try again. "
            "You can also download buffalo_l.zip manually from the InsightFace "
            "releases page and place det_10g.onnx / w600k_r50.onnx into models/.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())