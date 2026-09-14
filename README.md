# face-recognition

Face recognition with ArcFace + ONNX + 5-point face alignment.

**Status:** initial scaffold. The full pipeline (detection, alignment, ArcFace
embeddings, enrollment, matching, webcam + image recognition) is under active
development — README will be completed after the implementation is finished.

## Quick start (current)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m scripts.download_models
```