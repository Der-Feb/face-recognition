# face-recognition

A complete, runnable **face-recognition system** built for a Computer Science
practical. It detects faces in photos or from a **live webcam**, aligns them,
converts them into **512-dimensional numerical embeddings** using the
**ArcFace** deep-learning model (running locally as an ONNX model — no cloud,
no API key, no internet), and **recognises people** by matching their embedding
against a small local enrolment database.

Everything runs on a normal CPU with plain Python. Every single stage of the
pipeline is explicit, inspectable and testable — nothing is hidden inside a
black box — which makes the project easy to explain, easy to mark and easy to
extend.

---

## Table of contents

1. [Features](#1-features)
2. [How the recognition pipeline works](#2-how-the-recognition-pipeline-works)
3. [Project structure](#3-project-structure)
4. [Requirements](#4-requirements)
5. [Step-by-step setup (Steps 1–10)](#5-step-by-step-setup)
6. [Configuration reference](#6-configuration-reference)
7. [CLI tools reference](#7-cli-tools-reference)
8. [Understanding the matching threshold](#8-understanding-the-matching-threshold)
9. [Sample output and test results](#9-sample-output-and-test-results)
10. [Running the unit tests](#10-running-the-unit-tests)
11. [Troubleshooting](#11-troubleshooting)
12. [Ideas to extend the project](#12-ideas-to-extend-the-project)
13. [Privacy](#13-privacy)
14. [Licence and model credits](#14-licence-and-model-credits)

---

## 1. Features

- **Face detection** with the state-of-the-art **SCRFD** model — finds every
  face in an image and returns a **bounding box + 5 facial landmarks**
  (left/right eye, nose tip, left/right mouth corner).
- **Face alignment** — a similarity (rotation + scale + translation)
  transformation that normalises each face into the canonical 112×112 pose
  that the recognition network expects. If you stand the same way in two
  photos, the two faces become directly comparable.
- **Embeddings with ArcFace** — the aligned face is reduced to a
  512-dimensional vector. A face becomes a *point in a 512-dimensional space*;
  the same person maps to roughly the same point regardless of pose, lighting
  or expression.
- **Enrolment** — build a labelled database of embeddings from
  `data/faces/<person>/` photo folders (one to several photos per person),
  stored as `embeddings.npz` + `meta.json`.
- **Matching by cosine similarity** — compare a new face to every enrolled
  face and decide **Known** / **Unknown** using a configurable threshold.
- **Three runnable tools**:
  - recognise a single photo (`scripts/recognize_image.py`),
  - recognise live from a **webcam** (`scripts/recognize.py`),
  - visualise every stage in pop-up windows for your demo
    (`scripts/visualize.py`).
- **25 passing unit tests** (no models required).
- Everything **runs on CPU** with pure `pip` — works on any simple laptop.

---

## 2. How the recognition pipeline works

The whole system is a chain of five well-known computer-vision steps.
A face flows through the chain from left to right:

```
                     INPUT IMAGE (photo or webcam frame, BGR)
                  ─────────────────────────────────────────────►
                                         │
                 ┌───────────────────────▼────────────────────────┐
                 │                                                │
                 │  1. DETECTION  SCRFD (det_10g.onnx, 640 × 640) │
                 │                                                │
                 │   Finds every face, its bounding box and the   │
                 │   5 landmark points. Several boxes overlap, so │
                 │   Non-Maximum-Suppression keeps the best one.  │
                 └───────────────────────┬────────────────────────┘
                                         │  bbox [x1,y1,x2,y2] + 5 landmarks
                                         ▼
                 ┌───────────────────────▼────────────────────────┐
                 │                                                │
                 │  2. ALIGNMENT  similarity transform (Geometry) │
                 │                                                │
                 │   Warps the face so the 5 detected landmarks   │
                 │   land exactly on a fixed reference template.  │
                 │   A face photographed from the side or slightly│
                 │   rotated is now "corrected" into a standard   │
                 │   frontal 112 × 112 crop.                      │
                 └───────────────────────┬────────────────────────┘
                                         │  aligned_face (112, 112, 3)
                                         ▼
                 ┌───────────────────────▼────────────────────────┐
                 │                                                │
                 │  3. EMBEDDING  ArcFace (w600k_r50.onnx)        │
                 │                                                │
                 │   The aligned face becomes a 512-dim vector.   │
                 │   Nearby points in this space = same person.   │
                 │   Output is L2-normalized so all embeddings    │
                 │   live on a unit sphere (easy to compare).     │
                 └───────────────────────┬────────────────────────┘
                                         │  embedding (512,)
                                         ▼
                 ┌───────────────────────▼────────────────────────┐
                 │                                                │
                 │  4. ENROLLMENT   (run once per person)         │
                 │                                                │
                 │   Each enrolment photo is embedded with steps  │
                 │   1-3 and stored: embeddings.npz + meta.json   │
                 └───────────────────────┬────────────────────────┘
                                         │  labelled gallery
                                         ▼
                 ┌───────────────────────▼────────────────────────┐
                 │                                                │
                 │  5. MATCHING  cosine similarity + threshold    │
                 │                                                │
                 │   New face  vs  every enrolled face.           │
                 │   best score ≥ threshold  →  "Known: <name>"   │
                 │   best score <  threshold  →  "Unknown"        │
                 └───────────────────────┬────────────────────────┘
                                         ▼
                                  DECISION + annotated image
```

### 2.1 Why alignment matters so much

A raw photo may show a face slightly tilted, at an angle, or at slightly
different scale. If we just compared raw pixels, those small differences would
dominate. Instead:

1. SCRFD gives us 5 landmark positions (left eye, right eye, nose, left mouth
   corner, right mouth corner).
2. We compute an affine similarity transform `M` (`cv2.estimateAffinePartial2D`)
   that moves those landmarks **exactly** onto a fixed reference template.
3. We warp the face with `M` (`cv2.warpAffine`) and crop a 112×112 image.

The reference template used (the standard ArcFace one, in pixels inside the
112×112 crop) is:

```
 left eye    (38.29,  51.70)
 right eye   (73.53,  51.50)
 nose        (56.03,  71.74)
 mouth left  (41.55,  92.37)
 mouth right (70.73,  92.20)
```

After this, the same person always produces almost the *same* aligned face, no
matter how the photograph was taken — a tiny step that hugely improves
recognition accuracy.

### 2.2 The embedding and cosine similarity

ArcFace maps a 112×112 face crop to a 512-float vector `e`, normalised so that
`‖e‖ = 1`. Comparing two faces `a` and `b` is then just the **cosine of the
angle between the two vectors**:

```
cos(a, b) = Σᵢ aᵢ bᵢ        (both vectors have unit length)
```

- identical faces        →  cosine ≈ 1.0
- the same person, other photos →  usually 0.5 – 0.9
- two different people    →  usually < 0.3

The decision rule is trivial: the enrolled person with the **highest** score is
the candidate; if that score is at/above the threshold the face is **Known**,
otherwise it is **Unknown**.

### 2.3 Why ONNX?

The two deep networks (SCRFD detector and ArcFace recognizer) are exported to
**ONNX** — the *Open Neural Network Exchange* format — and executed with
**onnxruntime**. ONNX lets you run trained deep-learning models on a plain CPU
without installing TensorFlow/PyTorch or a GPU, which is exactly the right
level for an educational project.

---

## 3. Project structure

```
face-recognition/
│
├── app/                        # the actual implementation (importable package)
│   ├── __init__.py
│   ├── config.py               # all settings: paths, sizes, threshold (env-tunable)
│   ├── utils.py                # L2 norm, cosine similarity, NMS, image I/O
│   ├── detector.py             # stage 1 — SCRFD face detector (ONNX)
│   ├── aligner.py              # stage 2 — 5-point similarity-transform alignment
│   ├── embedder.py             # stage 3 — ArcFace embedding (ONNX)
│   ├── enrollment.py           # stage 4 — build / save / load the embedding DB
│   ├── matcher.py              # stage 5 — cosine similarity + threshold decision
│   └── recognition.py          # the pipeline that wires stages 1–5 together
│
├── scripts/                    # command-line tools you run from the terminal
│   ├── download_models.py      # downloads the two ONNX models (~180 MB)
│   ├── enroll.py               # enrol all photos in data/faces/
│   ├── recognize.py            # live webcam recognition
│   ├── recognize_image.py      # recognise one photo
│   └── visualize.py            # pop-up windows for every stage (demo tool)
│
├── tests/                      # 25 unit tests (no models needed)
│   ├── conftest.py
│   ├── test_alignment.py
│   ├── test_embedding.py
│   └── test_matching.py
│
├── data/                       # YOUR data (git-ignored)
│   ├── faces/<person>/…jpg     # enrolment photos — one folder per person
│   └── embeddings/             # generated: embeddings.npz + meta.json
│
├── models/                     # ONNX models (downloaded, git-ignored)
│   ├── det_10g.onnx            # SCRFD detector
│   └── w600k_r50.onnx          # ArcFace recognition model
│
├── outputs/                    # annotated results saved by the tools (git-ignored)
├── requirements.txt            # everything you need to `pip install`
├── LICENSE                     # MIT
└── README.md
```

---

## 4. Requirements

- **Python 3.9+** (the project was developed and tested on Python 3.14).
- Works on **Windows, Linux and macOS** (CPU-only).
- No GPU, no CUDA, no internet during inference.
- When you run the webcam tool you need a webcam (laptop camera or any USB
  webcam) and permission for Python to access it.

The dependencies (`requirements.txt`) are small and famous libraries:

| package        | purpose                                        |
|----------------|------------------------------------------------|
| `onnxruntime`  | executes the two ONNX models on CPU            |
| `opencv-python`| camera capture, image ops, drawing, windows    |
| `numpy`        | vectorised math (cosine similarity, NMS, …)    |
| `onnx`         | inspecting the model input/output shapes       |
| `pytest`       | running the unit tests                         |

---

## 5. Step-by-step setup

The full journey to a working recognition system, in order.

### Step 1 — Get the code

```bash
git clone https://github.com/kabanda-jordan/face-recognition.git
cd face-recognition
```

### Step 2 — Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your terminal prompt.

### Step 3 — Install the dependencies

```bash
pip install -r requirements.txt
```

Everything is CPU-only and pure pip — no special system installs.

### Step 4 — Download the ONNX models

```bash
python -m scripts.download_models
```

This downloads the official **InsightFace v0.7** pack `buffalo_l.zip`
(~180 MB) and extracts the two files into `models/`:

| file             | size   | model      | role                               |
|------------------|--------|------------|------------------------------------|
| `det_10g.onnx`   | ~16 MB | **SCRFD**  | face detection + 5 landmarks       |
| `w600k_r50.onnx` | ~166 MB| **ArcFace**| face → 512-d embedding             |

If the download is interrupted, just run the command again — it **resumes** a
partial download. When it finishes you should see the two files:

```bash
dir models            # Windows
ls -la models         # Linux / macOS
```

### Step 5 — Check everything is installed

```bash
python -c "import onnxruntime, cv2, numpy; print('ok', onnxruntime.__version__, cv2.__version__, numpy.__version__)"
```

### Step 6 — Add the people you want to recognise

Create one folder per person inside `data/faces/` and put 1–3 photos of that
person inside it. Square-ish, front-facing, well-lit photos with a clear face
work best.

```
data/faces/
├── alice/
│   ├── alice1.jpg
│   └── alice2.jpg
└── bob/
    └── bob.jpg
```

> Only image files directly inside each person folder are used
> (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`); sub-folders are ignored.
> The folder name becomes the person's name.

Concrete example with your own photos:

```bash
mkdir data\faces\jordan
copy C:\Users\you\Pictures\me1.jpg data\faces\jordan\
copy C:\Users\you\Pictures\me2.jpg data\faces\jordan\
```

### Step 7 — Enrol everyone

Enrolment runs every enrolment photo through the full pipeline **once**,
computes the 512-d embeddings, and saves them to
`data/embeddings/embeddings.npz` with the labels in `data/embeddings/meta.json`.

```bash
python -m scripts.enroll
```

Expected console output:

```
=== ENROLLMENT COMPLETE ===
  alice           2 embedding(s)
  bob             1 embedding(s)
  saved -> C:\...\face-recognition\data\embeddings\embeddings.npz
           C:\...\face-recognition\data\embeddings\meta.json
  identities: 2, total embeddings: 4
```

You can also enrol a single identity from explicit image paths:

```bash
python -m scripts.enroll --identity jordan --images photo1.jpg photo2.jpg
```

### Step 8 — Recognise a single photo

```bash
python -m scripts.recognize_image --image path/to/photo.jpg
```

Expected output for a known person:

```
Known: alice
Similarity: 0.9354
  bbox=[1607, 858, 3057, 3117] det_conf=0.849
```

And for a stranger:

```
Unknown
Similarity: 0.1234
  bbox=[407, 421, 994, 1193] det_conf=0.883
```

Useful options for this tool:

```bash
# save an annotated copy of the image (green box = known, red box = unknown)
python -m scripts.recognize_image --image photo.jpg --save outputs/annotated.jpg

# pop a window showing the annotated result
python -m scripts.recognize_image --image photo.jpg --show

# machine-readable output (JSON)
python -m scripts.recognize_image --image photo.jpg --json

# test a different threshold for this run only
python -m scripts.recognize_image --image photo.jpg --threshold 0.5
```

### Step 9 — Recognise live from your webcam

```bash
python -m scripts.recognize
```

A window opens showing your webcam feed. Every face gets:

- a **green** box + the person's name when recognised
  (`Known: alice`);
- a **red** box + `Unknown` when not enrolled.

Press **q** or **ESC** to quit. Options:

```bash
python -m scripts.recognize --camera 1                    # force a specific camera index
python -m scripts.recognize --threshold 0.5
python -m scripts.recognize --skip 5 --det-size 480       # faster on a low-end CPU
```

**Why the picture is fast but the labels "refresh" slowly.** Recognising a
face needs a full run of SCRFD + ArcFace, which on a normal laptop CPU takes
roughly **0.4–0.6 s** (already optimised — see `FR_ONNX_THREADS` in §6). If we
did that on every frame, the whole video would drop to ~2 FPS and look frozen.
So by default the tool **re-runs recognition only every 3rd frame** and keeps
the previous boxes/labels on screen in between: the video stays smooth, and the
name just updates a couple of times per second. Controls:

- `--skip N` — run recognition once every N frames (`1` = every frame,
  `N` = much smoother video, slower label updates);
- `--det-size W` — shrink the image fed to the detector (e.g. `480` or
  `416`); slightly less accurate on far/ small faces but noticeably faster.

> On many Windows laptops the built-in camera is index 0 and an external USB
> webcam is index 1. The tool **automatically picks the first camera that
> returns real content**, so on a laptop + USB-webcam setup it will normally
> choose the webcam for you.

### Step 10 — Demonstrate every stage to your teacher

```bash
python -m scripts.visualize --image path/to/photo.jpg
```

This is the "explain it" tool. It opens three windows, one after another,
showing exactly what the algorithm does internally:

1. **Detection** — the face box and the **5 landmark points** that SCRFD found
   (left eye, right eye, nose, mouth left, mouth right), colour-coded.
2. **Alignment** — the canonical **112×112 face crop** (upscaled) with the
   reference landmarks drawn on it — i.e. the exact image ArcFace embeds.
3. **Result** — the recognised identity plus a **similarity bar** coloured
   green/red by the threshold.

Press any key to advance to the next window and **q** to quit early. This is
perfect for a live classroom demo because every stage is visible.

---

## 6. Configuration reference

All settings live in `app/config.py` and every one of them can be overridden
with an environment variable — ideal for quick experiments without touching
the code.

| setting                          | variable                | default     | what it controls                       |
|----------------------------------|-------------------------|-------------|----------------------------------------|
| detector input size              | `FR_DETECTOR_INPUT_SIZE`| `640,640`   | resolution fed to SCRFD                |
| detector minimum confidence      | `FR_DETECTOR_CONFIDENCE`| `0.5`       | ignore faces below this score          |
| detector NMS IoU cutoff          | `FR_DETECTOR_NMS`       | `0.4`       | overlap threshold for duplicate boxes  |
| embedding strategy               | `FR_STRATEGY`           | `all`       | `all` = keep every embedding, `mean` = one averaged embedding per person |
| matching threshold               | `FR_THRESHOLD`          | `0.40`      | Known/Unknown boundary (see §8) |
| ONNX threads (speed)             | `FR_ONNX_THREADS`       | `2`         | intra-op threads for the ONNX models; `2` is fastest on most laptops (default all-cores measured ~6× slower), raise it on beefy desktops |

Example:

```bash
FR_THRESHOLD=0.5 python -m scripts.recognize_image --image photo.jpg
```

---

## 7. CLI tools reference

### `python -m scripts.download_models`

| flag      | default          | meaning                                  |
|-----------|------------------|------------------------------------------|
| `--model-dir` | `models/`    | where to download and extract            |
| `--url`       | InsightFace v0.7 `buffalo_l.zip` | download source          |

### `python -m scripts.enroll`

| flag               | default                 | meaning                                      |
|--------------------|-------------------------|----------------------------------------------|
| `--faces-dir`      | `data/faces/`           | directory holding one folder per person      |
| `--strategy`       | `all`                   | `all` or `mean` (see §5 Step 7)              |
| `--identity NAME`  | *(none)*                | enrol only this identity                     |
| `--images F ...`   | *(none)*                | photo paths used with `--identity`           |
| `--threshold-det`  | `0.5`                   | minimum face-detection confidence            |

### `python -m scripts.recognize_image`

| flag            | default | meaning                                          |
|-----------------|---------|--------------------------------------------------|
| `--image PATH`  | *(required)* | photo to analyse                            |
| `--threshold F` | `0.40`  | Known/Unknown boundary (matches config default)  |
| `--save PATH`   | *(none)*| write an annotated copy                         |
| `--json`        | off     | print results as JSON                            |
| `--show`        | off     | pop up a window with the annotated result        |

### `python -m scripts.recognize`

| flag            | default | meaning                                          |
|-----------------|---------|--------------------------------------------------|
| `--camera N`    | `0`     | preferred webcam device index                    |
| `--threshold F` | `0.40`  | Known/Unknown boundary                           |
| `--skip N`      | `3`     | run recognition once every N frames (video plays smoothly, labels update at ~1 frame in N) |
| `--det-size W`  | `640`   | square resolution fed to the SCRFD detector (smaller = faster) |

### `python -m scripts.visualize`

| flag            | default | meaning                                          |
|-----------------|---------|--------------------------------------------------|
| `--image PATH`  | *(required)* | photo to walk through                       |
| `--threshold F` | `0.40`  | threshold used in the result window              |

---

## 8. Understanding the matching threshold

Two faces are compared with **cosine similarity**:

- identical faces → `≈ 1.0`
- the same person, different photos → typically `0.5 – 0.9`
- different people → typically `< 0.3`

The system uses a single number to decide between *"...it's Alice"* and
*"...new person"*:

```
best similarity  ≥  threshold   →   Known (green)
best similarity  <   threshold   →   Unknown (red)
```

The default is **0.40**. There is **no universal correct value** — it depends
on your enrolled people, the cameras and the lighting, so real systems tune it
on their own data:

| threshold | symptom                                                        |
|-----------|----------------------------------------------------------------|
| too low   | strangers get labelled with someone's name (false accepts)     |
| too high  | your own enrolled people get labelled Unknown (false rejects)  |

How to tune it for your data:

```bash
# compare your enrolled identity against a photo of the SAME person
python -m scripts.recognize_image --image me_selfie.jpg --threshold 0.5

# compare against someone NOT enrolled
python -m scripts.recognize_image --image coworker.jpg --threshold 0.5
```

Pick a threshold between the two scores you observe.

---

## 9. Sample output and test results

The project was validated on public-domain portraits (Einstein, a footballer,
Marie Curie — see `outputs/testdata/`). The enrolment database contained
`einstein` (2 photos) and `footballer` (1 photo). Each row is a **new photo
that was NOT used for enrolment**:

| input photo               | best match    | similarity | decision   |
|---------------------------|---------------|------------|------------|
| Einstein, third portrait  | `einstein`    | 0.96       | Known ✔    |
| same footballer, photo 2  | `footballer`  | 0.68       | Known ✔    |
| Marie Curie portrait      | einstein / footballer | ~0.04 | Unknown ✔ |

This is exactly the behaviour wanted:

- a new photo of an enrolled person scores high (`0.96` — very confident);
- another photo of the same footballer scores clearly above threshold
  (`0.68`) even though his pose/lighting differs;
- a completely unseen face scores almost zero (`≈0.04`) and is correctly
  rejected.

The pipeline's decisions are therefore **repeatable and explainable**, which
makes them easy to present in a report or live demo.

---

## 10. Running the unit tests

The test-suite covers the pure math/logic parts so no models or webcam are
needed:

```bash
python -m pytest tests -q
```

Result:

```
25 passed in 8.56s
```

What the 25 tests actually verify (see `tests/`):

- **Alignment** (5 tests) — the warp produces a 112×112 image; the identity
  transform is used when the landmarks already match the reference; realistic
  landmarks land *exactly* on the ArcFace reference template; wrong or
  degenerate landmark inputs are rejected;
- **Embedding** (6 tests) — correct tensor shape after preprocessing,
  deterministic preprocessing, output shape (512,) and dimension, L2-unit-norm
  of the output, and that identical inputs give the same embedding while
  different inputs give different ones;
- **Matching** (14 tests) — L2-normalisation (unit norm, direction preserved,
  zero-vector edge case), cosine similarity (identical/orthogonal/opposite
  vectors, = the normalized dot product, "more similar &gt; less similar"),
  and the threshold decision rule (above → Known, exactly at the boundary →
  Known, below → Unknown, empty database → error, threshold is configurable).

---

## 11. Troubleshooting

| problem                                                        | fix |
|----------------------------------------------------------------|-----|
| `pip` says the requirements cannot be found                     | activate the virtual environment first (`Step 2`) |
| `import onnxruntime` fails                                     | `pip install -r requirements.txt` inside the venv |
| download hangs / interrupted                                   | run `python -m scripts.download_models` again — it resumes partial downloads |
| `enrollment database is empty; run python -m scripts.enroll`  | you ran recognition before enrolling; add photos under `data/faces/` and run `Step 7` |
| "could not find a working webcam"                              | close other apps using the camera; unplug/replug a USB webcam; grant camera permission (Windows Settings → Privacy & security → Camera); try `--camera 1` |
| webcam works but everything is `Unknown`                       | add more/better enrolment photos, then lower the threshold a little (e.g. `--threshold 0.3`) |
| everyone is recognised as everyone else (all `Known`, wrong)   | raise the threshold, e.g. `FR_THRESHOLD=0.6 ...` |
| "No face detected in the image."                               | choose clearer, front-facing photos; check the person is actually in the frame and the face is large enough |
| an enrolment photo fails                                       | use photos with exactly ONE clear face | 
| `ENROLLMENT FAILED: ... multiple faces` (or similar)           | each enrolment image must contain exactly one face |

---

## 12. Ideas to extend the project

If you want to go further than the practical asks:

1. **Threshold search** — sweep many thresholds and plot false-accept /
   false-reject rates (an EER curve).
2. **Anti-spoofing** — blink detection or liveness check before matching.
3. **Efficiency** — track each face across frames so you embed once per
   person per appearance, making the webcam tool much faster.
4. **Representatives** — implement the `mean` strategy and compare its
   accuracy against `all`.
5. **Visualisation** — project the 512-d embeddings into 2D with t-SNE/PCA to
   *see* that the same person clusters together.
6. **Face clustering** — add an unsupervised mode that groups unlabelled
   photos into identities.
7. **Landmark drawing** — extend `scripts/visualize.py` to draw eyebrows,
   a face mesh or the face image before/after alignment side-by-side.

---

## 13. Privacy

- Enrolment photos live only under `data/faces/` and computed embeddings only
  under `data/embeddings/` — **both are ignored by git** (see `.gitignore`),
  so they are never pushed to GitHub.
- All processing happens **locally on your machine**. Nothing is uploaded to
  the internet.
- Used responsibly, face recognition is a great teaching tool; used against
  people without consent it can be harmful. This project is intended purely
  for education and your own data.

---

## 14. Licence and model credits

- The two ONNX models are the official **InsightFace** models from the
  `buffalo_l` package (v0.7), distributed by the InsightFace project
  (MIT licence): <https://github.com/deepinsight/insightface>.
- Example test faces are public-domain portraits from Wikimedia Commons and
  the ONNX Model Zoo sample images, used only for evaluation.
- This project's own code is MIT licensed (see `LICENSE`).