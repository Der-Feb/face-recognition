# face-recognition

A real, runnable face-recognition system built for a Computer Science practical.
It finds faces, aligns them, turns them into **numeric embeddings** with the
**ArcFace** deep model (running locally in the ONNX format — no cloud, no API,
no internet needed) and then **recognises people** by comparing their embeddings
to a small local database.

Everything runs on CPU with a plain Python environment. Every stage of the
pipeline is explicit and inspectable — nothing is hidden inside a black box —
which makes it easy to explain and to extend.

---

## What the system does

The recognition pipeline is a chain of five well-known computer-vision steps:

```
                 ┌──────────────────────────────────────────────────────────┐
                 │                     INPUT IMAGE                          │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  1. DETECTION   (SCRFD ONNX model, 640×640)             │
                 │     finds every face + bounding box                     │
                 │     + 5 landmark points (eyes, nose, mouth corners)     │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  2. ALIGNMENT   (similarity transform, Geometry)        │
                 │     rotates/scale the face into a canonical             │
                 │     112×112 pose using the 5 landmarks                  │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  3. EMBEDDING   (ArcFace ResNet-100 ONNX, 112×112)      │
                 │     the aligned face becomes a 512-dimensional vector    │
                 │     (an "embedding") representing "this face"           │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  4. ENROLLMENT  (optional, run once per person)          │
                 │     one-or-more photos per person → a database of        │
                 │     labelled embeddings (embeddings.npz + meta.json)     │
                 └──────────────────────────────────────────────────────────┘
                                        │
                                        ▼
                 ┌──────────────────────────────────────────────────────────┐
                 │  5. MATCHING    (cosine similarity, 0.4 threshold)      │
                 │     how close is the new face to each enrolled person?   │
                 │     best match ≥ threshold  →  "Known: <name>"           │
                 │     best match <  threshold  →  "Unknown"                │
                 └──────────────────────────────────────────────────────────┘
```

That is the whole idea: a face becomes a *point in a 512-dimension space*, and
"recognising" someone means finding the enrolled point that is closest to the
new face.

---

## What is inside the repo

```
face-recognition/
├── app/
│   ├── config.py          # paths + tunable settings (threshold, model sizes…)
│   ├── utils.py           # cosine similarity, L2 norm, NMS, image I/O
│   ├── detector.py        # stage 1: SCRFD face detection + landmarks
│   ├── aligner.py         # stage 2: 5-point alignment to 112×112
│   ├── embedder.py        # stage 3: ArcFace ONNX → 512-d embedding
│   ├── enrollment.py      # stage 4: build/save/load the embeddings database
│   ├── matcher.py         # stage 5: cosine similarity + threshold decision
│   └── recognition.py     # pipeline that wires stages 1→5 together
├── scripts/
│   ├── download_models.py # downloads the two ONNX models
│   ├── enroll.py          # CLI: enroll all photos in data/faces/
│   ├── recognize.py       # CLI: live webcam recognition
│   ├── recognize_image.py # CLI: recognise faces in one photo
│   └── visualize.py       # CLI: pop-up windows for every stage (demo tool)
├── tests/                 # 25 passing unit tests
├── data/
│   ├── faces/<person>/…   # YOUR enrolment photos (one folder per person)
│   └── embeddings/…       # generated database (embeddings.npz, meta.json)
├── models/                # ONNX models (downloaded, not committed)
├── requirements.txt
└── README.md
```

---

## 1. Setup (5 minutes)

Requires **Python 3.9+** on Windows / Linux / macOS.

```bash
# 1. create a virtual environment and activate it
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux / macOS

# 2. install the dependencies (CPU-only, pure pip)
pip install -r requirements.txt

# 3. download the two ONNX models (~180 MB) into models/
python -m scripts.download_models
```

The downloader pulls the official InsightFace v0.7 models and extracts exactly
two files into `models/`:

| file               | size   | purpose                                |
|--------------------|--------|----------------------------------------|
| `det_10g.onnx`     | ~16 MB | SCRFD face detector (boxes + landmarks)|
| `w600k_r50.onnx`   | ~166 MB| ArcFace recognition model (embeddings) |

> The large model files and your face photos/embeddings are **ignored by git**
> (see `.gitignore`) — the repository stays small and private.

---

## 2. Add the people you want to recognise

Create one folder per person under `data/faces/`, and put 1–3 photos of each
person inside it. Square-ish, front-facing photos with a clear face work best.

```
data/faces/
├── alice/
│   ├── alice1.jpg
│   └── alice2.jpg
└── bob/
    └── bob.jpg
```

> Only photos are fetched from `data/faces/` — no manual metadata. Sub-folders
> are **not** scanned.

---

## 3. Enroll everyone

Enrollment runs every face through the whole pipeline **once**, stores the
resulting 512-d embeddings in `data/embeddings/embeddings.npz`, and labels them
via `data/embeddings/meta.json`.

```bash
python -m scripts.enroll
```

Sample output:

```
=== ENROLLMENT COMPLETE ===
  alice         2 embedding(s)
  bob           1 embedding(s)
  saved -> data\embeddings\embeddings.npz
           data\embeddings\meta.json
  identities: 2, total embeddings: 4
```

If a person appears in several photos each of those photos contributes its own
embedding (multi-embedding strategy `all`), which makes matching more reliable.

---

## 4. Recognise

### 4a. A single photo (great for checking precision)

```bash
python -m scripts.recognize_image --image path/to/photo.jpg
```

```
Known: alice
Similarity: 0.9354
  bbox=[1607, 858, 3057, 3117] det_conf=0.849
Annotated image saved to: outputs/annotated_photo.jpg
```

`--save FILE` writes an annotated copy, `--show` **pops a window** on the screen,
`--json` prints machine-readable output, and `--threshold F` overrides the
decision threshold for that run.

### 4b. Live webcam

```bash
python -m scripts.recognize
```

A window opens with your webcam feed. Each detected face gets a green box +
name when recognised (`Known: <name>`) or a red box (`Unknown`) otherwise.
Press **q** or **ESC** to exit. The tool automatically uses the first camera
that produces real content — handy on Windows, where an external USB webcam is
often index 1 instead of 0.

### 4c. Tour of every stage (the "explain it to the teacher" tool)

```bash
python -m scripts.visualize --image path/to/photo.jpg
```

Three windows pop up one after another and show exactly what the algorithm does:

1. **Detection** — face box + the 5 landmark points (left/right eye, nose,
   left/right mouth corner) that SCRFD found;
2. **Alignment** — the canonical 112×112 face crop that ArcFace will embed;
3. **Result** — label + a similarity bar, coloured green/red by the threshold.

Press any key to advance and **q** to quit early.

---

## 5. Understanding the matching threshold

Two faces are compared with **cosine similarity**: identical faces → `1.0`,
completely different → near `0.0`. The decision rule is:

```
best similarity ≥ threshold  →  Known (green)
best similarity <  threshold  →  Unknown (red)
```

The default threshold is **0.40**. There is no universal "right" value — you
tune it for your data:

| Threshold | behaviour                                   |
|-----------|---------------------------------------------|
| too low   | strangers get recognised as someone you know (false accepts) |
| too high  | your own people get rejected (false rejects) |

It is configurable in `app/config.py` (`MATCHING_THRESHOLD`), via the
`FR_THRESHOLD` environment variable, or per command with `--threshold F`.

---

## 6. Reproducible test results

Verified on public-domain faces (see `outputs/testdata/`), using two
enrolled identities (`einstein` with 2 photos, `footballer` with 1 photo):

| input photo          | best match          | similarity | decision |
|----------------------|---------------------|------------|----------|
| einstein photo 3     | einstein            | 0.96       | Known ✔  |
| footballer photo 2   | footballer          | 0.68       | Known ✔  |
| Marie Curie portrait | einstein / footballer | ~0.04     | Unknown ✔|

A high score for a new photo of an enrolled person, a decent score for
another photo of the same footballer, and near-zero for a never-seen face —
exactly what the pipeline should do.

---

## 7. Tests

```bash
python -m pytest tests -q
```

```
25 passed in 8.56s
```

The suite covers cosine similarity / L2-normalisation edge cases, NMS
behaviour, the 5-point alignment geometry, embedding normalisation, the
threshold acceptance rules and the enrollment↔matching round-trip.

---

## 8. Troubleshooting

| problem                                            | fix |
|----------------------------------------------------|-----|
| `import onnxruntime` fails                         | active the venv, `pip install -r requirements.txt` |
| `could not find a working webcam`                 | close other apps using the camera; unplug/replug the USB webcam; cameras need permission in Windows Settings → Privacy → Camera |
| `enrollment database is empty; run python -m scripts.enroll` | run enroll after adding photos to `data/faces/` |
| everything comes back `Unknown`                    | raise confidence first: add more/better photos to enrolment, then lower `--threshold` |
| model download is slow / interrupted            | re-run `python -m scripts.download_models` — it resumes a partial download |

---

## 9. Privacy

Face photos and computed embeddings can identify people, so:

- uploaded photos live only in `data/faces/` and are **ignored by git**;
- embeddings are also ignored by git;
- everything runs locally — nothing is sent to a server.

---

## 10. Model licence

The two ONNX models are the official **InsightFace** models (`buffalo_l` pack,
v0.7) distributed under the MIT licence; source:
<https://github.com/deepinsight/insightface>. This project itself is MIT (see
`LICENSE`).