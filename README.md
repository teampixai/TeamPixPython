# TeamPixPython

Detects blurry sport photos for photographers who unethically abused the burst mode.

Runs two YOLO models (body pose + face detection) on a folder of images, filters subjects down to your team by jersey colour, scores the sharpness of each person's face, and produces annotated previews so you can review the results before any files are moved.

---

## Getting started (Windows, no git/download-zip needed)

Nothing needs to be installed first — not Anaconda, Python or git. The setup script downloads its own private copy of [Miniconda](https://docs.conda.io/en/latest/miniconda.html) into the app folder, so the app keeps working no matter what other Anaconda/Miniconda/Python you install, change or remove on the PC, and it doesn't affect them either. Open a plain Command Prompt and paste these two commands:

1. Create a working folder and switch into it:

   ```
   mkdir TeamPixPython && cd TeamPixPython
   ```

2. Download and run the setup script:

   ```
   curl -L -o InitPhotoProcessing.bat https://raw.githubusercontent.com/teampixai/TeamPixPython/main/InitPhotoProcessing.bat && InitPhotoProcessing.bat
   ```

   (`curl` is built into Windows 10/11 by default.) On the first run you're asked to accept Anaconda's [Terms of Service](https://www.anaconda.com/legal) before Miniconda is downloaded. Add `/y` after `InitPhotoProcessing.bat` to accept them without the prompt. The script then installs everything into this folder:

   | Folder / file | Contents |
   |---|---|
   | `miniconda3\` | Private Miniconda (not added to PATH or the registry) |
   | `envs\teampixpython\` | The app's Python and all its dependencies |
   | `app\` | The app source |
   | `RunPhotoProcessing.bat` | The launcher |

   It also adds a **TeamPixPython** entry to the Start Menu. Use it, or double-click `RunPhotoProcessing.bat`, to open the web app — see below. Re-run the same command above any time to update everything to the latest version. A path without spaces is recommended. Windows on ARM64 is supported through the x64 build of Miniconda, which runs under emulation.

   To uninstall, delete the folder and the `TeamPixPython` folder in the Start Menu (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\TeamPixPython`).

---

## Quick start: the web app (`culling_app.py`)

The easiest way to use this tool day-to-day is the built-in web app — it wraps the whole pipeline (import, review, face tagging, export) behind a browser UI, so you rarely need the individual scripts described below.

```
python culling_app.py
```

This starts a local web server and opens a browser tab automatically (use `--no-browser` to skip that, `--port <n>` to change the port). If you set the project up with `InitPhotoProcessing.bat`, just open **TeamPixPython** from the Start Menu (or double-click the generated `RunPhotoProcessing.bat`) instead.

The app walks through five pages (top nav):

| Page | Purpose |
|---|---|
| **Team** | Create/edit a team: jersey colour(s), roster, optional OpenAI API key for LLM-assisted burst culling. You need at least one team before importing photos. None is created for you: on a first run the app opens this page (Select Album and Add Album redirect here, and Select Album stays disabled in the top nav) and you click **Add new team** to create your own. |
| **Select Album** | Pick an existing album, or click **+** to import a new folder of photos: choose the team, blur sensitivity, whether to run face recognition, and optionally **Ignore jersey color** to score every person regardless of team colour (handy for practices/non-game shoots). This kicks off `1_prep_review.py` in the background. |
| **Culling** | Review the scored photos (blurry/sharp/skipped), override any calls, and star-rate keepers. |
| **Faces** | Tag detected face clusters with player names. |
| **Summary** | Export the kept, tagged photos to a destination folder. |

**AI edit** (Culling page, next to the file path; Faces page, in the photo preview) opens an editor: the photo with a tabbed panel on the right (**Cancel**/**Apply** at the bottom apply to every tab).
- **AI Edit** — a before/after split view with **Brightness** (Highlights / Midtones / Shadows, −2…+2 EV), **Contrast** (−100…+100) — each with an **Auto** checkbox that fills in suggested values and locks the sliders — and **Sharpen players**, which draws a box around every detected player (teammates start selected; click a box to toggle it, the smallest box wins where boxes overlap). The preview is rendered by the same code as the final image (`apply_tone()` in `autoedit.py`; sharpening isn't previewed).
- **Crop** — ratio presets, all **H x W** and listed widest first: Instagram (5x4, 1x1), Phone (iPhone / Android 19.5x9, iPhone Fold lock screen 2034x1398, iPhone Fold home screen 2670x1878) and Print (9x16 … 16x9), plus **Custom size**, which shows the box's top-left corner and H × W in pixels. The photo shows a crop box you drag to move or resize from its edges and corners — presets keep their ratio, Custom is free; picking a preset keeps the box's center and size as far as the new ratio allows. Scroll to zoom; drag outside the box to pan. The crop starts as Custom covering the whole photo (no crop) and always applies: whatever it cuts off is darkened 75% on every tab.
- **Other** — **Reload** re-reads the photo from disk (e.g. after editing it elsewhere) and keeps your settings; **Revert…** asks for confirmation, then deletes the edited copy so the photo is its original again; **Edit with custom app…** opens the edited copy (created from the original if there is none yet) with an app you pick in Windows' "Open with" dialog — settings not yet applied in the editor aren't included.

**Apply** renders the full-resolution result via `autoedit.py -pipelines tone [aisharpen] [--crop X,Y,W,H]` (the crop runs last), then the review offers **Abort** (discard the result), **Save Copy** and **Save** (keep it as this photo's edit). **Save Copy** leaves the photo as it was and instead moves the result to `<album>/SaveCopy/<name>_<N>.jpg` (`N` = one past the highest number already used for that name) and adds it to the album as a new photo: a copy of the original's `album.json` entry with every detection re-framed onto the crop (bodies the crop removed are dropped, none left → "skipped"), listed in `info.json` next to the original, and dated with the original's capture time so it sits in the same burst. Where an accepted edit sits in the original is stored on the photo's `album.json` entry as `edited_crop` (normalised box, used to map detected players onto later edits) and `edited_crop_px` (`{"x", "y", "h", "w"}` in pixels of the original); both are removed by **Revert**. The Culling page's annotated preview (always drawn from the original) darkens everything outside a recorded crop by 80%, to show the photo was cropped. The three checkboxes are remembered (in a browser cookie) only when you accept. Pages launch the editor with `AiEdit.run(key, { onResult })`; `onResult({ file, outcome, changed, copy?, error? })` is called once when the run ends, `outcome` being `accepted` (Save), `rejected` (Abort), `copied` (Save Copy; `copy` is `{ key, path }` of the new photo), `cancelled` or `failed` and `changed` telling the page the photo's file changed (accepted, reverted, or opened in another app) so it should reload it — the Faces page's photo preview does this automatically (`Viewport.showImageWindow(..., { onAiEditResult })` also forwards the result). That preview, like the Culling page's "original" view, always shows the accepted edit when there is one; the burst and per-player MP4 renders on Album Tools always use the original photos. Debug: set `"save_silhouette": true` in `.culling_state.json` and each Apply that sharpens players also writes four full-resolution images to `<album>/debug/esrgan_silhouette/`: `<name>_orig.jpg` (the photo as read), `<name>_silhouette.png` (the blend mask; white = sharpened), `<name>_sharpened.jpg` (the sharpened layer: the photo after your brightness/contrast changes, with the sharpened crops pasted in unmasked) and `<name>_final.jpg` (the result). JPEGs are saved at quality 100. The same is available on the command line as `autoedit.py --save-silhouette DIR`.

---

## How it works

```
Image → normalize size (long edge = 1800 px)
      → Pose model      → body bounding boxes + 17 COCO keypoints
      → Face model       → face bounding boxes + 5 landmarks
      → match faces to bodies (head-keypoint overlap)
      → grade sharpness of each person's face crop
      → poll jersey colours → keep only your team (+ forced colours)
      → write the album's Album data (see algo/album.py) / info.json / blurry.csv / blur.lst
      → (optional) cluster faces into .FaceReco/
      → (later, in the web app) render annotated previews on demand
```

Detection runs on a pluggable **engine** (`--engine mediapipe` default, or `--engine yolo`):

| Engine | Pose | Face | License |
|---|---|---|---|
| `mediapipe` *(default)* | BlazePose (Pose Landmarker) | BlazeFace + FaceMesh-V2 (Face Landmarker) | Apache-2.0 (permissive) |
| `yolo` | YOLOv8n-pose | YOLOv8n-face | AGPL-3.0 / GPL-3.0 (copyleft — see licensing note below) |

The pipeline is a sequence of stages (`algo/stages/`):

| Stage | Responsibility |
|---|---|
| `ImageAnalysisStage` | Run both YOLO models, match faces to bodies |
| `GradingStage` | Compute the sharpness score and pass/fail per body |
| `JerseyCountingStage` | Detect the dominant team colour and disqualify off-team bodies |
| `FaceRecoStage` | Cluster qualifying faces into `.FaceReco/` (optional) |

Annotated previews are not a pipeline stage: `algo/stages/annotation.py` draws them on demand (see "Annotated previews" below).

### Sharpness scoring

Sharpness is computed by a pluggable `SharpnessEvaluator`. The active implementation is `GeometricMeanEvaluator`, which uses two classical metrics:

| Metric | What it measures |
|---|---|
| Laplacian variance | Fine detail / high-frequency content |
| Tenengrad | Gradient energy — robust to noise |

Each metric is normalised to a sharpness component in [0 – 1], then combined as a **geometric mean**:

```
score = sqrt(lap_sharp × ten_sharp)
```

The geometric mean only scores high when *both* metrics agree, which avoids false "sharp" results caused by noise, compression artefacts, or over-sharpening that inflate one metric while the other stays low. The raw values are still available in the CSV output. To switch back to the previous weighted-average formula, change the active evaluator in `algo/sharpness.py` to `LaplacianTenengradEvaluator`.

### Team / jersey filtering

`JerseyCountingStage` measures each body's jersey colour in CIE L\*a\*b\*, polls the dominant colour across all sharp bodies in a frame to identify "your" team, and disqualifies bodies wearing a different colour so the tool only scores and annotates your players. Matching is brightness-forgiving (a jersey in shadow still matches the same jersey in sunlight). Pass `--jerseycolor` to bias the team colour, `+`-prefix a colour to force-include it (e.g. goalie kit), or `--noteam` to disable filtering entirely.

---

## Workflow

The tool is split into steps so you can review and override the automatic classifications before any files are moved.

### Step 1 — Score

```
python 1_prep_review.py <image_or_directory> [options]
```

| Option | Default | Description |
|---|---|---|
| `--sensitivity low\|medium\|high\|<n>` | `medium` | Blur threshold; or pass a numeric value (0–1) directly |
| `--engine mediapipe\|yolo` | `mediapipe` | Detection/pose/face-landmark engine. `mediapipe` is Apache-2.0 licensed (default); `yolo` is the legacy engine, kept for comparison/rollback (AGPL-3.0/GPL-3.0 — see Licensing below) |
| `--jerseycolor "COLOR[;COLOR...]"` | `blue;white;+purple;+orange;+light blue;+pink` | Team colours; `+` = forced-include; empty string disables filtering |
| `--noteam` | off | Disable jersey filtering (score every detected person) |
| `--output <dir>` | `albums/<timestamp>-<input>/` | Root output directory |
| `--skip-facereco` | off | Don't run face-recognition clustering |
| `--face-db <dir>` | none | Match clusters against an existing face DB root (uses its `<team id>/` folder) or a team folder directly |
| `--face-db-match-threshold <n>` | `0.72` | Cosine similarity required between a face and its closest-matching person prototype |
| `--face-db-match-margin <n>` | `0.05` | Minimum similarity gap required over the best-matching different person before accepting a match |
| `--face-db-prototype-threshold <n>` | `0.62` | Cosine similarity used to split each DB person's own photos into visual sub-clusters |
| `--min-face-crop-px <n>` | `32` | Minimum short-edge crop size (pixels) trusted for matching/clustering |
| `--align-faces` | off | Similarity-align faces to a 5-point template before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |

Scores every image and writes an output folder:

```
<output_dir>/
    previews/         ← cache of annotated previews, rendered on demand by the web app (empty after an import); files named <key>.jpg
    thumbnails/       ← small square thumbnails of each photo for the culling nav strip, written during import; files named <key>.jpg
    album.json      ← the Album's data (see algo/album.py; full per-body data used by face recognition); each result also carries a "preview_path" naming where its preview is cached
    info.json         ← classification results (used by step 2)
    blurry.csv        ← one row per blurry image with score details
    blur.lst          ← plain list of blurry file paths
    run.log           ← full debug log
    .FaceReco/        ← face recognition clusters (unless --skip-facereco)
```

**Face Recognition (optional):**

By default, after the photos are scored, face recognition automatically clusters qualifying faces using the Facenet provider. The output is stored in `.FaceReco/` within the same output directory. Disable the auto-run with `--skip-facereco` if needed.

**Sensitivity thresholds** (score ≤ threshold → blurry):

| Level | Threshold | Use when |
|---|---|---|
| `low` | 0.16 | Flag only severely blurry images |
| `medium` *(default)* | 0.28 | Balanced |
| `high` | 0.42 | Flag even slightly soft images |

You can also pass any numeric value directly, e.g. `--sensitivity 0.45`.

**Annotated previews:**

Previews are rendered **on demand** by the web app, the first time a photo is shown, from the album's stored detections (`album.json`) and the current face-recognition result, and then cached in `previews/`. An import no longer pre-renders them, so it finishes sooner and a big album only pays for the photos you actually look at. Deleting a cached file is harmless: it is simply re-rendered the next time it is needed. The app also drops a photo's cached preview itself whenever something it shows changes: when you assign or reject a face on the Faces page, after a regrade changes a person's verdict, and after a face-recognition re-run. All previews are drawn by the same code (`algo/stages/annotation.py`), so they look alike however they were produced. A preview needs the original photo to still be on disk.

The culling page's nav-strip thumbnails are different: they show the plain photo (no boxes or scores) and are written once during import, from the image the analysis step has already decoded, so the nav strip never has to render previews. Nothing that changes a preview can make them stale. If a thumbnail is ever missing (deleted, or its write failed), the web app generates it on demand straight from the original photo. Albums imported earlier keep using their existing thumbnails.

Previews and thumbnails are named by the photo's full key (e.g. `IMG_0001.CR3.jpg`), so a RAW+JPEG pair of the same shot (`IMG_0001.JPG` + `IMG_0001.CR3`) gets separate files.

Legend:

- **Blue box / ✓** — person scored sharp
- **Red box / ✗** — person scored blurry
- **Green box** — narrow face crop used for scoring (minimal bbox around the 5 face landmarks)
- **Yellow box** — full face-detection bbox
- **Circles** — face landmarks (blue = confident, red = low confidence)
- **Score label** — sharpness score printed below the face box
- **Name** — next to the score: the person's name when face recognition has one (matched to your face database or assigned by you); otherwise **teammate** if their jersey matched your team colour; otherwise nothing
- All annotation is semi-transparent (`annotation_alpha = 0.25`); score text is opaque

### Step 2 — Review

Use the web app (`culling_app.py`) to review the scored photos (with their previews), override verdicts and star ratings, tag faces, and export the kept photos — see "Quick start" above.

---

## Face recognition

Face recognition runs automatically at the end of Step 1 (unless `--skip-facereco`).

| Option | Default | Description |
|---|---|---|
| `--cluster-threshold <n>` | `0.72` | Cosine similarity for clustering (higher = tighter clusters) |
| `--face-buffer-ratio <n>` | `0.15` | Extra crop padding around the face box (per side) |
| `--face-db <dir>` | none | Match faces against an existing face DB |
| `--face-db-match-threshold <n>` | `0.72` | Cosine similarity required between a face and its closest-matching person prototype |
| `--face-db-match-margin <n>` | `0.05` | Minimum similarity gap required over the best-matching different person before accepting a match |
| `--face-db-prototype-threshold <n>` | `0.62` | Cosine similarity used to split each DB person's own photos into visual sub-clusters ("prototypes") |
| `--min-face-crop-px <n>` | `32` | Minimum short-edge crop size (pixels) trusted for matching/clustering; smaller crops are skipped |
| `--align-faces` | off | Similarity-align faces before embedding |
| `--debug-align` | off | Write alignment QA images to `.FaceReco/.debug` |
| `--open-viewer` | off | Open the generated `.FaceReco/` folder afterwards |

**What it does:**

1. Loads per-body data from the Album (see algo/album.py).
2. Extracts face embeddings from qualifying bodies (via the selected provider), skipping crops smaller than `--min-face-crop-px`.
3. Optionally matches faces against a face DB and names matched clusters (see **How face-DB matching works** below).
4. Clusters the remaining (unmatched) faces by cosine similarity (agglomerative, average linkage).
5. Writes face crops and metadata to `<prep_output_dir>/.FaceReco/`.

### How face-DB matching works

A real person's reference photos rarely form one tight blob in embedding space - glasses on/off, lighting, angle, and expression all shift the embedding. Loading the face DB therefore splits **each person's own positive embeddings** into visually-cohesive **prototypes** (sub-clusters, via the same cosine agglomerative clustering used elsewhere) instead of treating a person as one blended average. A query face is scored against every person's single *closest* prototype.

A face is only assigned to a person when **all** of the following hold:

1. **Absolute floor** - the best prototype similarity meets `--face-db-match-threshold`.
2. **Margin over the runner-up** - the best-matching person must beat the best-matching *different* person by at least `--face-db-match-margin`. Two near-tied candidates is exactly the failure mode that mixes up similar-looking people; when the margin isn't met the face is left unmatched (and falls into ordinary clustering) rather than guessed.
3. **Not vetoed** - the face isn't at least as similar to that person's curated `Negative/` examples as it is to the match.

Each matched face's `matchScore`, `matchMargin`, and `matchRunnerUp` are written into `face.json` so a match can be audited after the fact.

Don't guess `--face-db-match-threshold` / `--face-db-match-margin` - calibrate them from your own face DB with `RebuildFaceDB.py` (below), which calibrates automatically every time it rebuilds.

Output layout:

```
<prep_output_dir>/.FaceReco/
    0000/
        Face/           ← qualified body faces
        Negative/       ← reserved for manually curated DB counter-examples
        face.json       ← editable metadata
    0001/
        Face/
        Negative/
        face.json
```

Each `face.json` stores editable person metadata (`name`, `playernum`) and a `faces` list with the original filename, original body JSON payload, crop file name, confidence, match diagnostics (`matchScore`/`matchMargin`/`matchRunnerUp`, set only for face-DB matches), an `aligned` flag, and a base64-encoded float32 embedding.

> **Note:** a *rebuilt* face DB (see below) uses a different, flatter `faces` schema — just a list of `{dtype, shape, encoding, value}` embeddings with no wrapper. That's the schema `--face-db` / `FaceDb.load` actually reads; the richer per-face metadata above is only present in the unlabeled `.FaceReco/` output of a fresh run, before `RebuildFaceDB.py` strips it down.

### Provider modules (`algo/`)

- `algo/facereco_provider.py` — abstract provider interface
- `algo/dlib_provider.py` — dlib implementation
- `algo/facenet_provider.py` — FaceNet (facenet-pytorch / VGGFace2) implementation
- `algo/face_crop_embed.py` — shared crop → detect → (optional align) → embed pipeline used by both prediction and DB rebuild

### Per-team face DB

The cross-album face DB at `./.FaceReco/` has one folder per team, named by the team's UUID (`team.json` → `id`; the team's display name can change freely):

```
.FaceReco/<team id>/<Player>/{face.json, Face/, Negative/}
```

A player on two teams is stored as two independent people, one per team folder. Matching during import, the Faces-page autocomplete, and the export's face-DB sync only touch the album's own team folder. Any name tagged on the Faces page (coaches, guests, …) is added to that team's folder at export, whether or not they're on the roster.

To move an old flat `.FaceReco/<Player>/` DB into this layout (copies each entry into every team whose roster has that name; names on no roster are left in place; a backup is made first):

```
python MigrateFaceDB.py --dry-run
python MigrateFaceDB.py
```

### Rebuilding (and calibrating) a face DB

```
python RebuildFaceDB.py <facedb_dir> [--align-faces] [--prototype-threshold 0.62] [--skip-calibration]
```

One command does both steps:

1. **Rebuild** — recomputes every embedding in a `.FaceReco/` directory from its saved crop images. Use `--align-faces` only if the database was originally built with alignment — the rebuild and prediction must use the same flag so embeddings live in the same space.
2. **Calibrate** (runs automatically afterwards; pass `--skip-calibration` to skip it) — runs leave-one-out cross-validation over the just-rebuilt DB: every photo is temporarily held out, scored against its own person's remaining photos (genuine similarity) and against every other person (impostor similarity) — exactly the comparison a real query faces. It reports:
   - A recommended `--face-db-match-threshold` (nearest the equal-error-rate point) and `--face-db-match-margin`.
   - **Confusions** — specific photos where a *different* person scored as high or higher than the true person. This is the most actionable signal for "faces are getting mixed up": it names the exact photo and the exact other person it's confused with, which is almost always either a genuine look-alike pair or a mislabeled crop sitting in the wrong person's `Face/` folder.
   - Pairwise prototype overlap — which people are closest to each other in embedding space overall.

Run it whenever the face DB changes meaningfully (new people, new photos) rather than reusing the default thresholds blindly — it only takes one command.

---


## Installation

### Requirements

- Python 3.10+
- PyTorch with CUDA (optional but strongly recommended for speed)

### Setup

```bash
# 1. Install PyTorch — CUDA 12.x example (torch 2.2.x is required by facenet-pytorch,
#    and cu121 is the newest CUDA 12 index that has it):
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Install remaining dependencies:
pip install -r requirements.txt
```

Models are downloaded automatically on first run:

- **mediapipe engine (default):** `pose_landmarker_full.task` and `face_landmarker.task`, downloaded from Google's
  [MediaPipe model zoo](https://storage.googleapis.com/mediapipe-models/) (Apache-2.0 licensed).
- **yolo engine (`--engine yolo`):** `yolov8n-pose.pt` via Ultralytics, and `yolov8n-face.pt` from the
  [akanametov/yolo-face](https://github.com/akanametov/yolo-face) release.
- **autoedit.py's body-only sharpening** (`algo/segmentation.py`): `selfie_segmenter.tflite`, also from the
  MediaPipe model zoo (Apache-2.0). Used to mask out just the detected players so Real-ESRGAN sharpening blends
  onto them alone, leaving the rest of the frame untouched — see `AI_SHARPEN_SCOPE` in `autoedit.py`.

### Licensing

**Permissive, no copyleft obligations** — everything used by the default `mediapipe` engine and default face
recognition path:

| Component | Used for | License |
|---|---|---|
| mediapipe (BlazePose / BlazeFace / FaceMesh-V2 / Image Segmenter "selfie" model) | default pose/face engine, autoedit.py's body-only sharpen mask | Apache-2.0 |
| torch / torchvision | hybrid engine's person-box detector (`fasterrcnn_resnet50_fpn_v2`) | BSD-3-Clause |
| facenet-pytorch | face embedding (default provider) | MIT |
| dlib | optional face embedding provider | Boost Software License 1.0 |
| face-recognition | wraps dlib's face models | MIT |
| numpy, scikit-learn | array ops, clustering | BSD-3-Clause |
| opencv-python | image I/O and processing | MIT (wrapper) + Apache-2.0 (OpenCV itself) |

**Weak copyleft (LGPL), used only as dynamically-linked binaries** — no source-disclosure obligation for this
project's own code, but keep in mind if you redistribute the binaries themselves:

- `rawpy` (MIT) wraps **LibRaw**, dual-licensed LGPL-2.1 or CDDL (or a paid commercial license). The PyPI wheel
  links it dynamically, which is the standard LGPL-compatible way to use it from a closed-source app.
- `opencv-python`'s prebuilt wheels bundle **FFmpeg** (LGPLv2.1); non-headless Linux wheels additionally bundle
  **Qt5** (LGPLv3). Not applicable to Windows/macOS wheels or `opencv-python-headless`.

**Copyleft — opt-in only, not the default:** the `--engine yolo` path uses Ultralytics YOLO models, which are
**AGPL-3.0** (or a paid Enterprise license) for the pose model, and **GPL-3.0** for the `akanametov/yolo-face`
model. These carry source-disclosure obligations for derivative/networked works — review them before enabling
`--engine yolo` in any distributed or hosted use of this tool.

All model weight files (`*.pt`, `*.task`) are `.gitignore`d and downloaded at runtime — none are committed to
or redistributed with this repository.

### Known limitation (mediapipe engine)

MediaPipe's Face Landmarker does not expose a reliable per-landmark occlusion/visibility score the way the YOLO
face model does, so `face_coverage_min_visible` (the check that disqualifies a face with too few visible
landmarks) is effectively a pass-through under `--engine mediapipe`.

---

## Key configuration (`AppConfig` in `algo/config.py`)

| Setting | Default | Description |
|---|---|---|
| `use_narrow_face_box` | `True` | Score and annotate on the minimal landmark bbox rather than the full face detection bbox |
| `face_min_size_fraction` | `0.025` | Discard faces whose long edge is < 2.5 % of the image long edge |
| `face_crop_min_long_edge_px` / `face_crop_max_long_edge_px` | `96` / `96` | Normalize every face crop's long edge to this pixel range (up *and* down) before sharpness scoring, removing scale-dependence from the Laplacian/Tenengrad metrics |
| `sharpness_contrast_normalize` | `True` | Rescale each face crop to a fixed contrast before sharpness scoring, so dark skin tones / underexposed subjects aren't penalised by the absolute-contrast metrics |
| `sharpness_contrast_reference` | `40.0` | Target standard deviation each crop is rescaled to |
| `sharpness_contrast_min_gain` / `sharpness_contrast_max_gain` | `0.5` / `4.0` | Clamp on that rescaling gain, guarding against noise amplification in near-flat crops |
| `face_detection_min_confidence` | `0.1` | MediaPipe Face Landmarker detection confidence. Low on purpose — it only ever sees an already-localised head crop |
| `face_detect_crop_min_long_edge_px` | `256` | Upscale head crops smaller than this before face detection |
| `face_keypoint_fallback` | `True` | Reconstruct a face from the pose model's nose/eye keypoints when the face detector finds nothing |
| `face_keypoint_fallback_conf` | `0.5` | Pose-keypoint confidence required to use that fallback |
| `face_coverage_min_visible` | `2` | Require this many of 5 face landmarks to be confident |
| `normalized_img_max_long_edge` | `1800` | Downscale input to this long edge before processing (never upscales) |
| `annotation_alpha` | `0.25` | Annotation translucency (0 = invisible, 1 = opaque) |
| `annotation_top_n_bodies` | `5` | Always annotate the N largest bodies regardless of pass/fail |
| `jersey_lab_match` | `True` | Match team colour by weighted L\*a\*b\* distance (brightness-forgiving) |
| `jersey_lab_l_weight` | `0.15` | Weight on the L\* (brightness) axis when matching jersey colour |
| `jersey_lab_max_dist` | `28.0` | Max weighted L\*a\*b\* distance to count as the team colour |
| `extended_threshold_offset` | `0.3` | Faces scoring above *sharp threshold − this* (floored at 0) get their jersey identified and a per-body `is_teammate` flag, even if they miss the sharp threshold. Verdicts/stars are unaffected. Overridable per album via `run_settings["extended_threshold_offset"]` in `album.json`. AI edit's **Sharpen players** pre-selects only `is_teammate` bodies (albums imported before this need a quick regrade to get the flag) |
| `jersey_binary_lightness` | `True` | Fallback Light/Dark bucketing when `jersey_lab_match` is off |

