# AGENTS.md — guide for coding agents and contributors

This file is for people and AI agents working on the code. The user-facing guide is [README.md](README.md),
which deliberately mentions only the web app (`culling_app.py`). Put developer detail here, not there.

---

## 1. What this is

TeamPixPython is a local photo-culling tool for team-sports photographers. A FastAPI web app
(`culling_app.py`) drives a detection pipeline that finds people, scores the sharpness of each face, keeps
only players in the team's jersey colour, optionally clusters and recognises faces, and can optionally rank
burst shots with an OpenAI vision model. The user reviews the results, tags faces, edits photos and exports
the keepers.

Main platform: **Windows 10/11**, installed with `InitPhotoProcessing.bat`. See §9 for other platforms.

## 2. Repo map

| Path | Role |
|---|---|
| `culling_app.py` | **The app.** FastAPI server (default `127.0.0.1:8010`) plus all page routes and `/api/*` endpoints. Long-running work is started as subprocesses (see §3). |
| `webui/culling/templates/*.html` | One Jinja2 template per page (`team`, `players`, `select-album`, `add-new-album`, `culling`, `faces`, `album-tools`), all extending `base.html` (top bar and step navigation). |
| `webui/culling/static/js/*.js` | One jQuery script per page, plus shared `common.js` (API helpers, `LogDialog`, heartbeat), `viewport.js` (zoom/pan image window) and `ai-edit.js` (the AI Edit dialog, `AiEdit.run(key, { onResult })`). jQuery 3.7.1 and jQuery UI 1.13.2 load from code.jquery.com. |
| `webui/index.html`, `webui/static/` | Older single-page UI. Not served by `culling_app.py`. |
| `1_prep_review.py` | Import/scoring worker. Turns a folder of photos into an album. |
| `apply_export.py` | Export worker. Copies kept photos flat to a destination, writes `players.csv`, adds EXIF star ratings (piexif). |
| `autoedit.py` | AI Edit worker: tone (`apply_tone()`), Real-ESRGAN player sharpening, crop. |
| `esrsharpen.py` | Standalone Real-ESRGAN/GFPGAN sharpening. |
| `RebuildFaceDB.py` | Recomputes face-DB embeddings and calibrates match thresholds. |
| `MigrateFaceDB.py` | One-off migration from a flat face DB to the per-team layout. |
| `RerunLLMCulling.py` | Re-runs LLM burst culling on an existing album. |
| `algo/` | Core library (see below). |
| `algo/stages/` | Pipeline stages: `image_analysis`, `grading`, `jersey_counting`, `face_reco`, `llm_culling`, `auto_adjust`, plus `annotation` (on-demand preview renderer, not a stage). |
| `algo/llm/` | OpenAI provider (`culling_provider.py`, default model `gpt-6-luna`, pricing tables) and prompts. |
| `tests/` | pytest suite (`conftest.py` puts the repo root on `sys.path`). `test/` and `TestImages/` are local, git-ignored scratch areas. |
| `InitPhotoProcessing.bat` | End-user Windows installer and updater (§8). |

Key `algo/` modules:

| Module | What it owns |
|---|---|
| `album.py` | `Album` / `AlbumImage` / `PersonRecord`: the **only** layer that knows the shape of `album.json`. `album_for(path)` returns a cached instance. |
| `album_info.py` | `AlbumInfo`: `info.json`, the per-category file lists (blur / sharp / skipped). |
| `config.py` | `AppConfig` tunables (§7). |
| `sharpness.py` | `SharpnessEvaluator` implementations. The active one is the module-level `sharpness_evaluator` (currently `ContrastNormalizedEvaluator`). |
| `regrade.py` | Quick regrade: re-buckets blur/sharp from stored scores without re-detecting. |
| `facereco.py`, `face_crop_embed.py`, `*_provider.py` | Face clustering and matching. Providers: `facenet_provider` (default), `dlib_provider`, `mediapipe_provider`, `torchvision_provider` (person boxes). |
| `segmentation.py` | MediaPipe selfie-segmenter mask used to limit sharpening to the players' bodies. |
| `burst_video.py` | Burst detection (by capture time) and MP4 rendering (needs `ffmpeg` on PATH). |
| `imagefiles.py` | Supported extensions: jpg/jpeg/png/bmp/tif/tiff/webp, plus RAW `.cr2`/`.cr3` via rawpy. |
| `utils.py` | Shared helpers (EXIF capture time, thumbnails, `cap_long_edge`, atomic save). |

## 3. Runtime architecture

- **Server:** `culling_app.py` serves the pages and a JSON API. Each page is a separate template and script,
  with no SPA router.
- **Long jobs run as subprocesses** started with `sys.executable`: import (`1_prep_review.py`), export
  (`apply_export.py`), AI Edit (`autoedit.py`), and face-DB rebuild (`RebuildFaceDB.py`, from export and
  from the Players page). The UI polls output endpoints such as `/api/processing-output?since=N` and shows
  the log in a dialog.
- **Rule: use the log-dialog UX for anything slow.** Any UI action that runs another Python script, or is
  likely to take more than about 3 seconds, must follow the same pattern:
  - run in a background thread tracked by a `BackgroundJobState` subclass in `culling_app.py`, where
    `claim()` returns 409 while it's busy;
  - expose a `…/status?since=N` endpoint that returns `state.status(since)`;
  - show progress on the page with `LogDialog.create({...})` from `common.js`.

  Never block the request, and don't build a one-off spinner or progress UI.
  `_stream_subprocess(cmd, log_fn=...)` forwards a script's output into any job log.
- **Heartbeat:** every page pings `/api/heartbeat` every 10 s. After `--heartbeat-timeout` seconds
  (default 60; 0 disables) without a ping, the server **prints a relaunch link instead of exiting**, so a
  running job survives a closed tab.
- **Native dialogs:** the folder picker (`/api/browse-folder`) uses tkinter, called directly from the
  FastAPI worker thread. "Open folder" uses `os.startfile`, and "Edit with custom app" uses the Windows
  "Open with" chooser (§9).
- **CLI:** `python culling_app.py [--host H] [--port P] [--no-browser] [--heartbeat-timeout S]`.

### Processing pipeline (import)

```
photo → normalise (long edge 1800 px, never upscale; RAW decoded via rawpy)
      → MediaPipeImageAnalysisStage  person boxes (torchvision Faster R-CNN) + pose / face landmarks (MediaPipe)
                                     (ImageAnalysisStage = legacy YOLO engine)
      → GradingStage         sharpness of each face crop → per-person pass/fail
      → JerseyCountingStage  dominant team colour (CIE L*a*b*) → drop off-team people (unless "ignore jersey")
      → FaceRecoStage        (optional) match to team face DB, cluster the rest into <album>/.FaceReco/
      → LLMCullingStage      (optional, needs OpenAI key) rank bursts / grade sharp + near-sharp frames 0–1 → star ratings (1–5) + keep
      → write album.json, info.json, thumbnails/, blurry.csv, blur.lst, run.log
```

Annotated previews are **not** rendered at import. `algo/stages/annotation.py` draws them the first time
they're requested and caches them in `previews/`. Code that changes what a preview shows (a face
assignment, a regrade, a face-reco re-run) must call `invalidate_previews`.

Detection engine: `--engine mediapipe` (default, Apache-2.0) or `--engine yolo` (legacy, AGPL/GPL,
opt-in only, never enable by default).

## 4. Data and files

All paths are relative to the app folder (`REPO_ROOT`, which is `<install root>\app\` for installed users).

| File / folder | Written by | Notes |
|---|---|---|
| `team.json` | Team page | Teams keyed by a **UUID** `id`. `name` is only for display. `legacyIds` keeps pre-UUID ids that old albums still reference. Holds `openaiApiKey` **in plaintext**. Git-ignored. |
| `.culling_state.json` | app | Last import/export paths, current album, debug flags (e.g. `save_silhouette`). Git-ignored. |
| `albums/<yyyymmdd-hhmmss>-<folder>/` | import | One album per import. Originals are **referenced in place, never copied or modified**. |
| ↳ `album.json` | `algo/album.py` only | Per-photo detections, scores, verdicts, stars, `edited_crop`/`edited_crop_px`, cached capture time, `run_settings`. |
| ↳ `info.json` | `algo/album_info.py` | Category lists (`Anno_Blur` / `Anno_Sharp` / `Anno_Skipped`). |
| ↳ `thumbnails/<key>.jpg` | import (or on demand) | Plain square thumbnails for the nav strip. |
| ↳ `previews/<key>.jpg` | on demand | Annotated preview cache. Safe to delete. |
| ↳ `editedimages/` | AI Edit | Accepted edits. **Revert** deletes the edit. |
| ↳ `SaveCopy/<name>_<N>.jpg` | AI Edit "Save Copy" | Added to the album as a new photo, with detections re-framed onto the crop. |
| ↳ `.FaceReco/<NNNN>/{Face/, Negative/, face.json}` | face reco | Per-album clusters. |
| ↳ `blurry.csv`, `blur.lst`, `run.log` | import | Report, list and debug log. |
| `.FaceReco/<team id>/<Player>/{face.json, Face/, Negative/}` | export (face-DB sync), Players page | Face DB shared across albums, **one folder per team**. A player on two teams is stored as two separate people. See "Players page" below. |
| `*.pt`, `*.pth`, `*.task`, `*.tflite` | first run | Model weights, downloaded at runtime and git-ignored. Never commit them. |

Photo **key** = the file's full name (e.g. `IMG_0001.CR3`), so a RAW+JPEG pair of the same shot counts as
two photos.

Export (`apply_export.py`) writes **flat** to the destination (kept photos plus `players.csv`) and only reads
the album. `_apply_edits` is still a placeholder (a byte-for-byte `copy2` of the original), so accepted AI
edits in `editedimages/` are **not** exported yet. Save Copy photos are exported because they are separate
album entries.

### Players page (team face-DB curation)

`/players` (`players.html` / `players.js`, `/api/players/*` in `culling_app.py`) edits the
`.FaceReco/<team id>/` folders directly.

- **People.** Folders are grouped by their `face.json` `name`, so an alias folder `Ansel` named "Ansel Chang"
  is merged into that player. The primary folder (`_sanitize_player_dirname(name)`) comes first. Roster
  players with no folder are listed with no images.
- **Move.** Moving a crop between `Face/` and `Negative/` also moves its embedding entry between `faces` and
  `negative_faces`, so matching sees the change without a rebuild.
- **Delete.** The crop moves to `<Player>/Deleted/`, its embedding is dropped, and its name is added to
  `face.json` `deletedCrops`. Export sync (`_sync_named_clusters_to_system_db`) skips crops that are in
  `deletedCrops` or in `Negative/`, so curated-out faces never come back.
- **Profile picture.** It is stored as `profileCrop` in the primary folder's `face.json`. When that crop is
  no longer a positive, the app falls back to the first positive, otherwise `null` (the UI then shows
  `static/img/secret-player.png`).
- **Rebuild.** `RebuildFaceDB.py` preserves `profileCrop` and `deletedCrops` when it rewrites `face.json`.
  **Rebuild face DB** (`POST /api/players/rebuild`, `FaceDbRebuildState`):
  1. permanently deletes every `Deleted/` and `Annotated.*` folder;
  2. runs `RebuildFaceDB.py` with calibration and the normal retire step.
- **Concurrency.** Commits and rebuilds return 409 while an export is running, and export refuses to start
  during a rebuild.

## 5. Invariants and gotchas

- **Never index into `album.json` entries directly.** Use `AlbumImage` / `PersonRecord`.
  `tests/test_no_raw_entry_access.py` fails the build on `["annotation_data"]`, `["evaluated"]`,
  `["stars_manual"]` and similar key accesses outside `algo/album.py`.
- **Sensitivity presets are duplicated:** `SENSITIVITY_PRESETS` in `culling_app.py` and
  `SENSITIVITY_THRESHOLDS` in `1_prep_review.py` (low 0.40 / medium 0.50 / high 0.62; score ≤ threshold
  means blurry). Change both together, and recalibrate when the evaluator changes.
- **Dependency pins:** `torch==2.2.2` / `torchvision==0.17.2` (required by facenet-pytorch), `numpy<2`
  (torch 2.2 ABI), `setuptools<81` (keeps `pkg_resources` for face_recognition_models). Install torch
  **before** `pip install -r requirements.txt`, or pip will replace the CUDA build with a CPU one.
  `esrsharpen.py`/`autoedit.py` patch basicsr's removed `torchvision.transforms.functional_tensor` import
  at runtime.
- **Team identity is the UUID**, not the name. Face-DB folders and `album.team_id` use it. Match `legacyIds` too.
- **Licensing:** keep the default path permissive (§10). Ultralytics/YOLO stays opt-in.
- **Face recognition is consent-gated in the UI** (a COPPA/GDPR notice in `add-new-album.html`). Keep it.
- `.gitignore` excludes `/.[0-9]*.md`, which is the naming convention for local scratch plan docs.
- **jQuery element props:** in `$("<div>", { ... })`, any key that is also a jQuery method is *called*
  rather than set as an attribute. With jQuery UI loaded, `draggable: true` therefore turns the element into
  a jQuery UI draggable, not an HTML5 drag source. Use `.attr("draggable", "true")`.

## 6. Developing and testing

```powershell
# Dev environment (conda, Python 3.12, CUDA 12 example)
conda create -n teampixpython python=3.12 pip git
conda activate teampixpython
conda install -c conda-forge dlib           # prebuilt, no compiler needed
pip install "setuptools<81"
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121   # or the CPU index
pip install -r requirements.txt
pip install https://github.com/ageitgey/face_recognition_models/archive/master.zip

# Run the app
python culling_app.py --no-browser --port 8010

# Tests (≈220 tests, ~40 s; no GPU or network needed)
python -m pytest tests -q
```

`tests/e2e_check.py` is a manual end-to-end script, not part of the pytest suite.

### Import worker CLI (`1_prep_review.py`): reference

The app builds this command line itself. It's useful for debugging an import without the UI.

| Option | Default | Description |
|---|---|---|
| `path` | – | Image file or folder |
| `--output <dir>` | `albums/<timestamp>-<input>/` | Album folder |
| `--sensitivity low\|medium\|high\|<0–1>` | `medium` | Blur threshold |
| `--engine mediapipe\|yolo` | `mediapipe` | Detection engine |
| `--jerseycolor "C1;C2;+C3"` | built-in list | Team colours. `+` forces inclusion. An empty string disables filtering. |
| `--team-color HUE:SHADE`, `--team-id ID` | – | Team colour override, and the team UUID stored on the album |
| `--noteam` | off | Score everyone (the UI's "Ignore jersey color") |
| `--skip-facereco` | off | Skip face clustering |
| `--face-db <dir>` | auto-discover | Face DB root or team folder |
| `--face-db-match-threshold` / `--face-db-match-margin` / `--face-db-prototype-threshold` | from `<face-db>/calibration.json`, else built-in | Match acceptance (best ≥ threshold, and best − runner-up ≥ margin) and prototype splitting |
| `--disable-face-db-calibration` | off | Ignore `calibration.json` |
| `--min-face-crop-px` | `32` | Smaller face crops are ignored for reco |
| `--debug-align` | off | Alignment QA images in `.FaceReco/.debug` |
| `--openaikey KEY` | `OPENAI_API_KEY` env var | Enables LLM burst culling |
| `--llm-model`, `--skip-llm-cull` | `gpt-6-luna` | LLM model, or opt out |
| `--autoadjust`, `--cpu-only`, `--enable-long-paths` | off | Auto exposure, force CPU, Windows long-path registry opt-in |
| `--rerun-facereco-only`, `--regrade-only` | off | Partial re-runs on an existing album (used by Album Tools) |

Run `python 1_prep_review.py --help` for the full, authoritative list.

## 7. Algorithms (summary)

- **Sharpness:** each face crop is resized so its long edge is 96 px and contrast-normalised (target std 40,
  gain clamped to 0.5–4), then scored with Laplacian variance and Tenengrad combined as a weighted
  geometric mean. The geometric mean is only high when *both* metrics agree, which makes it robust to noise
  and over-sharpening. See `algo/sharpness.py`.
- **Jersey filter:** the jersey patch colour is compared in L\*a\*b\* with brightness down-weighted
  (`jersey_lab_l_weight` 0.15, `jersey_lab_max_dist` 28), and the team colour is the most common one among
  sharp bodies (a pinned team colour, which may be a bare hue like `White`, gets its anchor from the bodies
  matching it, or from the reference colour). Forced (`+`) colours pass by label, or by LAB distance only when
  the forced colour is also the closest reference colour to the body. The extended threshold, `min(sharp, max(0.3, sharp − 0.2))` (`extended_threshold_floor` /
  `extended_threshold_offset` in `AppConfig`; an album may store its own offset), also tags near-sharp bodies
  with `is_teammate`, which AI Edit uses to pre-select players to sharpen, and splits blurry photos into
  2 stars (graded by the LLM) and 1 star.
- **Face-DB matching:** each person's reference embeddings are split into prototypes (agglomerative
  clustering). A face matches a person only if (1) its best prototype similarity ≥ the threshold, (2) it
  beats the best *other* person by the margin, and (3) the person's `Negative/` examples don't veto it.
  `matchScore`, `matchMargin` and `matchRunnerUp` are written to `face.json`.
  `RebuildFaceDB.py <db> [--prototype-threshold N] [--skip-calibration]` recomputes embeddings and runs a
  leave-one-out calibration that recommends a threshold and margin and lists confusable photos.
  A rebuilt DB's `face.json` uses a flatter `faces` schema (`{dtype, shape, encoding, value}`), which is the
  form `FaceDb.load` reads.
- **LLM culling and star ratings:** runs whenever a key is available. The app passes the team's key;
  otherwise the `OPENAI_API_KEY` environment variable is used. Every *gradable* photo is sent (sharp, or
  blurry above the extended threshold; bursts are ranked together and standalone frames are graded
  individually), as JPEGs downscaled to a 999 px long edge at quality 80. The default model is
  `gpt-6-luna`; the model that produced an album's grades is stored as `llm_model` in `album.json`. Cost is
  estimated from the per-model token tables in `algo/llm/culling_provider.py` (the `gpt-6-luna` rows are
  assumed equal to `gpt-5.6-luna`).
  - **Base stars:** sharp = 3; blurry above the extended threshold = 2; blurry at/below it, or no face = 1.
    `keep` = stars ≥ 3 (set with `AlbumImage.apply_auto_rating`, which skips hand-rated photos).
  - **High stars** (`algo/star_rules.py`, tunables `CLUSTERED_STAR_PARAMS` / `STANDALONE_STAR_PARAMS` in
    `algo/stages/llm_culling.py`): decided top-down. 5 stars = top 3% of the pool by `llm_grade` AND the best
    photo within ±1.5 s. The 5-star photos are then removed from the pool, and 4 stars = top 20% of what is
    left AND the best within ±0.5 s. A pool is one qualifying burst sequence (> 0.5 s or > 8 frames), or all
    standalone photos together. Each pool always yields at least one 5-star (its best photo). 2-star photos
    can be promoted; an ungraded photo keeps its base stars.
  - **Capture time** (`algo/utils.py::image_capture_timestamp`) carries sub-seconds (EXIF SubSecTime*, else
    the `-NNN` hundredths suffix of the file name, else `+0.0005 s`). A cached `capture_ts` without a
    millisecond part is invalid (`AlbumImage.capture_ts` returns None) and is re-read from the source.

`AppConfig` (`algo/config.py`) holds the tunables: `normalized_img_max_long_edge` (1800),
`use_narrow_face_box`, `face_min_size_fraction` (0.025), `face_crop_{min,max}_long_edge_px` (96),
`sharpness_contrast_*`, `face_detection_min_confidence` (0.1), `face_keypoint_fallback`,
`face_coverage_min_visible` (2), `jersey_lab_*`, `extended_threshold_offset` (0.2), `extended_threshold_floor` (0.3), `annotation_alpha` (0.25)
and `annotation_top_n_bodies` (5). Read the file for defaults and comments.

Known limitation: MediaPipe gives no per-landmark visibility score, so `face_coverage_min_visible` has no
effect under the mediapipe engine.

## 8. Installer (`InitPhotoProcessing.bat`)

The installer is self-contained and safe to re-run. It:

1. installs a private Miniconda into `<root>\miniconda3` (not on PATH and not registered, after the user
   accepts the ToS or passes `/y`);
2. creates the env `<root>\envs\teampixpython` (Python 3.12, with pip and git);
3. detects an NVIDIA driver ≥ 528 to choose CUDA (cu121) or CPU builds;
4. downloads the GitHub ZIP of `main` into `<root>\app` (keeping `albums\` and other local data);
5. installs dlib from conda-forge, torch 2.2.2, `requirements.txt`, and face_recognition_models from the
   GitHub ZIP;
6. writes `RunPhotoProcessing.bat`, which runs `app\culling_app.py`;
7. creates the Start Menu shortcut "TeamPixPython".

Conda config, package cache and environment variables are isolated from any system Python. If a change
adds a dependency, update **both** `requirements.txt` and, where it needs special handling, the installer.

## 9. Platform notes

The app is Windows-only today. Before claiming macOS or Linux support, fix these:

- `/api/browse-folder` creates a tkinter `Tk()` inside a FastAPI worker thread. macOS requires Tk on the
  main thread, so this will crash or hang there.
- `/api/edit/open-external` returns HTTP 501 on anything except Windows (it uses `os.startfile(..., "openas")`).
- There's no installer for other platforms. dlib and torch would need a separate setup, and CUDA isn't
  available on macOS (no MPS code paths exist; everything falls back to CPU).
- `_open_in_file_explorer` already handles `open` (macOS) and `xdg-open` (Linux).
- `1_prep_review.py --enable-long-paths` is Windows-only (registry). Album folder names are capped at 60 characters to stay under MAX_PATH.

## 10. Licensing rules for new dependencies

Default code paths must stay permissively licensed (MIT / BSD / Apache-2.0 / Boost / HPND). LGPL is
acceptable only as a dynamically linked binary (LibRaw through rawpy, FFmpeg in opencv wheels). Anything
GPL or AGPL (e.g. Ultralytics) must remain opt-in and documented. When you add a dependency or model,
add it to the attribution table in [README.md](README.md#license-and-third-party-attribution).
