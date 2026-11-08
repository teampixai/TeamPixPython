# TeamPixPython

**You held the shutter down for the whole game. We get it. We're here to help.**

TeamPixPython finds **your team's sharp shots** in a mountain of burst photos. It runs on your PC, and you
make the final call in your browser.

- 🎯 Checks every player's **face** for sharpness, not just the photo as a whole
- 👕 Ignores the other team by **jersey colour** (sorry, visitors)
- 🙂 Learns your players' **faces** and tags them next time
- ⭐ Exports the keepers with **star ratings** that Lightroom and Windows understand

> Why did the photographer break up with burst mode? Too many shots, not enough focus.

<!-- Screenshot: add a screenshot of the Culling page here, e.g. docs/screenshot-culling.png -->

---

## TL;DR: get started on Windows

Nothing to install first (no Python, no Anaconda, no git). In a **Command Prompt** (<kbd>Win</kbd>, type
`cmd`, <kbd>Enter</kbd>), paste:

```
mkdir C:\TeamPixPython && cd C:\TeamPixPython
curl -L -o InitPhotoProcessing.bat https://raw.githubusercontent.com/teampixai/TeamPixPython/main/InitPhotoProcessing.bat && InitPhotoProcessing.bat
```

Accept Anaconda's [Terms of Service](https://www.anaconda.com/legal) when asked, then make a coffee: the
first install downloads a few GB. When it's done, open **TeamPixPython** from the Start Menu.

That's it. Everything lives in `C:\TeamPixPython`, and nothing else on your PC is changed. Your system
Python won't even know we were here.

> Tips: use a folder path **without spaces** (spaces in paths are like a 4-2-4 formation with no
> keeper: technically allowed, but asking for trouble). Add `/y` after `InitPhotoProcessing.bat` to skip
> the Terms of Service prompt.

---

## System requirements

| | |
|---|---|
| **OS** | Windows 10 or 11, 64-bit. Windows on ARM works through x64 emulation, but slowly. |
| **Disk** | About **10 GB** free for the app, plus room for each album's thumbnails and previews. |
| **GPU** | Optional but **strongly recommended**: an NVIDIA graphics card with driver 528 or newer. The installer detects it and installs the GPU version automatically. Without one, everything still works on the CPU, just more slowly. Think "coach's halftime speech" slowly. |
| **Internet** | Needed for installing and updating, and for downloading AI models the first time. Afterwards it's only needed if you use the optional OpenAI features. |
| **Photos** | JPEG, PNG, TIFF, WEBP, BMP and Canon RAW (`.CR2` / `.CR3`). |

macOS and Linux are not supported at the moment. We haven't got a Mac to test on, and we'd rather not
ship untested code just to see how it goes.

---

## Your first album

Five steps, left to right along the top of the page (the optional **Players** step is covered
[below](#tidying-up-a-players-faces-players)). Like a fast break, but with more clicking.

1. **Team.** Name, jersey colour(s) and, optionally, the roster. The goalie's in a different kit? Add that
   colour as *forced*: no keeper left behind. An optional **OpenAI API key** turns on AI burst ranking
   (see [Privacy](#privacy-and-your-data)).
2. **Select Album.** Click **+**, pick a photo folder, the team and a blur sensitivity (*low* = only flag
   the truly smeared shots, *high* = also flag slightly soft ones). Tick **Recognize Faces** if you want
   it. For practices or anything where the team isn't in its usual kit, tick **Ignore jersey color**. Hit
   **Import Images** and watch the progress.
3. **Culling.** Photos come pre-sorted into *Sharp*, *Blurry* and *Skipped*. Disagree with a call? One click
   overrules it; unlike real refs, this one doesn't argue back. **AI Edit** fixes brightness and contrast,
   AI-sharpens selected players and crops for Instagram, phone or print. Your original is never touched;
   edits are saved as copies.
4. **Faces.** Name each group of faces once. The app remembers them for future albums. It never forgets a
   face, which is more than we can say for the coach.
5. **Album Tools.** **Export** the keepers, each with its star rating (Lightroom and Windows Explorer can
   read it), plus a `players.csv` of who's in which photo. Heads-up: export copies your **original** files.
   AI Edit results aren't applied yet, except photos you saved with **Save Copy**, which are exported as
   photos of their own. You can also re-tune sensitivity or jersey colour, import more photos, re-run face
   detection, and (experimental) turn bursts into MP4s, which needs [FFmpeg](https://ffmpeg.org/) on your
   PATH.

Closed the browser tab by accident? The app keeps running. Open **TeamPixPython** from the Start Menu
again, or use the link in the console window.

### Tidying up a player's faces (Players)

The **Players** step (right after Team) shows each player's picture and the faces the app has learned for
them. Pick a player in the list. The middle column, **Positive**, holds photos of that player, and the
right-hand column, **Negative**, holds look-alikes that must *not* be matched to them. Rest the mouse on a
photo for a moment to see it full size.

- **Drag** photos between the two columns when a face landed in the wrong place.
- **Delete** (or the **Del** key) removes photos that aren't usable. Ctrl- or Shift-click to select several.
- **Set as profile picture** puts the selected photo next to the player's name. Players with no photos yet
  get a mystery player in a disguise.

Changes are saved when you switch to another player. Deleted photos stay in a recycle bin until you click
**Rebuild face DB**, which empties it and re-learns the team's faces. That takes a minute or two, and the
log shows the progress.

---

## Updating and uninstalling

- **Update:** run the same two lines from the [TL;DR](#tldr-get-started-on-windows) again in the same
  folder. Your albums, teams and face database stay put. Updating is the only time we *want* you to repeat
  yourself.
- **Uninstall:** delete the install folder (e.g. `C:\TeamPixPython`) and the **TeamPixPython** Start Menu
  folder (`%APPDATA%\Microsoft\Windows\Start Menu\Programs\TeamPixPython`). The install folder also holds
  your albums, teams and face database (your original photos are elsewhere and aren't affected), so copy
  anything you want to keep first.

---

## Privacy and your data

No jokes in this section. Privacy is no laughing matter.

- **Everything runs on your PC.** Photos, faces and results never leave your computer, with the one
  exception below. The web app only listens on `127.0.0.1`, so it can't be reached from other devices.
- **OpenAI (optional).** If you add an OpenAI API key to a team, or have the `OPENAI_API_KEY` environment
  variable set, every photo graded *sharp* in an import is sent to OpenAI, downscaled to about 1000 px,
  so its model can rank burst shots and suggest star ratings. You pay OpenAI for this usage, and the cost
  estimate is shown in the import log. Without a key, nothing is sent.
- **Your API key is stored in plain text** in `team.json` inside the app folder. Don't share that file.
- **Your original photos are never modified, moved or deleted.** The app only reads them. Its own working
  files live in a separate album folder (see [below](#where-files-are-stored)). Deleting an album in the
  app deletes only that working folder.
- **Face recognition is off unless you turn it on**, and the app shows a consent notice first. Photographing
  and identifying minors may require parental or guardian consent under laws such as COPPA and GDPR, so
  make sure you have it. Face data (small face crops and numeric "embeddings") is stored only on your PC,
  in the app folder.
- The web pages load the jQuery and jQuery UI libraries from `code.jquery.com`.

---

## How it works (for the curious)

### What happens to each photo

```
Your photo (read-only)
  → shrink a working copy to 1800 px          (RAW files are decoded first)
  → find every person, their pose and face    (Google MediaPipe + a PyTorch person detector)
  → measure how sharp each face is            (classic edge/detail metrics; brightness-normalised so
                                               darker skin or shade isn't mistaken for blur)
  → keep only people in your team's jersey colour
  → photo = Sharp if a teammate's face is sharp, Blurry if not, Skipped if no teammate's face was usable
  → (optional) Face recognition: match faces to your team's known players, group the rest
  → (optional) OpenAI: rank each burst, keep the best frame, suggest 1–5 stars
  → you review and adjust in the browser → export copies of the keepers
```

The heavy lifting runs in a background process, so the browser stays responsive and shows live progress.
Sharp/blurry decisions come from fixed, explainable measurements, not a black-box model. You can re-run
them at a different sensitivity without re-scanning the photos. Instant replay, minus the VAR drama.

Why check faces and not the whole frame? A tack-sharp ball and grass with a blurry striker in front is
still a blurry photo. The app judges people, not grass.

### Reading the annotated preview

On the Culling page, the annotated view shows why each photo got its verdict:

- **Blue box / ✓**: a person scored sharp. **Red box / ✗**: a person scored blurry.
- **Green box**: the tight face area that was measured. **Yellow box**: the full detected face.
- **Dots**: face landmarks (blue = confident, red = uncertain).
- **Number** under the face: the sharpness score, followed by the player's name when face recognition
  knows it.
- A darkened border means the photo has been cropped in AI Edit.

### Where files are stored

Everything lives in the folder you installed into:

```
C:\TeamPixPython\
├─ miniconda3\                 private Python installer (not on PATH, not in the registry)
├─ envs\teampixpython\         the app's private Python and libraries
├─ RunPhotoProcessing.bat      launcher (what the Start Menu shortcut runs)
└─ app\                        the app itself (culling_app.py), plus all of your data:
   ├─ team.json                your teams, rosters (and OpenAI key, if set)
   ├─ .FaceReco\<team>\        your face database: one folder per player per team
   └─ albums\<date-time>-<name>\   one working folder per import:
      ├─ album.json, info.json     detection results, scores, verdicts, star ratings
      ├─ thumbnails\, previews\    small images for the UI (safe to delete; rebuilt when needed)
      ├─ editedimages\, SaveCopy\  your AI Edit results
      ├─ .FaceReco\                this album's face groups
      └─ run.log, blurry.csv       import log and blur report
```

Your original photos stay where they are. An album only stores references to them, so keep the source
folder in place while you work on an album. An export writes **copies** of the kept photos into the folder
you choose.

AI models (under 1 GB in total) are downloaded automatically the first time they're needed.

---

## For developers

Developer documentation (architecture, data formats, invariants, dev setup and tests) is in
[AGENTS.md](AGENTS.md). It's written for both human contributors and AI coding agents. Yes, the robots
get their own playbook now.

---

## License and third-party attribution

### This project

TeamPixPython is **open source**. The specific license is still being finalised; see [LICENSE](LICENSE).

> **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND**, express or implied, including but
> not limited to the warranties of merchantability, fitness for a particular purpose and non-infringement.
> In no event shall the authors or copyright holders be liable for any claim, damages or other liability
> (including lost, deleted or misclassified photos) arising from, out of or in connection with the
> software or its use. Always keep a backup of your photos.

### Third-party components

TeamPixPython builds on the following open-source software and models. Their licenses belong to their
respective authors. None of the model weights are included in this repository; they are downloaded from
their original publishers when first needed.

**Libraries**

| Component | Used for | License |
|---|---|---|
| [Python](https://www.python.org/) (incl. tkinter) | runtime, folder picker | PSF License |
| [Miniconda](https://docs.conda.io/en/latest/miniconda.html) / [conda-forge](https://conda-forge.org/) | private Python install (Windows installer) | [Anaconda ToS](https://www.anaconda.com/legal) / BSD-3-Clause (conda); per-package |
| [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision) | neural-network runtime, person detector | BSD-3-Clause |
| [MediaPipe](https://github.com/google-ai-edge/mediapipe) | pose, face landmarks, body segmentation | Apache-2.0 |
| [facenet-pytorch](https://github.com/timesler/facenet-pytorch) | face embeddings (default) | MIT |
| [dlib](http://dlib.net/) and [face_recognition](https://github.com/ageitgey/face_recognition) | alternative face embeddings | Boost Software License 1.0 / MIT |
| [OpenCV](https://opencv.org/) (`opencv-python`) | image processing | Apache-2.0 (OpenCV), MIT (wrapper) |
| [rawpy](https://github.com/letmaik/rawpy) / [LibRaw](https://www.libraw.org/) | RAW decoding | MIT / LGPL-2.1 or CDDL-1.0 (dynamically linked) |
| [Pillow](https://python-pillow.org/) | EXIF capture time | MIT-CMU (HPND) |
| [piexif](https://github.com/hMatoba/Piexif) | star ratings in exported EXIF | MIT |
| [NumPy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/) | maths, face clustering | BSD-3-Clause |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [BasicSR](https://github.com/XPixelGroup/BasicSR) | AI sharpening | BSD-3-Clause / Apache-2.0 |
| [GFPGAN](https://github.com/TencentARC/GFPGAN), [facexlib](https://github.com/xinntao/facexlib) | face restoration support | Apache-2.0 (see repo for bundled third-party terms) / MIT |
| [FastAPI](https://fastapi.tiangolo.com/), [Starlette](https://www.starlette.io/), [Pydantic](https://docs.pydantic.dev/) | web server | MIT / BSD-3-Clause / MIT |
| [Uvicorn](https://www.uvicorn.org/), [Jinja2](https://jinja.palletsprojects.com/) | web server, page templates | BSD-3-Clause |
| [python-multipart](https://github.com/Kludex/python-multipart) | form handling | Apache-2.0 |
| [jQuery](https://jquery.com/), [jQuery UI](https://jqueryui.com/) | browser UI | MIT |
| [OpenAI Python SDK](https://github.com/openai/openai-python) | optional burst ranking | Apache-2.0 |
| [FFmpeg](https://ffmpeg.org/) | optional MP4 rendering (installed separately by you); also bundled inside `opencv-python` wheels | LGPL-2.1+ |

**AI models (downloaded at first use)**

| Model | Source | License |
|---|---|---|
| Pose Landmarker, Face Landmarker, Selfie Segmenter | [Google MediaPipe models](https://ai.google.dev/edge/mediapipe/solutions/guide) | Apache-2.0 |
| Faster R-CNN ResNet-50 FPN v2 (COCO) | torchvision | BSD-3-Clause |
| InceptionResnetV1 (VGGFace2) | facenet-pytorch | MIT |
| dlib face models ([face_recognition_models](https://github.com/ageitgey/face_recognition_models)) | dlib / face_recognition | see repository (CC0 and dlib model terms) |
| RealESRGAN_x4plus, realesr-general-x4v3 | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) | BSD-3-Clause |
| GFPGAN v1.4 | [GFPGAN releases](https://github.com/TencentARC/GFPGAN/releases) | Apache-2.0 (see repo) |

**Optional, off by default (copyleft).** A legacy detection engine based on
[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0, or a paid Enterprise license) and
[yolo-face](https://github.com/akanametov/yolo-face) (GPL-3.0) is kept for comparison. The app never uses it,
but the `ultralytics` package is installed as a dependency. Review those licenses before enabling it in any
distributed or hosted use.

If you notice a missing or incorrect attribution, please open an issue. Credit where credit's due: this
project stands on the shoulders of giants, and we try hard not to blur them.
