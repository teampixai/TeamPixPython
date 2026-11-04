#!/usr/bin/env python3
"""autoedit.py — Configurable auto-editing pipeline for photographs.

Runs a user-selected set of editing pipelines over one image, a directory of
images, or a list file of image paths. Pipelines are composed in a fixed,
code-defined order (never the order the user typed them) because they have
tonal interdependencies; the result of each pipeline feeds the next, producing
one output file per input image.

Usage:
    python autoedit.py photo.jpg
    python autoedit.py C:\\shoot\\keepers -pipelines "brightness level"
    python autoedit.py keepers.lst -pipelines "all openai_ig" --openaikey sk-...
    python autoedit.py photo.cr3 -o ./edited -ext png
    python autoedit.py photo.jpg -pipelines tone --crop 100,50,1600,2000

Available pipelines (see PIPELINE_ORDER for the canonical execution order):
    tone             Highlights/midtones/shadows exposure (EV) + contrast,
                     from --highlights/--midtones/--shadows/--contrast, or
                     propose_tone()'s auto values when none are given. Also
                     drives the culling app's edit-mode live preview.
    level            Auto black/white point correction.
    brightness       Auto exposure correction (highlight-safe gamma).
    aisharpen        Real-ESRGAN (realesr-general-x4v3, compact) detail
                     restoration at input resolution. Optional GFPGAN
                     face-enhance pass via --face-enhance (off by default).
                     With --boxes-file, sharpens+blends only the detected
                     players instead of the whole frame — see
                     AI_SHARPEN_SCOPE/sharpen_bodies() below.
    openai_autoedit  Generative retouch via the OpenAI image model.
    openai_ig        Generative heroic 4:5 Instagram reframe.

"all" expands to every pipeline that does NOT start with "openai" (except
"tone", which would double up with level+brightness), so the generative
(billed) stages are always opt-in by name.

This script is deliberately standalone: the image-reading and extension
helpers are duplicated from algo/stages/image_analysis.py rather than imported,
so running a brightness-only edit doesn't drag in YOLO/ultralytics. Real-ESRGAN
weights are auto-downloaded to the repo root on first use, mirroring
esrsharpen.py, which this script is intended to eventually replace.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import math
import queue
import sys
import threading
import types
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# facexlib (a GFPGAN dependency, only imported when --face-enhance is used)
# still loads its ResNet50 backbone via torchvision's old pretrained=True API.
warnings.filterwarnings("ignore", message=r".*'pretrained'.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r".*for 'weights' are deprecated.*", category=UserWarning)

log = logging.getLogger("autoedit")

_REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Supported input formats
# ---------------------------------------------------------------------------

_RAW_EXTENSIONS: frozenset[str] = frozenset({".cr3", ".cr2"})
IMAGE_EXTENSIONS: frozenset[str] = (
    frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}) | _RAW_EXTENSIONS
)

OUTPUT_EXTENSIONS: frozenset[str] = frozenset({"jpg", "png"})

DEFAULT_OUTPUT_DIR = Path("output_img")
DEFAULT_COMPRESSION = 95
DEFAULT_WORKERS = 4

# ---------------------------------------------------------------------------
# Tonal analysis tuning
# ---------------------------------------------------------------------------

# Tonal statistics are measured on a downscaled copy: it is both much faster
# than sorting a full-resolution frame and acts as a low-pass filter so hot
# pixels and specular pinpoints can't define the black/white points.
_ANALYSIS_LONG_EDGE = 512

# Weighting runs from 1.0 at the frame centre down to this at the corners.
# Deliberately never metered off a detected face/subject: dark-skinned players
# drag a subject-metered reading down and the whole frame gets blown out.
_CENTER_WEIGHT_FLOOR = 0.25
_CENTER_WEIGHT_SIGMA = 0.40

_LEVEL_LOW_PERCENTILE = 0.5
_LEVEL_HIGH_PERCENTILE = 99.5
# Skip levelling entirely when the histogram already spans this much range.
_LEVEL_SPAN_SKIP = 0.95
# Caps how hard a low-contrast (hazy, backlit) frame gets pushed.
_LEVEL_MAX_GAIN = 1.6

_BRIGHTNESS_TARGET = 0.45
_BRIGHTNESS_MIN_GAMMA = 0.45
_BRIGHTNESS_MAX_GAMMA = 2.20

# ---------------------------------------------------------------------------
# AI sharpening
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SharpenModelSpec:
    """One selectable Real-ESRGAN checkpoint + the net architecture it needs."""

    key: str
    label: str
    model_name: str
    model_url: str
    kind: str  # "compact" (SRVGGNetCompact) or "rrdbnet" (RRDBNet)
    num_feat: int = 64
    num_conv: int = 32     # compact only
    num_block: int = 23    # rrdbnet only
    num_grow_ch: int = 32  # rrdbnet only


# Two switchable models for A/B testing (-sharpen-model), both restoring detail
# at outscale=1 (no resolution change):
# - "compact" (default): realesr-general-x4v3, SRVGGNetCompact — a fraction of
#   x4plus's size/VRAM/compute, meant for general photo restoration.
# - "x4plus": the original RealESRGAN_x4plus RRDBNet (23 residual blocks) —
#   much heavier but restores more/stronger detail; also used by esrsharpen.py.
SHARPEN_MODELS: dict[str, SharpenModelSpec] = {
    "compact": SharpenModelSpec(
        key="compact", label="realesr-general-x4v3 (compact)",
        model_name="realesr-general-x4v3.pth",
        model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        kind="compact", num_feat=64, num_conv=32,
    ),
    "x4plus": SharpenModelSpec(
        key="x4plus", label="RealESRGAN_x4plus (RRDBNet)",
        model_name="RealESRGAN_x4plus.pth",
        model_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        kind="rrdbnet", num_feat=64, num_block=23, num_grow_ch=32,
    ),
}
DEFAULT_SHARPEN_MODEL = "compact"

# ---------------------------------------------------------------------------
# Body-only sharpening (see sharpen_bodies() below)
# ---------------------------------------------------------------------------
# Code-only switch (deliberately not exposed via CLI/UI — edit this constant
# directly to change it): "bodies" sharpens+blends just the detected players'
# boxes, leaving the rest of the frame pixel-identical to the source; "full"
# restores the old whole-frame behaviour. Falls back to "full" automatically
# per-image when no body boxes are supplied (see AiSharpenPipeline.apply()),
# so photos with nothing detected are unaffected either way.
AI_SHARPEN_SCOPE = "bodies"  # "bodies" | "full"

# Extra margin added around each detected body box before cropping, as a
# fraction of that box's own width/height. Gives the segmentation model
# background context around the player and keeps the blended region's edge
# away from the (often tight) detection box boundary. Tune freely.
BODY_SHARPEN_PADDING_PCT = 0.20

# Extra Gaussian blur (pixels) applied to the segmentation confidence mask
# before using it as the alpha-blend weight, on top of the model's own
# confidence gradient — cheap insurance against a visible seam at the
# player's silhouette. 0 disables it.
BODY_SHARPEN_MASK_BLUR_PX = 3

# The segmentation mask covers every person in a crop, so a blurry neighbour
# standing inside the padded crop would otherwise be sharpened too. The mask
# is therefore clipped to the target's own body box, grown by this fraction
# of the box's width/height (so arms/legs poking a little outside the
# detector's box aren't cut off) and softened by this Gaussian sigma (px).
BODY_SHARPEN_CLIP_MARGIN_PCT = 0.05
BODY_SHARPEN_CLIP_FEATHER_PX = 6

# A segmentation counts as successful only if it marks at least this
# fraction of the padded crop as person (confidence > 0.5). Below it the
# model effectively found nobody, so the box is skipped and left untouched.
BODY_SHARPEN_MIN_PERSON_COVERAGE = 0.01

# Minimum per-keypoint confidence before a body's own 17 COCO keypoint may
# "vote" for a connected component in the segmentation mask — see
# _select_body_component() below. Caller-supplied keypoints are optional
# (see sharpen_bodies()'s keypoints_norm param); this only affects boxes
# whose caller did pass them.
BODY_SHARPEN_KEYPOINT_CONF_MIN = 0.3

# COCO-17 limb bones (pairs of keypoint indexes) of the body-shape model that
# tells overlapping players apart (see _protected_keep_mask()): arms, legs.
_ARM_BONES: tuple[tuple[int, int], ...] = ((5, 7), (7, 9), (6, 8), (8, 10))
_LEG_BONES: tuple[tuple[int, int], ...] = ((11, 13), (13, 15), (12, 14), (14, 16))
# Body-shape model sizes, as fractions of the player's own box height: limb
# half-thickness, head radius, the torso polygon's (shoulders-hips) margin,
# and how much deeper a head counts than a body part of the same size —
# heads win ownership fights, since a sharpened stranger's face is the
# most visible error.
_BODY_ARM_RADIUS = 0.035
_BODY_LEG_RADIUS = 0.055
_BODY_HEAD_RADIUS = 0.07
_BODY_TORSO_MARGIN = 0.02
_BODY_HEAD_BONUS = 0.06
# Ownership rules (fractions of the target's box height): a protected
# player only takes a pixel when clearly deeper inside than the target
# (_BODY_TIE_MARGIN, which keeps near-equal depths from flickering); pixels
# further than _BODY_ALLOWED_MARGIN outside the target's own body model are
# never sharpened (merged neighbours, the ball, ...); and an unselected
# detection whose keypoints sit within _BODY_DUPLICATE_DISTANCE of the
# target's on average is the same person detected twice, not a neighbour.
_BODY_TIE_MARGIN = 0.04
_BODY_ALLOWED_MARGIN = 0.06
_BODY_DUPLICATE_DISTANCE = 0.12
# The ownership map is computed on a grid with at most this many cells along
# the crop's long edge, then upscaled — plenty for a feathered mask.
_PROTECT_GRID_LONG_EDGE = 512
# Peak-VRAM-per-output-pixel at tile=0/half precision, measured on a real GPU
# (Titan V) across 512-1600px square images: compact ~407-494 B/px, rrdbnet
# ~6970-7015 B/px (both stable once the fixed CUDA/model overhead amortizes).
# Padded up for safety margin/other GPUs' larger fixed overhead.
_VRAM_BYTES_PER_PIXEL: dict[str, int] = {"compact": 700, "rrdbnet": 9000}
# Only budget this fraction of currently-free VRAM for a tile, leaving
# headroom for allocator fragmentation and the next job's ramp-up.
_VRAM_SAFETY_FACTOR = 0.7
_TILE_MIN_PX = 256
_TILE_MAX_PX = 2048

# CPU has no cheap equivalent of torch.cuda.mem_get_info, so it keeps the old
# fixed-threshold behaviour: above this long edge, tile at a safe fixed size.
_AUTO_TILE_THRESHOLD_PX = 1600
_AUTO_TILE_SIZE_PX = 512

# Optional GFPGAN face-restoration pass (--face-enhance, off by default),
# same checkpoint esrsharpen.py already downloads.
_GFPGAN_MODEL_NAME = "GFPGANv1.4.pth"
_GFPGAN_MODEL_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
)


# ---------------------------------------------------------------------------
# OpenAI image model
# ---------------------------------------------------------------------------

_OPENAI_IMAGE_MODEL = "gpt-image-2"
_OPENAI_QUALITY = "high"
_OPENAI_TIMEOUT_SECONDS = 300.0

# gpt-image-2 accepts arbitrary sizes within these constraints.
_GPT_IMAGE_MAX_EDGE = 3840
_GPT_IMAGE_EDGE_MULTIPLE = 16
_GPT_IMAGE_MIN_PIXELS = 655_360
_GPT_IMAGE_MAX_PIXELS = 8_294_400
_GPT_IMAGE_MAX_ASPECT = 3.0
_GPT_IMAGE_TARGET_PIXELS = 4_000_000

# Exact 4:5, both edges multiples of 16, comfortably inside the pixel cap.
# Downsampled locally afterwards to Instagram's native feed size, which cannot
# be requested directly (1080 and 1350 are not multiples of 16).
_IG_REQUEST_SIZE = (1728, 2160)
_IG_FINAL_SIZE = (1080, 1350)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_IDENTITY_AND_KIT_RULES = """\
Hard constraints — these override every other instruction:
- Do not change the identity of any face. Every person must remain immediately
  recognisable as the same individual: preserve bone structure, facial
  proportions, the shape and spacing of eyes, nose and mouth, skin tone and
  undertone, hairline, hair texture and facial expression.
- Do not slim, reshape, age, de-age, or otherwise "beautify" any face or body
  beyond the mild blemish cleanup described above.
- Do not alter clothing, kit or equipment design in any way. Letters, numbers,
  player names, team crests, sponsor logos, manufacturer marks, stripes,
  patterns, trim, kit colours and fabric texture must be reproduced exactly as
  they appear in the source image. Do not invent, correct, straighten or
  re-letter any text or number, even if it is partially obscured or distorted.
- Do not add, remove, duplicate or reposition any person, limb, or piece of
  equipment.
- Do not change the pose or the action taking place.
- Do not add text, captions, borders, frames, watermarks or graphics.
- The result must read as an unretouched-looking photograph, not an
  illustration, painting or render."""

_AUTOEDIT_PROMPT = f"""\
Retouch this photograph to professional sports-publication quality, as a
skilled photo editor would in Lightroom and Photoshop.

You may:
- Correct exposure, white balance, contrast, saturation and colour so the
  subject reads cleanly and the skin tones look natural and accurate.
- Recover detail in blown highlights and blocked-up shadows.
- Reduce sensor noise and mild compression artefacts, and recover fine detail.
- Gently clean transient skin blemishes, stray flyaway hairs, sweat glare and
  sensor dust.
- Deepen and tidy the background so the subject separates from it, without
  changing what is actually in that background.

Preserve the original composition, framing and aspect ratio exactly.

{_IDENTITY_AND_KIT_RULES}"""

_IG_PROMPT = f"""\
Reframe this photograph into a single dramatic, heroic vertical image with a
4:5 aspect ratio, intended for an Instagram feed post.

Composition:
- Identify the main subject: the person most central to the action and most in
  focus. Build the crop around them so they dominate the frame.
- Crop in tighter than the original for impact, but keep the action readable —
  retain the ball or key object if it is part of the moment, leave natural
  headroom, and do not cut limbs at a joint.
- Place the subject using strong, intentional composition and give the frame a
  cinematic, poster-like sense of scale and energy.
- Apply the same professional colour, contrast and tonal grading you would
  apply to a published hero shot, keeping skin tones natural.

Framing constraint: build the result only from content already present in the
source photograph. Do not outpaint, extend, or invent any new background,
scenery, crowd or ground beyond the edges of the original frame.

{_IDENTITY_AND_KIT_RULES}"""


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def read_image(path: Path) -> np.ndarray | None:
    """Read *path* as a BGR uint8 array, RAW decoded at full resolution.

    Uses imdecode over a byte read rather than cv2.imread so non-ASCII paths
    work on Windows.
    """
    if path.suffix.lower() in _RAW_EXTENSIONS:
        import rawpy
        try:
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, output_bps=8, half_size=False)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            log.error("rawpy failed to decode %s: %s", path.name, exc)
            return None
    try:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        log.error("Could not read %s: %s", path.name, exc)
        return None
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        log.error("Could not decode %s", path.name)
    return image


def write_image(image: np.ndarray, path: Path, ext: str, compression: int) -> bool:
    if ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(compression, 1, 100))]
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(f".{ext}", image, params)
    if not ok:
        log.error("Encoding failed for %s", path.name)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())
    return True


def parse_crop(text: str) -> tuple[int, int, int, int]:
    """Parses --crop's "X,Y,W,H" (source pixels) into a 4-tuple of ints."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--crop expects X,Y,W,H")
    try:
        x, y, w, h = (int(round(float(p))) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--crop expects numbers: {exc}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("--crop width and height must be positive")
    return x, y, w, h


def crop_image(
    image: np.ndarray, crop: tuple[int, int, int, int], source_size: tuple[int, int] | None = None
) -> np.ndarray:
    """Cuts the (x, y, w, h) rectangle out of *image*, clamped to its bounds.

    *crop* is in the pixels of the image as read from disk (*source_size*,
    (width, height)); if a pipeline changed the image's size since, the
    rectangle is scaled to match so it still covers the same part of the
    scene."""
    height, width = image.shape[:2]
    x, y, w, h = crop
    if source_size is not None and source_size != (width, height):
        sx, sy = width / source_size[0], height / source_size[1]
        x, y, w, h = round(x * sx), round(y * sy), round(w * sx), round(h * sy)
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)
    w = min(max(1, w), width - x)
    h = min(max(1, h), height - y)
    return image[y:y + h, x:x + w].copy()


def collect_inputs(input_path: Path) -> list[Path]:
    """Resolve *input_path* into the list of images to process.

    A directory is enumerated non-recursively; a supported image resolves to
    itself; anything else is treated as a newline-delimited list file whose
    blank and '#'-prefixed lines are ignored.
    """
    if input_path.is_dir():
        return sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not input_path.is_file():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    if input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]

    images: list[Path] = []
    for lineno, raw_line in enumerate(
        input_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (input_path.parent / candidate).resolve()
        if candidate.suffix.lower() not in IMAGE_EXTENSIONS:
            log.warning("List line %d: unsupported image type, skipping: %s", lineno, line)
            continue
        if not candidate.is_file():
            log.warning("List line %d: file not found, skipping: %s", lineno, line)
            continue
        images.append(candidate)
    return images


# ---------------------------------------------------------------------------
# Tonal analysis helpers
# ---------------------------------------------------------------------------

def _analysis_view(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = _ANALYSIS_LONG_EDGE / max(h, w)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA
    )


def _luminance(image: np.ndarray) -> np.ndarray:
    """BT.709 luma of a BGR uint8 image, as float in [0, 1]."""
    bgr = image.astype(np.float32) / 255.0
    return bgr[:, :, 0] * 0.0722 + bgr[:, :, 1] * 0.7152 + bgr[:, :, 2] * 0.2126


def _center_weights(height: int, width: int) -> np.ndarray:
    """Gaussian falloff from 1.0 at the frame centre to _CENTER_WEIGHT_FLOOR."""
    ys = (np.arange(height, dtype=np.float32) / max(height - 1, 1)) - 0.5
    xs = (np.arange(width, dtype=np.float32) / max(width - 1, 1)) - 0.5
    dist_sq = ys[:, None] ** 2 + xs[None, :] ** 2
    falloff = np.exp(-dist_sq / (2.0 * _CENTER_WEIGHT_SIGMA ** 2))
    return _CENTER_WEIGHT_FLOOR + (1.0 - _CENTER_WEIGHT_FLOOR) * falloff


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, axis=None)
    sorted_values = values.ravel()[order]
    cumulative = np.cumsum(weights.ravel()[order])
    return float(sorted_values[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


def _apply_lut(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return cv2.LUT(image, lut)


def auto_level(image: np.ndarray) -> np.ndarray:
    """Stretch the luminance histogram's black and white points to full range.

    The same scale and offset are applied to all three channels so the
    correction can't introduce a colour cast.
    """
    lum = _luminance(_analysis_view(image))
    low, high = np.percentile(lum, [_LEVEL_LOW_PERCENTILE, _LEVEL_HIGH_PERCENTILE])
    span = float(high - low)
    if span <= 1e-6:
        log.debug("level: degenerate histogram, skipped")
        return image
    if span >= _LEVEL_SPAN_SKIP:
        log.debug("level: histogram already spans %.3f, skipped", span)
        return image

    gain = min(1.0 / span, _LEVEL_MAX_GAIN)
    ramp = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip((ramp - float(low)) * gain, 0.0, 1.0) * 255.0
    log.debug("level: black=%.3f white=%.3f gain=%.3f", low, high, gain)
    return _apply_lut(image, lut.round().astype(np.uint8))


def auto_brightness(image: np.ndarray) -> np.ndarray:
    """Correct exposure toward a target midtone using a gamma curve.

    Gamma is used rather than a linear gain because it maps 0 to 0 and 1 to 1,
    so it cannot clip highlights -- which matters directly for white kit and
    bright pitches. The reading is a centre-weighted MEDIAN so a bright sky or
    a large shadow mass can't drag it.
    """
    small = _analysis_view(image)
    lum = _luminance(small)
    median = _weighted_median(lum, _center_weights(*lum.shape[:2]))
    if not 1e-3 < median < 1.0 - 1e-3:
        log.debug("brightness: median %.4f out of correctable range, skipped", median)
        return image

    gamma = float(
        np.clip(
            math.log(_BRIGHTNESS_TARGET) / math.log(median),
            _BRIGHTNESS_MIN_GAMMA,
            _BRIGHTNESS_MAX_GAMMA,
        )
    )
    if abs(gamma - 1.0) < 0.01:
        log.debug("brightness: already on target (median %.3f), skipped", median)
        return image

    ramp = np.arange(256, dtype=np.float32) / 255.0
    lut = np.power(ramp, gamma) * 255.0
    log.debug("brightness: median=%.3f gamma=%.3f", median, gamma)
    return _apply_lut(image, lut.round().astype(np.uint8))


# ---------------------------------------------------------------------------
# Manual tone controls (the "tone" pipeline)
# ---------------------------------------------------------------------------
#
# The single source of truth for the culling app's edit-mode brightness and
# contrast: its live preview (culling_app.py) calls apply_tone() on a
# downscaled copy and the final render runs the same function through
# TonePipeline, so preview and output can only differ in resolution.

@dataclass(frozen=True)
class ToneParams:
    """Exposure in EV per tonal zone (-TONE_EV_LIMIT..+TONE_EV_LIMIT) plus a
    contrast amount (-TONE_CONTRAST_LIMIT..+TONE_CONTRAST_LIMIT); all zero
    is the identity."""

    highlights: float = 0.0
    midtones: float = 0.0
    shadows: float = 0.0
    contrast: float = 0.0

    def clamped(self) -> "ToneParams":
        ev = lambda v: float(np.clip(v, -TONE_EV_LIMIT, TONE_EV_LIMIT))  # noqa: E731
        return ToneParams(
            highlights=ev(self.highlights),
            midtones=ev(self.midtones),
            shadows=ev(self.shadows),
            contrast=float(np.clip(self.contrast, -TONE_CONTRAST_LIMIT, TONE_CONTRAST_LIMIT)),
        )

    def is_identity(self) -> bool:
        return not any((self.highlights, self.midtones, self.shadows, self.contrast))


TONE_EV_LIMIT = 2.0
TONE_CONTRAST_LIMIT = 100.0

# Resolution of the luminance -> luminance tone curve.
_TONE_CURVE_SIZE = 1024
# Linear-light values above this roll off smoothly toward 1.0 instead of
# hard-clipping when an exposure boost pushes them past white.
_TONE_SHOULDER = 0.8
# Logistic steepness at contrast +100 (mid-grey slope ~2.5x).
_TONE_CONTRAST_STEEPNESS = 10.0

# propose_tone() fitting: luminance bins, the share of the fit weight spread
# evenly across all bins (so sparsely used tones still get a sane curve),
# the pull toward zero, and the solver's iteration budget.
_PROPOSE_BINS = 64
_PROPOSE_BIN_FLOOR = 0.15
_PROPOSE_RIDGE = 0.01
_PROPOSE_ITERATIONS = 40
# Highlights are only ever lifted this much: brightening them risks blowing
# out white kit and sunlit pitches.
_PROPOSE_HIGHLIGHT_MAX_EV = 0.5
_PROPOSE_DEADBAND_EV = 0.1


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1.0 / 2.4) - 0.055)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def tone_zone_weights(lum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(shadow, midtone, highlight) weights for luminance in [0, 1]; they sum
    to 1 everywhere, peak at black / mid-grey / white respectively and
    cross-fade smoothly so zone edits never leave a visible seam."""
    low = _smoothstep(0.0, 0.5, lum)
    high = _smoothstep(0.5, 1.0, lum)
    return 1.0 - low, low - high, high


def _shoulder(v: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Rolls *v* (= linear value * *gain*) off above _TONE_SHOULDER so the
    boosted white point (*gain*) lands exactly on 1.0: identity wherever
    gain <= 1, continuous in gain, slope 1 at the shoulder."""
    s = _TONE_SHOULDER
    white = np.maximum(gain, 1.0)
    m = (white - s) / (1.0 - s)
    t = np.clip((v - s) / (white - s), 0.0, 1.0)
    rolled = s + (1.0 - s) * (m * t) / (1.0 + (m - 1.0) * t)
    return np.where(v > s, rolled, v)


def tone_curve(params: ToneParams, lum: np.ndarray) -> np.ndarray:
    """Maps display luminance in [0, 1] to its toned value. Exposure is a
    per-zone EV gain applied in linear light; contrast is then an S-curve
    (positive) or its inverse (negative) pivoting on mid-grey."""
    p = params.clamped()
    shadow_w, mid_w, high_w = tone_zone_weights(lum)
    gain = np.power(2.0, shadow_w * p.shadows + mid_w * p.midtones + high_w * p.highlights)
    out = _linear_to_srgb(_shoulder(_srgb_to_linear(lum) * gain, gain))
    return np.clip(_contrast_curve(out, p.contrast / TONE_CONTRAST_LIMIT), 0.0, 1.0)


def _contrast_curve(y: np.ndarray, amount: float) -> np.ndarray:
    """Contrast S-curve pivoting on mid-grey, endpoints fixed. Positive
    *amount* (0..1) is a normalised logistic of steepness
    _TONE_CONTRAST_STEEPNESS * sqrt(amount) — the sqrt makes the visible
    effect grow linearly with the slider; negative uses its exact inverse,
    so -x undoes +x."""
    k = _TONE_CONTRAST_STEEPNESS * math.sqrt(abs(amount))
    if k < 1e-3:
        return y
    lo = 1.0 / (1.0 + math.exp(k / 2.0))
    span = 1.0 - 2.0 * lo
    if amount > 0:
        return (1.0 / (1.0 + np.exp(-k * (y - 0.5))) - lo) / span
    v = np.clip(lo + y * span, 1e-12, 1.0 - 1e-12)
    return 0.5 + np.log(v / (1.0 - v)) / k


def _tone_lut(params: ToneParams) -> np.ndarray:
    """tone_curve() sampled on _TONE_CURVE_SIZE points and forced monotonic
    — opposing zone settings (e.g. shadows +2, midtones -2) could otherwise
    invert tones where the zones overlap."""
    ramp = np.linspace(0.0, 1.0, _TONE_CURVE_SIZE, dtype=np.float64)
    return np.maximum.accumulate(tone_curve(params, ramp))


def apply_tone(image: np.ndarray, params: ToneParams) -> np.ndarray:
    """Applies *params* to a BGR uint8 image. The tone curve acts on BT.709
    luminance and the result is applied as a per-pixel gain to all three
    channels, so hue is preserved. Purely per-pixel: the same params give
    the same result at any resolution."""
    if params.clamped().is_identity():
        return image
    lut = _tone_lut(params)
    ramp = np.linspace(0.0, 1.0, _TONE_CURVE_SIZE, dtype=np.float64)
    gain_lut = np.where(ramp > 1e-4, lut / np.maximum(ramp, 1e-4), 1.0).astype(np.float32)
    lum = _luminance(image)
    idx = np.rint(lum * (_TONE_CURVE_SIZE - 1)).astype(np.int32)
    gain = gain_lut[idx]
    # Pure black has no hue to preserve: lift it by the curve directly.
    black_lift = np.float32(lut[0] * 255.0)
    out = image.astype(np.float32) * gain[:, :, None]
    if black_lift > 0:
        out = np.maximum(out, black_lift)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def propose_tone(image: np.ndarray) -> ToneParams:
    """Suggested ("Auto") tone settings for a BGR uint8 image.

    Expresses what the established automatic corrections — auto_level()
    followed by auto_brightness() — would do to this photo in terms of the
    four sliders: both are run on the 512px analysis view, the resulting
    input -> output luminance mapping is binned, and highlights/midtones/
    shadows/contrast are least-squares fitted (Levenberg-Marquardt) so that
    tone_curve() reproduces it, weighted by how much of the (centre-
    weighted) frame sits in each bin. A light pull toward zero keeps tones
    the photo barely uses from producing wild values.
    """
    small = _analysis_view(image)
    lum_in = _luminance(small).astype(np.float64)
    lum_out = _luminance(auto_brightness(auto_level(small))).astype(np.float64)
    weights = _center_weights(*lum_in.shape[:2]).astype(np.float64)

    idx = np.minimum((lum_in * _PROPOSE_BINS).astype(np.int64), _PROPOSE_BINS - 1).ravel()
    w_flat = weights.ravel()
    wsum = np.bincount(idx, w_flat, _PROPOSE_BINS)
    used = wsum > 0
    x = np.bincount(idx, w_flat * lum_in.ravel(), _PROPOSE_BINS)[used] / wsum[used]
    y = np.bincount(idx, w_flat * lum_out.ravel(), _PROPOSE_BINS)[used] / wsum[used]
    w = wsum[used] / wsum.sum() + _PROPOSE_BIN_FLOOR / _PROPOSE_BINS
    sqrt_w = np.sqrt(w)

    # Solved in a scaled space where every component spans roughly +-2.
    scale = np.array([1.0, 1.0, 1.0, TONE_CONTRAST_LIMIT / TONE_EV_LIMIT])
    lower = np.array([-TONE_EV_LIMIT] * 3 + [-TONE_EV_LIMIT])
    upper = np.array([_PROPOSE_HIGHLIGHT_MAX_EV, TONE_EV_LIMIT, TONE_EV_LIMIT, TONE_EV_LIMIT])

    def residuals(p: np.ndarray) -> np.ndarray:
        h, m, s, c = p * scale
        fit = tone_curve(ToneParams(highlights=h, midtones=m, shadows=s, contrast=c), x)
        return np.concatenate([sqrt_w * (fit - y), _PROPOSE_RIDGE * p])

    p = np.zeros(4)
    r = residuals(p)
    cost = float(r @ r)
    damping = 1e-3
    for _ in range(_PROPOSE_ITERATIONS):
        jac = np.empty((r.size, 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = 1e-4
            jac[:, k] = (residuals(p + dp) - r) / 1e-4
        jtj = jac.T @ jac
        step = np.linalg.solve(jtj + damping * np.diag(np.diag(jtj) + 1e-9), -jac.T @ r)
        candidate = np.clip(p + step, lower, upper)
        r_new = residuals(candidate)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            p, r, cost = candidate, r_new, cost_new
            damping /= 3.0
            if np.abs(step).max() < 1e-4:
                break
        else:
            damping *= 4.0

    h, m, s, c = p * scale
    ev = {"highlights": h, "midtones": m, "shadows": s}
    ev = {k: (0.0 if abs(v) < _PROPOSE_DEADBAND_EV else round(float(v), 1)) for k, v in ev.items()}
    params = ToneParams(contrast=float(round(c)) + 0.0, **ev)
    log.debug("tone: proposed %s", params)
    return params


# ---------------------------------------------------------------------------
# AI sharpening: two-tier dispatch
# ---------------------------------------------------------------------------

def _patch_basicsr_torchvision_compat() -> None:
    """basicsr imports ``torchvision.transforms.functional_tensor``, removed in
    torchvision >= 0.17 (this repo pins 0.17.2). Must run before ANY
    basicsr/realesrgan import."""
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    import torchvision.transforms.functional as _functional
    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = _functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _ensure_sharpen_weights(spec: SharpenModelSpec) -> str:
    """Download *spec*'s weight file up front, on one thread, so parallel
    processors can't race each other into a half-written checkpoint."""
    _patch_basicsr_torchvision_compat()
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        url=spec.model_url, model_dir=str(_REPO_ROOT), progress=True,
        file_name=spec.model_name,
    )


def _ensure_gfpgan_weights() -> str:
    """Download the GFPGAN checkpoint up front (--face-enhance only)."""
    _patch_basicsr_torchvision_compat()
    from basicsr.utils.download_util import load_file_from_url

    return load_file_from_url(
        url=_GFPGAN_MODEL_URL, model_dir=str(_REPO_ROOT), progress=True,
        file_name=_GFPGAN_MODEL_NAME,
    )


def _pick_tile_size(image: np.ndarray, device: str, spec: SharpenModelSpec) -> int:
    """Return the RealESRGANer tile size to use for *image* on *device*.

    GPU: sized from CURRENTLY FREE VRAM (queried live, so it adapts to other
    processes/jobs sharing the card) rather than a fixed pixel threshold — a
    whole image is processed in one shot (tile=0, fastest, no tile seams)
    whenever it fits the VRAM budget, only falling back to the largest tile
    that does fit otherwise. CPU has no cheap equivalent of
    torch.cuda.mem_get_info, so it keeps the old fixed-threshold behaviour.
    """
    h, w = image.shape[:2]
    if not device.startswith("cuda"):
        long_edge = max(h, w)
        return _AUTO_TILE_SIZE_PX if long_edge > _AUTO_TILE_THRESHOLD_PX else 0

    import torch
    free_bytes, _total = torch.cuda.mem_get_info(device)
    budget = free_bytes * _VRAM_SAFETY_FACTOR
    bytes_per_px = _VRAM_BYTES_PER_PIXEL[spec.kind]
    if w * h * bytes_per_px <= budget:
        return 0
    tile_edge = int((budget / bytes_per_px) ** 0.5)
    tile_edge = max(_TILE_MIN_PX, min(_TILE_MAX_PX, tile_edge))
    return tile_edge - (tile_edge % 32)


class SharpenProcessor(ABC):
    """A single unit of sharpening capacity.

    Implementations own whatever resources they need exclusively, so the
    dispatcher never has to lock around a shared model. A web-service backed
    processor can be added later behind this same interface.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """Return *image* sharpened, at the same resolution."""

    @property
    @abstractmethod
    def label(self) -> str:
        ...


class LocalSharpenProcessor(SharpenProcessor):
    """Real-ESRGAN, optionally with a GFPGAN face-enhance pass, pinned to one
    local device.

    Models are built lazily and are never shared with another processor, which
    is what makes inference thread-safe without locking.
    """

    # Construction is serialised across processors so two processors building
    # at once can't race into a half-written checkpoint read.
    _build_lock = threading.Lock()

    def __init__(
        self, device: str, realesrgan_path: str, spec: SharpenModelSpec,
        gfpgan_path: str | None = None,
    ) -> None:
        self._device = device
        self._realesrgan_path = realesrgan_path
        self._spec = spec
        self._gfpgan_path = gfpgan_path
        self._upsampler = None
        self._face_enhancer = None

    @property
    def label(self) -> str:
        return self._device

    def _build(self):
        import torch
        from realesrgan import RealESRGANer

        device = torch.device(self._device)
        spec = self._spec
        if spec.kind == "compact":
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            net = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=spec.num_feat,
                num_conv=spec.num_conv, upscale=4, act_type="prelu",
            )
        else:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=spec.num_feat,
                num_block=spec.num_block, num_grow_ch=spec.num_grow_ch, scale=4,
            )
        upsampler = RealESRGANer(
            scale=4,
            model_path=self._realesrgan_path,
            model=net,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=self._device.startswith("cuda"),
            device=device,
        )
        if self._gfpgan_path is None:
            return upsampler, None

        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path=self._gfpgan_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device,
        )
        return upsampler, face_enhancer

    def process(self, image: np.ndarray) -> np.ndarray:
        if self._upsampler is None:
            with LocalSharpenProcessor._build_lock:
                if self._upsampler is None:
                    log.info("[%s] loading Real-ESRGAN (%s)%s …",
                             self._device, self._spec.label,
                             " + GFPGAN face-enhance" if self._gfpgan_path else "")
                    self._upsampler, self._face_enhancer = self._build()

        # RealESRGANer's tiling knob is `tile_size`, not `tile` (its constructor
        # arg is named `tile` but stores it as `self.tile_size`) -- setting the
        # wrong attribute here would silently no-op and always process the
        # whole image untiled, regardless of size.
        self._upsampler.tile_size = _pick_tile_size(image, self._device, self._spec)
        if self._face_enhancer is not None:
            _, _, output = self._face_enhancer.enhance(
                image, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            output, _ = self._upsampler.enhance(image, outscale=1)
        if output.shape[:2] != image.shape[:2]:
            # outscale=1 should already guarantee this; resize defensively so a
            # library-version quirk can never silently change dimensions.
            output = cv2.resize(
                output, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4
            )
        return output


@dataclass
class _SharpenJob:
    image: np.ndarray
    done: threading.Event = field(default_factory=threading.Event)
    result: np.ndarray | None = None
    error: BaseException | None = None


class SharpenDispatcher:
    """Blocking work queue in front of a pool of :class:`SharpenProcessor`.

    File workers submit and block; processors run independently. One CPU
    processor exists as spillover capacity and only takes a job when every GPU
    processor is already busy -- CPU inference is roughly 30-40x slower, so an
    ungated CPU worker would grab jobs and straggle at the end of a batch. With
    no GPU present the "all GPUs busy" test is vacuously true, so the same rule
    makes the CPU processor the sole worker.
    """

    _CPU_POLL_SECONDS = 0.05

    def __init__(
        self, model_spec: SharpenModelSpec | None = None, face_enhance: bool = False,
    ) -> None:
        self._model_spec = model_spec or SHARPEN_MODELS[DEFAULT_SHARPEN_MODEL]
        self._face_enhance = face_enhance
        self._queue: queue.Queue[_SharpenJob | None] = queue.Queue()
        self._lock = threading.Lock()
        self._gpu_busy = 0
        self._gpu_count = 0
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> None:
        log.info("Sharpen model: %s%s", self._model_spec.label,
                  " + GFPGAN face-enhance" if self._face_enhance else "")
        realesrgan_path = _ensure_sharpen_weights(self._model_spec)
        gfpgan_path = _ensure_gfpgan_weights() if self._face_enhance else None

        import torch
        self._gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        for index in range(self._gpu_count):
            device = f"cuda:{index}"
            log.info("Sharpen processor: %s (%s)", device, torch.cuda.get_device_name(index))
            self._spawn(
                LocalSharpenProcessor(device, realesrgan_path, self._model_spec, gfpgan_path),
                self._gpu_loop,
                f"sharpen-{device}",
            )

        log.info(
            "Sharpen processor: cpu (%s)",
            "sole worker" if self._gpu_count == 0 else "spillover only",
        )
        self._spawn(
            LocalSharpenProcessor("cpu", realesrgan_path, self._model_spec, gfpgan_path),
            self._cpu_loop,
            "sharpen-cpu",
        )

    def _spawn(self, processor: SharpenProcessor, loop, name: str) -> None:
        thread = threading.Thread(target=loop, args=(processor,), name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def submit(self, image: np.ndarray) -> np.ndarray:
        job = _SharpenJob(image=image)
        self._queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        assert job.result is not None
        return job.result

    def shutdown(self) -> None:
        self._stop.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join(timeout=10.0)

    @staticmethod
    def _run(processor: SharpenProcessor, job: _SharpenJob) -> None:
        try:
            job.result = processor.process(job.image)
        except BaseException as exc:  # surfaced to the waiting file worker
            job.error = exc
        finally:
            job.done.set()

    def _gpu_loop(self, processor: SharpenProcessor) -> None:
        while not self._stop.is_set():
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                self._gpu_busy += 1
            try:
                self._run(processor, job)
            finally:
                with self._lock:
                    self._gpu_busy -= 1

    def _cpu_loop(self, processor: SharpenProcessor) -> None:
        while not self._stop.is_set():
            with self._lock:
                all_gpus_busy = self._gpu_busy >= self._gpu_count
            if not all_gpus_busy:
                self._stop.wait(self._CPU_POLL_SECONDS)
                continue
            try:
                job = self._queue.get(timeout=self._CPU_POLL_SECONDS)
            except queue.Empty:
                continue
            if job is None:
                return
            self._run(processor, job)


# ---------------------------------------------------------------------------
# Body-only sharpening
# ---------------------------------------------------------------------------

def _box_clip_mask(
    shape_hw: tuple[int, int],
    box_px: tuple[float, float, float, float],
) -> np.ndarray:
    """Float32 [0, 1] mask of *shape_hw* that is 1 inside *box_px*
    (x1, y1, x2, y2 in the mask's own pixel coordinates), grown by
    BODY_SHARPEN_CLIP_MARGIN_PCT and feathered by BODY_SHARPEN_CLIP_FEATHER_PX."""
    height, width = shape_hw
    x1, y1, x2, y2 = box_px
    margin_x = (x2 - x1) * BODY_SHARPEN_CLIP_MARGIN_PCT
    margin_y = (y2 - y1) * BODY_SHARPEN_CLIP_MARGIN_PCT
    cx1 = max(0, int(round(x1 - margin_x)))
    cy1 = max(0, int(round(y1 - margin_y)))
    cx2 = min(width, int(round(x2 + margin_x)))
    cy2 = min(height, int(round(y2 + margin_y)))
    clip = np.zeros((height, width), dtype=np.float32)
    clip[cy1:cy2, cx1:cx2] = 1.0
    if BODY_SHARPEN_CLIP_FEATHER_PX > 0:
        clip = cv2.GaussianBlur(clip, (0, 0), sigmaX=BODY_SHARPEN_CLIP_FEATHER_PX)
    return clip


def _select_body_component(
    mask: np.ndarray,
    keypoints_px: list[tuple[float, float, float]],
) -> tuple[np.ndarray, dict | None]:
    """When a body's own 17 COCO keypoints (*keypoints_px*: (x, y,
    confidence) in the mask's own pixel coordinates) land on more than one
    disconnected blob in the segmentation *mask* — i.e. a neighbouring
    player's blob also falls inside this body's padded crop, touching or
    overlapping the rectangular box-clip without actually being this body's
    silhouette — keeps only the blob(s) the keypoints sit on and zeroes the
    rest, so the neighbour doesn't get sharpened too. This is a much
    tighter, silhouette-shaped alternative to the box-clip above (which is
    still applied afterwards regardless, as a backstop).

    Falls back to returning *mask* unchanged (second element None) when
    there are no confident-enough keypoints to vote with, or when none of
    them land on any foreground blob (pose missed/occluded) — the box-clip
    is still the fallback in both cases, matching the behaviour before this
    function existed. Returns (new_mask, info) where info is a dict with
    vote-count details for logging, or None when the mask was left as-is.
    """
    confident = [
        (x, y) for x, y, conf in keypoints_px if conf >= BODY_SHARPEN_KEYPOINT_CONF_MIN
    ]
    if not confident:
        return mask, None
    binary = (mask > 0.5).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary)
    if num_labels <= 1:  # nothing but background — nothing to select between
        return mask, None
    height, width = labels.shape
    voted: set[int] = set()
    for x, y in confident:
        px, py = int(round(x)), int(round(y))
        if 0 <= px < width and 0 <= py < height:
            label = int(labels[py, px])
            if label != 0:
                voted.add(label)
    if not voted:
        return mask, None
    keep = np.isin(labels, list(voted))
    info = {
        "n_keypoints": len(confident), "n_components": num_labels - 1, "n_kept": len(voted),
    }
    return mask * keep.astype(np.float32), info


def _signed_depth(shape: np.ndarray, step: float) -> np.ndarray:
    """Signed distance (crop pixels; negative inside) to the filled region of
    the uint8 0/1 grid *shape*; +inf everywhere if it is empty."""
    if not shape.any():
        return np.full(shape.shape, np.inf, dtype=np.float32)
    inside = cv2.distanceTransform(shape, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - shape, cv2.DIST_L2, 3)
    return (outside - inside) * np.float32(step)


def _body_depth(
    grid_hw: tuple[int, int],
    step: float,
    box_px: tuple[float, float, float, float],
    keypoints_px: list[tuple[float, float, float]],
) -> tuple[np.ndarray, bool]:
    """(depth, modelled): a float32 field over the ownership grid giving how
    deep inside this player's body each cell is (negative = inside, in crop
    pixels; positive = outside), and whether it came from their keypoints.

    The body is modelled from the player's COCO keypoints as a head circle,
    a filled shoulders-hips torso and thick limbs (sizes scale with the box
    height — see _BODY_* constants); heads count _BODY_HEAD_BONUS deeper.
    Parts whose keypoints aren't confident are skipped; a player with no
    usable part at all is an ellipse inscribed in their box (modelled=False,
    so callers shouldn't trust it to rule pixels out)."""
    gh, gw = grid_hw
    x1, y1, x2, y2 = box_px
    height = max(y2 - y1, 1.0)
    ok = [conf >= BODY_SHARPEN_KEYPOINT_CONF_MIN for _x, _y, conf in keypoints_px] if len(keypoints_px) == 17 else []
    g = lambda i: (int(round(keypoints_px[i][0] / step)), int(round(keypoints_px[i][1] / step)))  # noqa: E731

    def limbs(bones, radius):
        shape = np.zeros((gh, gw), np.uint8)
        thickness = max(1, int(round(2 * radius * height / step)))
        for a, b in bones:
            if ok and ok[a] and ok[b]:
                cv2.line(shape, g(a), g(b), 1, thickness)
        return shape

    torso = np.zeros((gh, gw), np.uint8)
    if ok and all(ok[i] for i in (5, 6, 11, 12)):
        cv2.fillPoly(torso, [np.array([g(5), g(6), g(12), g(11)], np.int32)], 1)
        cv2.polylines(torso, [np.array([g(5), g(6), g(12), g(11)], np.int32)], True, 1,
                      max(1, int(round(2 * _BODY_TORSO_MARGIN * height / step))))
    head = np.zeros((gh, gw), np.uint8)
    head_pts = [keypoints_px[i][:2] for i in range(5) if ok and ok[i]]
    if head_pts:
        cx, cy = np.mean(head_pts, axis=0)
        cv2.circle(head, (int(round(cx / step)), int(round(cy / step))),
                   max(1, int(round(_BODY_HEAD_RADIUS * height / step))), 1, -1)

    parts = [
        _signed_depth(limbs(_ARM_BONES, _BODY_ARM_RADIUS), step),
        _signed_depth(limbs(_LEG_BONES, _BODY_LEG_RADIUS), step),
        _signed_depth(torso, step),
        _signed_depth(head, step) - np.float32(_BODY_HEAD_BONUS * height),
    ]
    depth = np.minimum.reduce(parts)
    if np.isinf(depth).all():
        fallback = np.zeros((gh, gw), np.uint8)
        cv2.ellipse(fallback, (int(round((x1 + x2) / 2 / step)), int(round((y1 + y2) / 2 / step))),
                    (max(1, int(round((x2 - x1) * 0.35 / step))), max(1, int(round(height * 0.45 / step)))),
                    0, 0, 360, 1, -1)
        return _signed_depth(fallback, step), False
    return depth, True


def _same_person(
    keypoints_a: list[tuple[float, float, float]],
    keypoints_b: list[tuple[float, float, float]],
    height: float,
) -> bool:
    """True if two detections' keypoints coincide (mean distance over the
    keypoints both are confident about under _BODY_DUPLICATE_DISTANCE of
    *height*, with at least 5 of them) — one person detected twice."""
    if len(keypoints_a) != 17 or len(keypoints_b) != 17:
        return False
    both = [(a, b) for a, b in zip(keypoints_a, keypoints_b)
            if a[2] >= BODY_SHARPEN_KEYPOINT_CONF_MIN and b[2] >= BODY_SHARPEN_KEYPOINT_CONF_MIN]
    if len(both) < 5:
        return False
    mean = float(np.mean([math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in both]))
    return mean < _BODY_DUPLICATE_DISTANCE * height


def _protected_keep_mask(
    shape_hw: tuple[int, int],
    target_box_px: tuple[float, float, float, float],
    target_keypoints_px: list[tuple[float, float, float]],
    protected_px: list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]],
) -> np.ndarray | None:
    """Float32 [0, 1] mask of *shape_hw* that is 0 wherever sharpening the
    target would also sharpen someone else, 1 elsewhere; None if nothing is
    excluded. All coordinates are in the crop's own pixels.

    The segmentation mask can't tell touching people apart (nor the ball,
    nor a merged neighbour's leg from the target's), so the target's own
    body model (see _body_depth()) decides:
      * pixels more than _BODY_ALLOWED_MARGIN of the box height away from it
        are dropped (skipped when the target has no usable keypoints);
      * pixels clearly deeper inside a protected player's body than inside
        the target's (by _BODY_TIE_MARGIN) are dropped — comparing depth in
        plain pixels means the larger, usually nearer and so occluding,
        player wins an overlap;
      * a protected detection that is the same person as the target is
        ignored."""
    height, width = shape_hw
    step = max(1, int(math.ceil(max(height, width) / _PROTECT_GRID_LONG_EDGE)))
    grid = (max(1, math.ceil(height / step)), max(1, math.ceil(width / step)))
    target_height = max(target_box_px[3] - target_box_px[1], 1.0)
    target_depth, modelled = _body_depth(grid, step, target_box_px, target_keypoints_px)
    keep = (target_depth < _BODY_ALLOWED_MARGIN * target_height) if modelled else np.ones(grid, dtype=bool)
    for box_px, keypoints_px in protected_px:
        if _same_person(target_keypoints_px, keypoints_px, target_height):
            continue
        depth, _ = _body_depth(grid, step, box_px, keypoints_px)
        keep &= ~((depth < 0) & (depth < target_depth - _BODY_TIE_MARGIN * target_height))
    if keep.all():
        return None
    keep_full = cv2.resize(keep.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    if BODY_SHARPEN_CLIP_FEATHER_PX > 0:
        keep_full = cv2.GaussianBlur(keep_full, (0, 0), sigmaX=BODY_SHARPEN_CLIP_FEATHER_PX)
    return keep_full


def sharpen_bodies(
    image: np.ndarray,
    boxes_norm: list[tuple[float, float, float, float]],
    dispatcher: SharpenDispatcher,
    label: str = "",
    keypoints_norm: list[list[tuple[float, float, float]]] | None = None,
    protected_norm: list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]] | None = None,
    debug_layers: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Sharpen only the given players, leaving the rest of *image*
    pixel-identical to the source.

    For each normalised (x1, y1, x2, y2) box: pad it for context, crop,
    run the crop through Real-ESRGAN (via *dispatcher* — the same model
    used for whole-frame sharpening, just on a much smaller input), get a
    person-confidence mask for the crop from the segmentation backend (see
    algo/segmentation.py), narrow that mask down to just this player's own
    silhouette (using their 17 COCO keypoints if *keypoints_norm* supplied
    them — see _select_body_component()), clip it to the player's own box so
    other people in the padded crop stay untouched, and alpha-blend the
    sharpened crop over the original crop using it before pasting it back.
    Overlapping boxes are blended in order, each on top of whatever is
    already there.

    The caller decides who is sharpened: culling_app.py only passes boxes of
    the players the user selected in edit mode (see its
    _selected_body_entries()).

    *keypoints_norm*, if given, must be the same length as *boxes_norm*:
    keypoints_norm[i] is boxes_norm[i]'s own normalised (x, y, confidence)
    17 COCO keypoints (empty list if that body has none). None, or an empty
    list for a given box, just skips keypoint-assisted component selection
    for it — the box-clip alone is still applied.

    *debug_layers*, if given (the save_silhouette debug feature), is filled
    with two full-frame layers, composited across every sharpened box:
    "silhouette" (float32 [0, 1]: the alpha actually used for blending —
    after keypoint blob selection, unselected-player exclusion, feathering
    and the box clip) and "sharpened" (uint8 BGR: *image* with every
    sharpened padded crop pasted in unmasked — the layer that silhouette
    blends over *image* to give the result).

    *protected_norm*, if given, lists players that must NOT be sharpened
    as (normalised box, normalised keypoints) pairs. Whenever any overlap a
    target's crop, the target's mask is also limited to the target's own
    keypoint-based body shape and loses the pixels a protected player
    clearly owns (see _protected_keep_mask()) — so an unselected neighbour
    touching or standing in front of/behind a selected player, and anything
    merged into the same segmentation blob (a leg, the ball), stays
    untouched, while a second detection of the same player is ignored.

    Logs every box it was given and, per box, whether segmentation worked
    (and how much of the crop it marked as person) or why the box was skipped.
    *label* (usually the photo's file name) prefixes those lines.
    """
    from algo.segmentation import DEFAULT_SEGMENTER_BACKEND, get_segmenter

    segmenter = get_segmenter(DEFAULT_SEGMENTER_BACKEND)
    img_h, img_w = image.shape[:2]
    result = image.copy()
    prefix = f"[{label}] " if label else ""
    total = len(boxes_norm)
    sharpened = 0

    log.info("%sBody sharpen: %d box(es) passed in (image %dx%d, segmenter=%s)",
             prefix, total, img_w, img_h, DEFAULT_SEGMENTER_BACKEND)

    for index, (x1, y1, x2, y2) in enumerate(boxes_norm, start=1):
        box_w_px = (x2 - x1) * img_w
        box_h_px = (y2 - y1) * img_h
        pad_x = box_w_px * BODY_SHARPEN_PADDING_PCT
        pad_y = box_h_px * BODY_SHARPEN_PADDING_PCT
        px1 = max(0, int(round(x1 * img_w - pad_x)))
        py1 = max(0, int(round(y1 * img_h - pad_y)))
        px2 = min(img_w, int(round(x2 * img_w + pad_x)))
        py2 = min(img_h, int(round(y2 * img_h + pad_y)))
        log.info("%s  box %d/%d: norm=(%.3f, %.3f, %.3f, %.3f) body_px=(%d, %d)-(%d, %d) "
                 "padded_crop_px=(%d, %d)-(%d, %d) [%dx%d]",
                 prefix, index, total, x1, y1, x2, y2,
                 round(x1 * img_w), round(y1 * img_h), round(x2 * img_w), round(y2 * img_h),
                 px1, py1, px2, py2, px2 - px1, py2 - py1)
        if px2 - px1 < 2 or py2 - py1 < 2:
            log.warning("%s  box %d/%d: skipped - degenerate (outside the frame or a sliver)",
                        prefix, index, total)
            continue

        crop = image[py1:py2, px1:px2]
        # Segment first: it is cheap, and a failed mask means the (much more
        # expensive) sharpen pass for this box would be wasted.
        try:
            mask = segmenter.segment(crop)
        except Exception as exc:
            log.warning("%s  box %d/%d: segmentation FAILED (%s: %s) - box left untouched",
                        prefix, index, total, type(exc).__name__, exc)
            continue
        if mask.shape[:2] != crop.shape[:2]:
            mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)

        kps =  keypoints_norm[index - 1] if keypoints_norm and index - 1 < len(keypoints_norm) else []
        keypoints_px = [(kx * img_w - px1, ky * img_h - py1, kconf) for kx, ky, kconf in kps]
        mask, component_info = _select_body_component(mask, keypoints_px)
        if component_info is not None:
            log.info("%s  box %d/%d: keypoint-selected %d/%d segmentation blob(s) using %d "
                     "confident keypoint(s)",
                     prefix, index, total, component_info["n_kept"], component_info["n_components"],
                     component_info["n_keypoints"])

        if protected_norm:
            to_crop = lambda bx: (bx[0] * img_w - px1, bx[1] * img_h - py1, bx[2] * img_w - px1, bx[3] * img_h - py1)  # noqa: E731
            crop_h, crop_w = crop.shape[:2]
            protected_px = []
            for p_box, p_kps in protected_norm:
                cb = to_crop(p_box)
                if cb[2] <= 0 or cb[3] <= 0 or cb[0] >= crop_w or cb[1] >= crop_h:
                    continue
                protected_px.append((cb, [(kx * img_w - px1, ky * img_h - py1, kc) for kx, ky, kc in p_kps]))
            if protected_px:
                keep = _protected_keep_mask(crop.shape[:2], to_crop((x1, y1, x2, y2)), keypoints_px, protected_px)
                if keep is not None:
                    before = float((mask > 0.5).mean())
                    mask = mask * keep
                    log.info("%s  box %d/%d: limited to this player's own body (%d other detection(s) "
                             "nearby); person %.1f%% -> %.1f%% of crop",
                             prefix, index, total, len(protected_px), before * 100,
                             float((mask > 0.5).mean()) * 100)

        person_coverage = float((mask > 0.5).mean())
        if person_coverage < BODY_SHARPEN_MIN_PERSON_COVERAGE:
            log.warning("%s  box %d/%d: segmentation FAILED - no person found in the crop "
                        "(%.1f%% of pixels, need >= %.1f%%) - box left untouched",
                        prefix, index, total, person_coverage * 100,
                        BODY_SHARPEN_MIN_PERSON_COVERAGE * 100)
            continue

        try:
            sharpened_crop = dispatcher.submit(crop)
        except Exception as exc:
            log.warning("%s  box %d/%d: segmentation OK (person %.1f%% of crop) but sharpening "
                        "FAILED (%s: %s) - box left untouched",
                        prefix, index, total, person_coverage * 100, type(exc).__name__, exc)
            continue

        if sharpened_crop.shape[:2] != crop.shape[:2]:
            sharpened_crop = cv2.resize(
                sharpened_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4
            )
        if BODY_SHARPEN_MASK_BLUR_PX > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=BODY_SHARPEN_MASK_BLUR_PX)
        mask = mask * _box_clip_mask(
            crop.shape[:2],
            (x1 * img_w - px1, y1 * img_h - py1, x2 * img_w - px1, y2 * img_h - py1),
        )

        alpha = np.clip(mask, 0.0, 1.0)[..., None]
        blended = sharpened_crop.astype(np.float32) * alpha + crop.astype(np.float32) * (1.0 - alpha)
        result[py1:py2, px1:px2] = np.clip(blended, 0, 255).astype(np.uint8)
        if debug_layers is not None:
            silhouette = debug_layers.setdefault("silhouette", np.zeros((img_h, img_w), dtype=np.float32))
            np.maximum(silhouette[py1:py2, px1:px2], np.clip(alpha[..., 0], 0.0, 1.0),
                       out=silhouette[py1:py2, px1:px2])
            debug_layers.setdefault("sharpened", image.copy())[py1:py2, px1:px2] = sharpened_crop
        sharpened += 1
        log.info("%s  box %d/%d: segmentation OK - person %.1f%% of crop, %.1f%% after clipping "
                 "to the body box; sharpened and blended",
                 prefix, index, total, person_coverage * 100, float((mask > 0.5).mean()) * 100)

    log.info("%sBody sharpen: %d/%d box(es) sharpened, %d skipped",
             prefix, sharpened, total, total - sharpened)
    return result


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _snap_to_multiple(value: float) -> int:
    return max(
        _GPT_IMAGE_EDGE_MULTIPLE,
        int(round(value / _GPT_IMAGE_EDGE_MULTIPLE)) * _GPT_IMAGE_EDGE_MULTIPLE,
    )


def gpt_image_size(width: int, height: int) -> tuple[int, int]:
    """Nearest size to *width* x *height* that gpt-image-2 will accept.

    Both edges must be multiples of 16, neither may exceed 3840, the long:short
    ratio may not exceed 3:1, and the total pixel count must fall between
    655,360 and 8,294,400.
    """
    aspect = width / height
    clamped = min(max(aspect, 1.0 / _GPT_IMAGE_MAX_ASPECT), _GPT_IMAGE_MAX_ASPECT)
    if abs(clamped - aspect) > 1e-6:
        log.warning(
            "Aspect ratio %.2f exceeds the model's 3:1 limit; requesting %.2f instead",
            aspect, clamped,
        )
    aspect = clamped

    pixels = float(np.clip(width * height, _GPT_IMAGE_MIN_PIXELS, _GPT_IMAGE_TARGET_PIXELS))
    for _ in range(8):
        edge_h = math.sqrt(pixels / aspect)
        edge_w = aspect * edge_h
        overshoot = max(edge_w, edge_h) / _GPT_IMAGE_MAX_EDGE
        if overshoot > 1.0:
            edge_w /= overshoot
            edge_h /= overshoot
        out_w, out_h = _snap_to_multiple(edge_w), _snap_to_multiple(edge_h)
        total = out_w * out_h
        if total < _GPT_IMAGE_MIN_PIXELS:
            pixels *= _GPT_IMAGE_MIN_PIXELS / total * 1.02
            continue
        if total > _GPT_IMAGE_MAX_PIXELS:
            pixels *= _GPT_IMAGE_MAX_PIXELS / total * 0.98
            continue
        return out_w, out_h
    return 1024, 1024


def openai_edit(
    client,
    image: np.ndarray,
    prompt: str,
    size: tuple[int, int],
) -> np.ndarray:
    """Send *image* to the OpenAI image model and return the edited result."""
    request_w, request_h = size
    # Resized before encoding so the upload stays well inside the API's size
    # limit and matches the resolution actually being asked for.
    payload = cv2.resize(image, (request_w, request_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", payload)
    if not ok:
        raise RuntimeError("Could not PNG-encode the image for upload")

    response = client.images.edit(
        model=_OPENAI_IMAGE_MODEL,
        image=("input.png", io.BytesIO(buf.tobytes()), "image/png"),
        prompt=prompt,
        size=f"{request_w}x{request_h}",
        quality=_OPENAI_QUALITY,
        output_format="png",
    )
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(response.data[0].b64_json), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if decoded is None:
        raise RuntimeError("Could not decode the image returned by the API")
    return decoded


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

@dataclass
class EditContext:
    """Services shared by every stage across every worker thread."""

    source: Path
    dispatcher: SharpenDispatcher | None = None
    openai_client: object | None = None
    # Normalised (x1, y1, x2, y2) boxes — fractions of width/height, matching
    # algo.models.Box's own convention — for the players detected in this
    # photo. Populated from --boxes-file (see main()); callers are expected to
    # list only the players that should be sharpened (culling_app.py passes
    # just the players the user selected in edit mode). None/empty means "no boxes available for
    # this image", in which case AiSharpenPipeline falls back to whole-frame
    # sharpening regardless of AI_SHARPEN_SCOPE.
    body_boxes: list[tuple[float, float, float, float]] | None = None
    # Same length/order as body_boxes when given: body_keypoints[i] is
    # body_boxes[i]'s own normalised (x, y, confidence) 17 COCO keypoints
    # (empty list for a box with none). Also populated from --boxes-file;
    # lets sharpen_bodies() tell this player's own segmentation blob apart
    # from a neighbour's (see _select_body_component()). None/empty just
    # means no keypoint-assisted selection for any box.
    body_keypoints: list[list[tuple[float, float, float]]] | None = None
    # Explicit settings for the "tone" pipeline (--highlights/--midtones/
    # --shadows/--contrast); None means "use propose_tone()'s auto values".
    tone: ToneParams | None = None
    # Players that must NOT be sharpened, as (normalised box, normalised
    # keypoints) pairs — --boxes-file entries marked "protect" (see
    # sharpen_bodies()'s protected_norm).
    protected_bodies: list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]] | None = None
    # Debug (--save-silhouette): folder to write the body-sharpen debug images
    # into (see _save_sharpen_debug()); None = don't. `original` is the photo
    # as read from disk (set only when silhouette_dir is), and
    # `silhouette_saved` is flipped once the sharpen step has written its
    # images, so process_one() knows to add the final result next to them.
    silhouette_dir: Path | None = None
    original: np.ndarray | None = None
    silhouette_saved: bool = False


class EditPipeline(ABC):
    name: str = ""
    description: str = ""  # shown in the -pipelines CLI help

    @abstractmethod
    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        ...


class TonePipeline(EditPipeline):
    name = "tone"
    description = (
        "Highlights/midtones/shadows exposure + contrast (--highlights etc.; auto if none given)"
    )

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        params = ctx.tone if ctx.tone is not None else propose_tone(image)
        log.info(
            "[%s] tone: highlights %+.2f, midtones %+.2f, shadows %+.2f EV, contrast %+.0f",
            ctx.source.name, params.highlights, params.midtones, params.shadows, params.contrast,
        )
        return apply_tone(image, params)


class LevelPipeline(EditPipeline):
    name = "level"
    description = "Auto black/white point correction"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        return auto_level(image)


class BrightnessPipeline(EditPipeline):
    name = "brightness"
    description = "Auto exposure correction (highlight-safe gamma)"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        return auto_brightness(image)


class AiSharpenPipeline(EditPipeline):
    name = "aisharpen"
    description = "Real-ESRGAN detail restoration (--sharpen-model, --face-enhance)"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        assert ctx.dispatcher is not None
        if AI_SHARPEN_SCOPE == "bodies" and ctx.body_boxes:
            debug_layers: dict[str, np.ndarray] | None = {} if ctx.silhouette_dir is not None else None
            result = sharpen_bodies(
                image, ctx.body_boxes, ctx.dispatcher, label=ctx.source.name,
                keypoints_norm=ctx.body_keypoints, protected_norm=ctx.protected_bodies,
                debug_layers=debug_layers,
            )
            if debug_layers:
                _save_sharpen_debug(debug_layers, ctx)
            elif debug_layers is not None:
                log.warning("[%s] save_silhouette: no body was sharpened, nothing to save", ctx.source.name)
            return result
        reason = (
            f"AI_SHARPEN_SCOPE={AI_SHARPEN_SCOPE!r}" if AI_SHARPEN_SCOPE != "bodies"
            else "no body boxes were passed in for this photo"
        )
        log.info("[%s] Sharpening the whole frame (%s)", ctx.source.name, reason)
        return ctx.dispatcher.submit(image)


# JPEG quality of the debug images (--save-silhouette): effectively lossless.
DEBUG_JPEG_QUALITY = 100


def _save_debug_image(image: np.ndarray, path: Path, source: Path) -> None:
    """Debug: writes a full-resolution, minimally compressed JPEG (or, for a
    .png path, an 8-bit PNG). Logs instead of raising: a debug write must
    not fail the edit."""
    ext = path.suffix.lstrip(".").lower()
    try:
        if write_image(image, path, ext, DEBUG_JPEG_QUALITY):
            log.info("[%s] save_silhouette: wrote %s", source.name, path)
    except OSError as exc:
        log.warning("[%s] save_silhouette: could not write %s: %s", source.name, path, exc)


def _save_sharpen_debug(layers: dict[str, np.ndarray], ctx: EditContext) -> None:
    """Debug (--save-silhouette): writes, into ctx.silhouette_dir as
    <source stem>_*, the original photo (_orig.jpg), the silhouette mask
    (_silhouette.png, white = sharpened), and the sharpened layer
    (_sharpened.jpg) — sharpen_bodies()'s debug_layers. The edit's final
    result (_final.jpg) is saved by process_one() once it exists."""
    directory, stem = ctx.silhouette_dir, ctx.source.stem
    if ctx.original is not None:
        _save_debug_image(ctx.original, directory / f"{stem}_orig.jpg", ctx.source)
    _save_debug_image(np.rint(layers["silhouette"] * 255.0).astype(np.uint8),
                      directory / f"{stem}_silhouette.png", ctx.source)
    _save_debug_image(layers["sharpened"], directory / f"{stem}_sharpened.jpg", ctx.source)
    ctx.silhouette_saved = True

class OpenAiAutoEditPipeline(EditPipeline):
    name = "openai_autoedit"
    description = "Generative retouch via the OpenAI image model"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        height, width = image.shape[:2]
        size = gpt_image_size(width, height)
        log.info("[%s] openai_autoedit at %dx%d", ctx.source.name, *size)
        return openai_edit(ctx.openai_client, image, _AUTOEDIT_PROMPT, size)


class OpenAiInstagramPipeline(EditPipeline):
    name = "openai_ig"
    description = "Generative heroic 4:5 Instagram reframe"

    def apply(self, image: np.ndarray, ctx: EditContext) -> np.ndarray:
        log.info("[%s] openai_ig at %dx%d", ctx.source.name, *_IG_REQUEST_SIZE)
        result = openai_edit(ctx.openai_client, image, _IG_PROMPT, _IG_REQUEST_SIZE)
        return cv2.resize(result, _IG_FINAL_SIZE, interpolation=cv2.INTER_LANCZOS4)


# Canonical execution order, independent of the order the user lists them.
# Levels sets the black/white endpoints first and brightness then corrects the
# midtone against an already-normalised histogram (Photoshop Levels semantics);
# sharpening follows all tonal work; the generative stages run last so they see
# the best possible local edit. "tone" is the manual alternative to
# level+brightness (the culling app uses it on its own) and runs first.
PIPELINE_ORDER: tuple[type[EditPipeline], ...] = (
    TonePipeline,
    LevelPipeline,
    BrightnessPipeline,
    AiSharpenPipeline,
    OpenAiAutoEditPipeline,
    OpenAiInstagramPipeline,
)

PIPELINES_BY_NAME: dict[str, type[EditPipeline]] = {cls.name: cls for cls in PIPELINE_ORDER}

# Never pulled in by "all": the openai_* stages are billed, and "tone" would
# double up with level+brightness.
_EXCLUDED_FROM_ALL: frozenset[str] = frozenset({"tone"})


def resolve_pipelines(tokens: list[str]) -> list[EditPipeline]:
    """Expand and de-duplicate user pipeline names into ordered instances."""
    requested: set[str] = set()
    for token in tokens:
        for name in token.split():
            key = name.strip().lower()
            if not key:
                continue
            if key == "all":
                requested.update(
                    n for n in PIPELINES_BY_NAME
                    if not n.startswith("openai") and n not in _EXCLUDED_FROM_ALL
                )
            elif key in PIPELINES_BY_NAME:
                requested.add(key)
            else:
                log.warning("Unknown pipeline '%s', ignoring.", name)
    return [cls() for cls in PIPELINE_ORDER if cls.name in requested]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def output_path_for(source: Path, output_dir: Path, ext: str) -> Path:
    """Timestamped auto-name for *source*'s output, disambiguated with a
    '_n' suffix (first available n, starting at 1) if that exact path is
    already taken — e.g. two sources sharing a stem processed in the same
    second. An explicit --out-file path bypasses this entirely and always
    overwrites (handled by the caller/write_image)."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{source.stem}_processed_{stamp}"
    candidate = output_dir / f"{base}.{ext}"
    n = 1
    while candidate.exists():
        candidate = output_dir / f"{base}_{n}.{ext}"
        n += 1
    return candidate


def process_one(
    source: Path,
    pipelines: list[EditPipeline],
    output_dir: Path,
    ext: str,
    compression: int,
    dispatcher: SharpenDispatcher | None,
    openai_client,
    out_file: Path | None = None,
    body_boxes: list[tuple[float, float, float, float]] | None = None,
    body_keypoints: list[list[tuple[float, float, float]]] | None = None,
    tone: ToneParams | None = None,
    protected_bodies: list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]] | None = None,
    silhouette_dir: Path | None = None,
    crop: tuple[int, int, int, int] | None = None,
) -> Path | None:
    image = read_image(source)
    if image is None:
        return None
    log.info("[%s] %dx%d", source.name, image.shape[1], image.shape[0])
    source_size = (image.shape[1], image.shape[0])

    ctx = EditContext(
        source=source, dispatcher=dispatcher, openai_client=openai_client,
        body_boxes=body_boxes, body_keypoints=body_keypoints, tone=tone,
        protected_bodies=protected_bodies, silhouette_dir=silhouette_dir,
        # No pipeline mutates its input, so the untouched original is kept by reference.
        original=image if silhouette_dir is not None else None,
    )
    for pipeline in pipelines:
        try:
            image = pipeline.apply(image, ctx)
        except Exception as exc:
            log.error("[%s] pipeline '%s' failed: %s", source.name, pipeline.name, exc)
            return None

    # Cropping runs last so every pipeline (notably body-only sharpening,
    # whose --boxes-file boxes are relative to the full source frame) sees
    # the whole photo.
    if crop is not None:
        image = crop_image(image, crop, source_size)
        log.info("[%s] cropped to %dx%d", source.name, image.shape[1], image.shape[0])

    # out_file may equal source (re-editing an already-edited image in
    # place) — safe since read_image() fully decodes into memory up front,
    # nothing streams from disk during pipeline processing.
    destination = out_file if out_file is not None else output_path_for(source, output_dir, ext)
    if not write_image(image, destination, ext, compression):
        return None
    if ctx.silhouette_saved:
        _save_debug_image(image, silhouette_dir / f"{source.stem}_final.jpg", source)
    log.info("[%s] wrote %s (%dx%d)", source.name, destination, image.shape[1], image.shape[0])
    return destination


def _parse_boxes_file(raw: dict) -> tuple[
    dict[str, list[tuple[float, float, float, float]]],
    dict[str, list[list[tuple[float, float, float]]]],
    dict[str, list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]]],
]:
    """Parses --boxes-file's JSON (see its --help) into (boxes_by_name,
    keypoints_by_name, protected_by_name), all keyed by file name.
    boxes_by_name/keypoints_by_name are same-length, same-order lists of the
    bodies to sharpen — index i in each belongs to the same body. Each raw
    entry is either a plain [x1, y1, x2, y2] box (no keypoints) or a
    {"box": [...], "keypoints": [[x, y, conf], ...]} object (see
    culling_app.py's _selected_body_entries(), the one real caller of this
    format); an object with "protect": true is instead a body that must not
    be sharpened and goes to protected_by_name as (box, keypoints)."""
    boxes_by_name: dict[str, list[tuple[float, float, float, float]]] = {}
    keypoints_by_name: dict[str, list[list[tuple[float, float, float]]]] = {}
    protected_by_name: dict[str, list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]]] = {}
    for name, entries in raw.items():
        boxes: list[tuple[float, float, float, float]] = []
        keypoints: list[list[tuple[float, float, float]]] = []
        protected: list[tuple[tuple[float, float, float, float], list[tuple[float, float, float]]]] = []
        for entry in entries:
            if isinstance(entry, dict):
                kps = [tuple(kp) for kp in entry.get("keypoints", [])]
                if entry.get("protect"):
                    protected.append((tuple(entry["box"]), kps))
                    continue
                boxes.append(tuple(entry["box"]))
                keypoints.append(kps)
            else:
                boxes.append(tuple(entry))
                keypoints.append([])
        boxes_by_name[name] = boxes
        keypoints_by_name[name] = keypoints
        protected_by_name[name] = protected
    return boxes_by_name, keypoints_by_name, protected_by_name


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputPath", type=Path,
        help="An image, a directory of images (non-recursive), or a newline-delimited list file.",
    )
    parser.add_argument(
        "-pipelines", "--pipelines", nargs="*", default=["all"], metavar="NAME",
        help=(
            "Space-separated pipelines to run: "
            + "; ".join(f"{name} ({cls.description})" for name, cls in PIPELINES_BY_NAME.items())
            + ". 'all' (default) selects every non-openai pipeline. Unknown names are "
              "warned about and ignored; order is decided by this script, not by you."
        ),
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR, metavar="DIR",
        help=f"Output directory (default: ./{DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "-ext", "--ext", default="jpg", choices=sorted(OUTPUT_EXTENSIONS),
        help="Output file format (default: jpg).",
    )
    parser.add_argument(
        "-compression", "--compression", type=int, default=DEFAULT_COMPRESSION, metavar="1-100",
        help=f"JPEG quality when -ext is jpg (default: {DEFAULT_COMPRESSION}). Ignored for png.",
    )
    parser.add_argument(
        "--openaikey", default=None, metavar="KEY",
        help="OpenAI API key. Without it, the openai_* pipelines are skipped.",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, metavar="N",
        help=(
            f"Images to process concurrently (default: {DEFAULT_WORKERS}). Each image still "
            "runs its pipelines strictly in sequence."
        ),
    )
    parser.add_argument(
        "--out-file", type=Path, default=None, metavar="PATH",
        help=(
            "Exact output file path, overriding -o/-ext's timestamped naming. Only valid "
            "with a single input image. May equal inputPath to overwrite it in place "
            "(re-editing an already-edited image). If this path already exists it is "
            "overwritten; auto-named output (no --out-file) instead gets a '_n' suffix."
        ),
    )
    parser.add_argument(
        "--sharpen-model", choices=sorted(SHARPEN_MODELS), default=DEFAULT_SHARPEN_MODEL,
        metavar="NAME",
        help=(
            "Real-ESRGAN checkpoint used by the aisharpen pipeline, to A/B test detail "
            f"restoration strength (default: {DEFAULT_SHARPEN_MODEL}). Choices: "
            + ", ".join(f"{k} ({v.label})" for k, v in SHARPEN_MODELS.items())
        ),
    )
    parser.add_argument(
        "--face-enhance", action="store_true",
        help="Run a GFPGAN face-restoration pass after Real-ESRGAN in the aisharpen "
             "pipeline (off by default).",
    )
    parser.add_argument(
        "--boxes-file", type=Path, default=None, metavar="PATH",
        help=(
            "JSON file mapping each input file's NAME (e.g. \"photo.jpg\", not a full "
            "path) to a list of body entries for the aisharpen pipeline's body-only mode "
            "(see AI_SHARPEN_SCOPE in this file) to sharpen+blend just the detected "
            "players instead of the whole frame. Each entry is either a plain "
            "[x1, y1, x2, y2] normalised box (fractions of width/height), or a "
            "{\"box\": [x1, y1, x2, y2], \"keypoints\": [[x, y, conf], ...]} object "
            "additionally carrying that body's 17 COCO keypoints (also normalised; "
            "conf in [0, 1]) so segmentation can tell this player's own silhouette apart "
            "from a neighbour's (see _select_body_component in this file) — keypoints are "
            "optional per entry. An object with \"protect\": true is a body that must NOT "
            "be sharpened: where it overlaps a sharpened body, its own pixels are left "
            "untouched. Images with no entry (or no --boxes-file at all) fall "
            "back to whole-frame sharpening."
        ),
    )
    for zone in ("highlights", "midtones", "shadows"):
        parser.add_argument(
            f"--{zone}", type=float, default=None, metavar="EV",
            help=(
                f"tone pipeline: {zone} exposure in EV (-{TONE_EV_LIMIT:g}..+{TONE_EV_LIMIT:g}). "
                "If none of --highlights/--midtones/--shadows/--contrast is given, the tone "
                "pipeline uses its own auto values; otherwise unspecified ones are 0."
            ),
        )
    parser.add_argument(
        "--contrast", type=float, default=None, metavar="AMOUNT",
        help=f"tone pipeline: contrast (-{TONE_CONTRAST_LIMIT:g}..+{TONE_CONTRAST_LIMIT:g}, 0 = unchanged).",
    )
    parser.add_argument(
        "--crop", type=parse_crop, default=None, metavar="X,Y,W,H",
        help=(
            "Crop the result to this rectangle, in pixels of the input image (top-left "
            "X,Y plus width W and height H; clamped to the image). Applied after every "
            "pipeline. Requires exactly one input image."
        ),
    )
    parser.add_argument(
        "--save-silhouette", type=Path, default=None, metavar="DIR",
        help=(
            "Debug: when the aisharpen pipeline sharpens bodies (see --boxes-file), save "
            "full-resolution debug images into DIR, named after the photo: <name>_orig.jpg "
            "(original), <name>_silhouette.png (blend mask, white = sharpened), "
            "<name>_sharpened.jpg (the sharpened layer) and <name>_final.jpg (the result)."
        ),
    )
    args = parser.parse_args()

    tone = None
    tone_args = (args.highlights, args.midtones, args.shadows, args.contrast)
    if any(v is not None for v in tone_args):
        tone = ToneParams(*(v or 0.0 for v in tone_args)).clamped()
    boxes_by_name, keypoints_by_name, protected_by_name = {}, {}, {}
    if args.boxes_file is not None:
        try:
            raw = json.loads(args.boxes_file.read_text(encoding="utf-8-sig"))
            boxes_by_name, keypoints_by_name, protected_by_name = _parse_boxes_file(raw)
        except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as exc:
            log.warning("Could not read --boxes-file %s: %s", args.boxes_file, exc)
        else:
            log.info("Loaded --boxes-file %s: %s", args.boxes_file,
                     ", ".join(f"{name}: {len(boxes)} box(es), {len(protected_by_name.get(name, []))} protected"
                               for name, boxes in boxes_by_name.items())
                     or "no entries")

    try:
        sources = collect_inputs(args.inputPath)
    except (FileNotFoundError, OSError, UnicodeDecodeError) as exc:
        log.error("%s", exc)
        sys.exit(1)
    if not sources:
        log.error("No supported images found at: %s", args.inputPath)
        sys.exit(1)
    if args.out_file is not None and len(sources) != 1:
        log.error("--out-file requires exactly one input image (got %d).", len(sources))
        sys.exit(1)
    if args.crop is not None and len(sources) != 1:
        log.error("--crop requires exactly one input image (got %d).", len(sources))
        sys.exit(1)

    pipelines = resolve_pipelines(args.pipelines)
    if not pipelines:
        log.error("No valid pipelines selected.")
        sys.exit(1)

    openai_client = None
    if any(p.name.startswith("openai") for p in pipelines):
        if args.openaikey:
            import openai
            openai_client = openai.OpenAI(
                api_key=args.openaikey, timeout=_OPENAI_TIMEOUT_SECONDS
            )
        else:
            log.warning("No --openaikey supplied; skipping the openai_* pipelines.")
            pipelines = [p for p in pipelines if not p.name.startswith("openai")]
            if not pipelines:
                log.error("Nothing left to run.")
                sys.exit(1)

    dispatcher = None
    if any(p.name == "aisharpen" for p in pipelines):
        dispatcher = SharpenDispatcher(SHARPEN_MODELS[args.sharpen_model], args.face_enhance)
        dispatcher.start()

    log.info(
        "Processing %d image(s) through: %s",
        len(sources), " -> ".join(p.name for p in pipelines),
    )

    succeeded = 0
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = pool.map(
                lambda source: process_one(
                    source, pipelines, args.output, args.ext, args.compression,
                    dispatcher, openai_client, out_file=args.out_file,
                    body_boxes=boxes_by_name.get(source.name),
                    body_keypoints=keypoints_by_name.get(source.name),
                    protected_bodies=protected_by_name.get(source.name),
                    silhouette_dir=args.save_silhouette,
                    tone=tone,
                    crop=args.crop,
                ),
                sources,
            )
            succeeded = sum(1 for result in results if result is not None)
    finally:
        if dispatcher is not None:
            dispatcher.shutdown()

    failed = len(sources) - succeeded
    log.info("Done. %d succeeded, %d failed. Output: %s", succeeded, failed, args.output)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
