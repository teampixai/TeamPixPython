"""algo/segmentation.py — Swappable person-segmentation backends.

Used by autoedit.py's body-only sharpening mode (see AI_SHARPEN_SCOPE in
that file) to get a precise person-vs-background mask for a body crop, so a
sharpened crop can be alpha-blended back onto the original photo without a
hard seam at the player's silhouette.

Backends implement the single :class:`PersonSegmenter` interface so a faster
or more precise model can be swapped in later (e.g. SAM2, BiRefNet) without
touching any call site — just add a class here and register it in
``SEGMENTER_BACKENDS``.

Current default: ``mediapipe_selfie`` — MediaPipe's Tasks API Image
Segmenter with the "selfie" model. Chosen as the starting backend because it
loads and runs the fastest of the realistic options: a tiny (~250 KB),
single-purpose 256x256 model, Apache-2.0 licensed, and mediapipe is already a
hard dependency of this repo (unlike e.g. ultralytics/YOLO-seg, which is
AGPL-3.0 and which autoedit.py deliberately avoids depending on).
"""

from __future__ import annotations

import logging
import threading
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

log = logging.getLogger("BlurPictureDetector")

_MODEL_DIR = Path(__file__).resolve().parent.parent

# Modern Tasks API Image Segmenter model (not the legacy
# mediapipe.solutions.selfie_segmentation one) — single-channel confidence
# output: higher value = more likely "person". Bootstrap pattern mirrors
# algo/mediapipe_provider.py's face_landmarker.task/pose_landmarker_full.task.
_SELFIE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
_SELFIE_MODEL_PATH = _MODEL_DIR / "selfie_segmenter.tflite"


def _ensure_model(url: str, path: Path) -> Path:
    if path.exists():
        log.debug("Segmentation model present: %s", path)
        return path
    log.info("Downloading %s from %s ...", path.name, url)
    urllib.request.urlretrieve(url, path)
    log.info("Download complete: %s", path)
    return path


class PersonSegmenter(ABC):
    """One person-vs-background segmentation backend."""

    @abstractmethod
    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return a float32 array, same (height, width) as *image_bgr*, with
        values in [0, 1]: the confidence that each pixel belongs to a
        person (1 = foreground, 0 = background)."""


class MediaPipeSelfieSegmenter(PersonSegmenter):
    """MediaPipe Tasks API Image Segmenter, "selfie" model — see module
    docstring for why this is the default/fastest backend. Lazily loads a
    single process-wide instance on first use (same pattern as
    algo/mediapipe_provider.py's landmarker singletons); mediapipe itself is
    only imported here, not at module load time, so loading this module
    doesn't pull mediapipe into processes that never segment anything."""

    def __init__(self) -> None:
        self._segmenter = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._segmenter is not None:
            return self._segmenter
        with self._lock:
            if self._segmenter is not None:
                return self._segmenter
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            model_path = _ensure_model(_SELFIE_MODEL_URL, _SELFIE_MODEL_PATH)
            options = mp_vision.ImageSegmenterOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                running_mode=mp_vision.RunningMode.IMAGE,
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = mp_vision.ImageSegmenter.create_from_options(options)
            log.info("MediaPipe selfie segmenter loaded")
        return self._segmenter

    def segment(self, image_bgr: np.ndarray) -> np.ndarray:
        import cv2
        import mediapipe as mp

        segmenter = self._ensure_loaded()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = segmenter.segment(mp_image)
        # Single-category model: confidence_masks[0] is the person-confidence
        # map directly (no separate background channel to pick between). Some
        # MediaPipe builds return it as (H, W, 1); callers expect (H, W).
        mask = result.confidence_masks[0].numpy_view()
        return np.squeeze(mask, axis=-1).astype(np.float32) if mask.ndim == 3 else mask.astype(np.float32, copy=True)


# Registry mapping a backend name to its class, plus one lazily-constructed
# singleton per name (segmentation runs on many crops in a row — the model
# should only be loaded once per process). Add new backends by defining a
# PersonSegmenter subclass above and registering it here; nothing else needs
# to change to make it selectable.
SEGMENTER_BACKENDS: dict[str, type[PersonSegmenter]] = {
    "mediapipe_selfie": MediaPipeSelfieSegmenter,
}

# The backend new call sites should use unless they have a specific reason
# not to — see module docstring for why this one is the current default.
DEFAULT_SEGMENTER_BACKEND = "mediapipe_selfie"

_instances: dict[str, PersonSegmenter] = {}
_instances_lock = threading.Lock()


def get_segmenter(name: str = DEFAULT_SEGMENTER_BACKEND) -> PersonSegmenter:
    """Process-wide singleton :class:`PersonSegmenter` for *name*."""
    instance = _instances.get(name)
    if instance is not None:
        return instance
    with _instances_lock:
        instance = _instances.get(name)
        if instance is not None:
            return instance
        cls = SEGMENTER_BACKENDS.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown segmentation backend {name!r}; available: "
                f"{', '.join(sorted(SEGMENTER_BACKENDS))}"
            )
        instance = cls()
        _instances[name] = instance
    return instance
