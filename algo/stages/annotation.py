from __future__ import annotations

import logging
import os
import unicodedata
import uuid
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable

import cv2
import numpy as np

from algo.config import AppConfig, app_config
from algo.frame import Frame
from algo.models import Box, Face
from algo.utils import apply_auto_adjustment

if TYPE_CHECKING:  # algo.album imports nothing from here, but keep this module light
    from algo.album import AlbumImage, PersonRecord

log = logging.getLogger("TeamPixPython")

PREVIEWS_SUBDIR = "previews"

# What the preview says about a person who has no face-recognition name but
# whose jersey matched the album's team (Body.is_teammate).
TEAMMATE_LABEL = "teammate"

# COCO 17-keypoint skeleton: pairs of indices to connect with a line.
_COCO_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 4),   # head
    (5, 6),                             # shoulders
    (5, 7), (7, 9),                     # left arm
    (6, 8), (8, 10),                    # right arm
    (5, 11), (6, 12),                   # torso sides
    (11, 12),                           # hips
    (11, 13), (13, 15),                 # left leg
    (12, 14), (14, 16),                 # right leg
)


def _draw_status_icon(
    image: np.ndarray,
    x: int,
    y: int,
    size: int,
    color: tuple[int, int, int],
    *,
    passed: bool,
) -> None:
    """Draw a filled *size* × *size* badge with a white ✓ or ✗ at (x, y)."""
    h, w = image.shape[:2]
    x2, y2 = min(x + size, w), min(y + size, h)
    sw, sh = x2 - x, y2 - y
    lw = max(2, size // 14)

    cv2.rectangle(image, (x, y), (x2, y2), color, -1)

    if passed:
        p1 = (x + int(sw * 0.15), y + int(sh * 0.50))
        p2 = (x + int(sw * 0.40), y + int(sh * 0.76))
        p3 = (x + int(sw * 0.85), y + int(sh * 0.24))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p2, p3, (255, 255, 255), lw, cv2.LINE_AA)
    else:
        p1 = (x + int(sw * 0.20), y + int(sh * 0.20))
        p2 = (x + int(sw * 0.80), y + int(sh * 0.80))
        p3 = (x + int(sw * 0.80), y + int(sh * 0.20))
        p4 = (x + int(sw * 0.20), y + int(sh * 0.80))
        cv2.line(image, p1, p2, (255, 255, 255), lw, cv2.LINE_AA)
        cv2.line(image, p3, p4, (255, 255, 255), lw, cv2.LINE_AA)


def _identity_label(body) -> str:
    """The text naming a person on the preview: their face-recognition name
    when they have one, otherwise "teammate" if the jersey check flagged
    them, otherwise nothing."""
    name = (body.player_name or "").strip()
    if name:
        return name
    return TEAMMATE_LABEL if body.is_teammate else ""


# Fonts tried (in order) for labels that OpenCV's built-in Hershey font can't
# render -- anything outside ASCII, e.g. accented or CJK player names.
_UNICODE_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "DejaVuSans.ttf",
    "arial.ttf",
)


@lru_cache(maxsize=16)
def _unicode_font(px: int):
    """A TrueType font at *px*, or None if Pillow or every candidate font is
    unavailable."""
    try:
        from PIL import ImageFont
    except ImportError:
        return None
    for candidate in _UNICODE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, px)
        except OSError:
            continue
    return None


def _ascii_fallback(text: str) -> str:
    """Best-effort ASCII rendering for when no Unicode font is available
    ("José" -> "Jose"); characters with no ASCII form become "?"."""
    out = []
    for ch in text:
        folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode()
        out.append(folded or "?")
    return "".join(out)


def _text_stroke(px: int) -> int:
    """Stroke width that gives Pillow-drawn text roughly the weight of the
    bold OpenCV labels it sits next to."""
    return max(1, round(px / 24))


def _text_extent(text: str, font, scale: float, thick: int, px: int) -> tuple[int, int]:
    """(width, height) in pixels that `_draw_label` will use for *text*."""
    if text.isascii():
        (w, h), _ = cv2.getTextSize(text, font, scale, thick)
        return w, h
    ttf = _unicode_font(px)
    if ttf is None:
        (w, h), _ = cv2.getTextSize(_ascii_fallback(text), font, scale, thick)
        return w, h
    left, top, right, bottom = ttf.getbbox(text, anchor="ls", stroke_width=_text_stroke(px))
    return right - left, bottom - top


def _draw_label(
    image: np.ndarray, text: str, org: tuple[int, int],
    font, scale: float, color: tuple[int, int, int], thick: int, px: int,
) -> None:
    """cv2.putText for ASCII; Pillow with a real TrueType font for anything
    else, so non-ASCII names aren't drawn as "???". *org* is the baseline-left
    corner, as for cv2.putText."""
    if text.isascii():
        cv2.putText(image, text, org, font, scale, color, thick, cv2.LINE_AA)
        return
    ttf = _unicode_font(px)
    if ttf is None:
        cv2.putText(image, _ascii_fallback(text), org, font, scale, color, thick, cv2.LINE_AA)
        return

    from PIL import Image, ImageDraw

    x, y = org
    stroke = _text_stroke(px)
    left, top, right, bottom = ttf.getbbox(text, anchor="ls", stroke_width=stroke)
    pad = 2
    x0, y0 = max(0, x + left - pad), max(0, y + top - pad)
    x1, y1 = min(image.shape[1], x + right + pad), min(image.shape[0], y + bottom + pad)
    if x1 <= x0 or y1 <= y0:
        return
    region = Image.fromarray(cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
    rgb = (color[2], color[1], color[0])
    ImageDraw.Draw(region).text(
        (x - x0, y - y0), text, font=ttf, fill=rgb, anchor="ls", stroke_width=stroke, stroke_fill=rgb,
    )
    image[y0:y1, x0:x1] = cv2.cvtColor(np.asarray(region), cv2.COLOR_RGB2BGR)


def _write_jpeg_atomic(path: Path, image: np.ndarray, quality: int) -> None:
    """Write *image* to *path* so a concurrent reader never sees a half-written
    file: previews are rendered on demand while the web app may already be
    serving the same path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}")
    if not cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"cv2.imwrite failed to write preview: {tmp}")
    os.replace(tmp, path)


# How much of the picture a preview darkens outside the accepted edit's crop.
CROPPED_OUT_DARKEN = 0.8


def darken_outside_crop(
    image: np.ndarray,
    crop: "tuple[float, float, float, float] | None",
    amount: float = CROPPED_OUT_DARKEN,
) -> np.ndarray:
    """*image* with everything outside the normalised ``(x1, y1, x2, y2)``
    *crop* darkened by *amount* (0.8 = 80% black), so a preview of the
    original shows what an accepted crop removed. Returns *image* itself when
    there is nothing to darken."""
    if crop is None:
        return image
    h, w = image.shape[:2]
    x1, y1 = round(crop[0] * w), round(crop[1] * h)
    x2, y2 = round(crop[2] * w), round(crop[3] * h)
    if x1 <= 0 and y1 <= 0 and x2 >= w and y2 >= h:
        return image
    keep = 1.0 - amount
    out = image.copy()
    for rows, cols in (
        (slice(0, y1), slice(None)),
        (slice(y2, h), slice(None)),
        (slice(y1, y2), slice(0, x1)),
        (slice(y1, y2), slice(x2, w)),
    ):
        band = out[rows, cols]
        if band.size:
            out[rows, cols] = np.rint(band * keep).astype(image.dtype)
    return out


def _annotate_frame(
    frame: Frame,
    preview_path: Path,
    config: AppConfig,
    crop: "tuple[float, float, float, float] | None" = None,
) -> None:
    """Write *frame*'s annotated preview to *preview_path*.

    - No image data  → do nothing (read error upstream).
    - No bodies      → no person detected; save original as-is.
    - Otherwise      → draw boxes, keypoints, scores, and each person's name
                       (or "teammate").

    Thumbnails are not made here: they show the plain photo and are written
    once at import (see algo/stages/image_analysis.py).

    With *crop* (a normalised box, see AlbumImage.edited_crop) everything
    outside it is darkened 80% (darken_outside_crop) to signify the photo was
    cropped.
    """
    if frame.image is None:
        return

    if not frame.bodies:
        _write_jpeg_atomic(preview_path, darken_outside_crop(frame.image, crop), 60)
        return

    annotated = apply_auto_adjustment(frame.image, frame.auto_adjustment).copy()
    overlay   = annotated.copy()
    h_out, w_out = annotated.shape[:2]
    sx = w_out
    sy = h_out
    ann_scale = max(h_out, w_out) / config.normalized_img_max_long_edge

    face_thick     = max(1, round(config.annotation_face_box_thickness    * ann_scale))
    box_thick      = max(1, round(config.annotation_box_thickness          * ann_scale))
    icon_size      = max(20, round(config.annotation_icon_size             * ann_scale))
    kp_radius      = max(3, round(config.annotation_face_kp_radius         * ann_scale))
    kp_thick       = max(1, round(config.annotation_face_kp_thickness      * ann_scale))
    body_kp_size   = max(2, round(config.annotation_body_kp_size           * ann_scale))
    body_kp_thick  = max(1, round(config.annotation_body_kp_thickness      * ann_scale))
    skeleton_thick = max(1, round(config.annotation_skeleton_thickness      * ann_scale))
    narrow_thick   = max(1, round(config.annotation_narrow_face_box_thickness * ann_scale))
    font           = cv2.FONT_HERSHEY_SIMPLEX
    font_thick     = config.annotation_score_font_thickness
    (_, _base_h), _ = cv2.getTextSize("Mg", font, 1.0, font_thick)
    font_scale           = config.annotation_score_font_size_px            * ann_scale / max(_base_h, 1)
    rejection_font_scale = config.annotation_rejection_reason_font_size_px * ann_scale / max(_base_h, 1)

    score_labels: list[tuple] = []
    # (text, x, y, colour) -- who each person is; drawn after the score labels.
    identity_labels: list[tuple] = []
    identity_px = max(8, round(config.annotation_score_font_size_px * ann_scale * 1.1))
    identity_gap = max(6, round(8 * ann_scale))
    # Bodies to annotate: top-N by bbox area ∪ all passed bodies.
    # Sort largest-first so bigger boxes render behind smaller ones.
    sorted_by_size = sorted(frame.bodies, key=lambda b: b.bbox.area, reverse=True)
    top_n_ids = {id(b) for b in sorted_by_size[:config.annotation_top_n_bodies]}
    bodies_to_render = [b for b in sorted_by_size if id(b) in top_n_ids or b.passed]

    for body in bodies_to_render:
        b = body.bbox
        rbx1 = int(b.x1 * sx); rby1 = int(b.y1 * sy)
        rbx2 = int(b.x2 * sx); rby2 = int(b.y2 * sy)

        body_color = (
            config.annotation_box_color_fail if not body.passed
            else config.annotation_box_color_pass
        )
        cv2.rectangle(overlay, (rbx1, rby1), (rbx2, rby2), body_color, box_thick)
        _draw_status_icon(overlay, rbx1, rby1, icon_size, body_color, passed=body.passed)

        # Skeleton
        kps = body.keypoints
        for ka, kb in _COCO_SKELETON:
            if ka >= len(kps) or kb >= len(kps):
                continue
            pa, pb = kps[ka].point, kps[kb].point
            if (pa.x == 0 and pa.y == 0) or (pb.x == 0 and pb.y == 0):
                continue
            cv2.line(overlay,
                     (int(pa.x * sx), int(pa.y * sy)),
                     (int(pb.x * sx), int(pb.y * sy)),
                     body_color, skeleton_thick, cv2.LINE_AA)

        # Body keypoints
        half = body_kp_size // 2
        for kp in kps:
            if kp.point.x == 0 and kp.point.y == 0:
                continue
            rkpx = int(kp.point.x * sx)
            rkpy = int(kp.point.y * sy)
            cv2.rectangle(overlay,
                          (rkpx - half, rkpy - half),
                          (rkpx + half, rkpy + half),
                          body_color, body_kp_thick, cv2.LINE_AA)

        # Face bbox + sharpness score label
        face_label: tuple[int, int, int] | None = None  # (x, baseline y, width) of the score text
        if body.best_face is not None:
            fb: Box = body.best_face.bbox
            rfx1 = int(fb.x1 * sx); rfy1 = int(fb.y1 * sy)
            rfx2 = int(fb.x2 * sx); rfy2 = int(fb.y2 * sy)

            nfb: Box | None = body.best_narrow_box
            if config.use_narrow_face_box and nfb is not None:
                rnx1 = int(nfb.x1 * sx); rny1 = int(nfb.y1 * sy)
                rnx2 = int(nfb.x2 * sx); rny2 = int(nfb.y2 * sy)
                cv2.rectangle(overlay, (rnx1, rny1), (rnx2, rny2), body_color, narrow_thick)
                label_x, label_bottom = rnx1, rny2
            else:
                cv2.rectangle(overlay, (rfx1, rfy1), (rfx2, rfy2), body_color, face_thick)
                label_x, label_bottom = rfx1, rfy2

            label = f"{body.sharpness_score:.2f}"
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thick)
            text_y = min(label_bottom + th + baseline + 2, h_out - 1)
            score_labels.append((label, label_x, text_y, body_color, font_scale))
            face_label = (label_x, text_y, tw)

        # Name (or "teammate") next to the score, or -- with no face box to
        # hang it on -- just under the body box's pass/fail badge.
        identity = _identity_label(body)
        if identity:
            iw, _ih = _text_extent(identity, font, font_scale, font_thick, identity_px)
            if face_label is not None:
                ix, iy = face_label[0] + face_label[2] + identity_gap, face_label[1]
                if ix + iw > w_out - 2:
                    # No room beside the score: stack it just below instead of
                    # sliding it back over the score text.
                    ix, iy = face_label[0], min(face_label[1] + _ih + max(4, identity_gap // 2), h_out - 1)
            else:
                ix = rbx1 + 2
                iy = min(rby1 + icon_size + _ih + 4, h_out - 1)
            ix = max(0, min(ix, w_out - iw - 2))
            identity_labels.append((identity, ix, iy, body_color))

        # Cloth colour label (passed) or rejection reason (failed)
        if body.passed:
            cloth = body.cloth_color
            if cloth not in ("N/A", "Unknown"):
                (clw, clh), _ = cv2.getTextSize(cloth, font, font_scale, font_thick)
                cl_x = max(rbx1, rbx2 - clw - 4)
                cl_y = max(clh + 4, rby1 + clh + 4)
                score_labels.append((cloth, cl_x, cl_y, body_color, font_scale))
        else:
            reason = body.rejection_reason or "rejected (no reason given)"
            (rw, rh), _ = cv2.getTextSize(reason, font, rejection_font_scale, font_thick)
            r_x = max(rbx1, rbx2 - rw - 4)
            r_y = max(rh + 4, rby1 + rh + 4)
            score_labels.append((reason, r_x, r_y, body_color, rejection_font_scale))

        # Face landmark circles
        if body.best_face is not None:
            ann_face: Face = body.best_face
            for lm in ann_face.landmarks:
                rkpx = int(lm.point.x * sx)
                rkpy = int(lm.point.y * sy)
                kp_color = (
                    config.annotation_box_color_pass
                    if lm.confidence >= config.face_coverage_conf_threshold
                    else config.annotation_box_color_fail
                )
                cv2.circle(overlay, (rkpx, rkpy), kp_radius, kp_color, kp_thick, cv2.LINE_AA)

    cv2.addWeighted(overlay, config.annotation_alpha,
                    annotated, 1.0 - config.annotation_alpha, 0, annotated)

    for label, lx, ly, lcolor, fscale in score_labels:
        cv2.putText(annotated, label, (lx, ly), font, fscale, lcolor, font_thick, cv2.LINE_AA)
    for text, lx, ly, lcolor in identity_labels:
        _draw_label(annotated, text, (lx, ly), font, font_scale, lcolor, font_thick, identity_px)

    _write_jpeg_atomic(preview_path, darken_outside_crop(annotated, crop), 60)


# ---------------------------------------------------------------------------
# On-demand previews
# ---------------------------------------------------------------------------
#
# A preview is a CACHE of "this photo with its detections drawn on", rendered
# the first time something asks for it (culling_app.py) from the album's
# persisted per-person data -- it is no longer pre-generated for the whole
# album at import. Everything that draws one goes through `_annotate_frame`
# above, via `render_album_preview`, so every preview looks the same however
# and whenever it was produced.

def render_album_preview(
    image: "AlbumImage",
    config: AppConfig | None = None,
    player_name_for: "Callable[[PersonRecord], str] | None" = None,
) -> bool:
    """Draw *image*'s annotated preview to its `preview_path`.

    The photo is re-decoded from its original (the same reader the import
    used) and annotated from the album's stored per-person records. Each
    person's name comes from *player_name_for(record)* when given -- the web
    app passes the current face-recognition result -- falling back to the
    name stored on the record; with no name, "teammate" is drawn for people
    the jersey check flagged.

    Returns False (writing nothing) when the entry has no preview path or the
    source photo is missing or can't be decoded.

    The preview always shows the ORIGINAL photo; if an accepted edit cropped
    it (`AlbumImage.edited_crop`), what the crop removed is darkened 80%.
    """
    preview = image.preview_path
    source = image.original_path
    if preview is None or source is None or not source.is_file():
        return False

    # Lazy: image_analysis pulls in the whole detection stack (ultralytics...),
    # which a process that only serves previews shouldn't pay for until needed.
    from algo.stages.image_analysis import _read_image

    decoded = _read_image(source)
    if decoded is None:
        return False

    bodies = []
    for record in image.bodies:
        body = record.to_body()
        body.player_name = ((player_name_for(record) if player_name_for else "") or record.player_name or "").strip()
        bodies.append(body)

    frame = Frame(
        path=source,
        bodies=bodies,
        image=decoded,
        auto_adjustment=image.auto_adjustment,
        output_key=image.key,
    )
    _annotate_frame(frame, preview, config or app_config, crop=image.edited_crop)
    return True


def invalidate_previews(album_path: Path, stems: "Iterable[str] | None" = None) -> int:
    """Delete cached previews so they are re-rendered, with current data, the
    next time they are requested. Thumbnails are left alone: they show the
    plain photo, so nothing that changes a preview makes them stale.

    *stems* are preview filename stems (`AlbumImage.preview_stem`); None
    invalidates every cached preview in the album. Returns how many files
    were removed.
    """
    folder = Path(album_path) / PREVIEWS_SUBDIR
    if not folder.is_dir():
        return 0
    if stems is None:
        targets = [fp for fp in folder.iterdir() if fp.is_file()]
    else:
        targets = [folder / f"{stem}.jpg" for stem in stems]
    removed = 0
    for fp in targets:
        try:
            fp.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("Could not delete cached preview %s: %s", fp, exc)
    return removed
