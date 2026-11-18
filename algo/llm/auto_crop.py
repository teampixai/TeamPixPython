"""Asks an OpenAI vision model how to level and crop one photo (the edit
panel's Auto crop).

The model sees a small copy of the photo and is briefed like a sports photo
editor (an editorial, rule-of-thirds crop that removes distractions but leaves
room to breathe, centred on a player in the team's jersey colour). It answers
with the Level to apply (degrees, positive = counter-clockwise) and two crops,
one per entry of CROP_FORMATS: an Instagram 4:5 portrait and one in the
photo's own proportions. Each crop is a box in the coordinates of the image
the model was shown, i.e. before the level is applied. The panel lets the user
pick one and turns it into a crop on the levelled photo
(webui/culling/static/js/ai-edit.js).
"""

from __future__ import annotations

import base64
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Optional

from algo.llm.culling_provider import DEFAULT_OPENAI_MODEL, OpenAIProvider

log = logging.getLogger(__name__)

# Long edge, in pixels, of the copy of the photo sent to the model.
AUTO_CROP_IMAGE_LONG_EDGE = 768
AUTO_CROP_JPEG_QUALITY = 85
AUTO_CROP_TIMEOUT_SECONDS = 120.0
# A suggestion keeping less than this fraction of either side is rejected as
# implausible.
MIN_BOX_SIDE = 0.1
MAX_LEVEL_DEGREES = 180.0

# The crops asked for, in the order the panel offers them. A ratio of None
# means the photo's own proportions. Ratios are height:width, like the panel's
# crop presets.
INSTAGRAM = "instagram"
ORIGINAL = "original"
CROP_FORMATS: dict[str, Optional[tuple[int, int]]] = {INSTAGRAM: (5, 4), ORIGINAL: None}

_ROLE = """ROLE. You are an elite sports photography editor working at the \
editorial standard of ESPN and Sports Illustrated. Photography is the art of \
reduction: a photograph is complete when nothing more can be removed without \
weakening it, but a powerful photograph also needs room to breathe. Your goal \
is not the tightest crop but the most visually compelling one: intentional \
composition, balanced negative space, no unnecessary distractions. """

_CROP_RULES = """a. Story. Find the primary athlete, the essential action or emotional moment, \
the teammates, opponents and ball that matter to it, and the athlete's \
expression, movement, posture and direction of attention. Not every strong \
sports photo is intense action: a player walking with a thoughtful expression \
can be a compelling portrait. Decide the strongest story the photo can tell.
b. Reduction. Remove what does not strengthen that story: unrelated players \
and spectators, referees not part of the moment, coaches, parents, bags, \
signs, parked cars, other games, excess grass, sky, fences or buildings, \
awkward partial figures at the edges, and empty space with no purpose. Do not \
cut important relationships (the ball when it matters, a relevant opponent, \
a meaningful gesture, context that adds to the story) just to crop tighter, \
and do not keep people only because they wear the team's colour.
c. Rule of thirds. Use it as the starting point, not a rigid rule: put the \
athlete's face or eyes near a thirds intersection when it suits, avoid simply \
centring the athlete, and leave space in the direction of movement or gaze. \
Stronger visual relationships win over lining things up with the grid.
d. Breathing room. Do not crop too tightly. Around the region that tells the \
story, add about 5 to 10% intentional margin where the photo allows: room \
above the head, hands, feet and elbows not jammed against the edges, enough \
ground under the feet, and space in the direction of movement. The athlete \
must not feel trapped. Negative space must have a purpose, not be leftover \
background.
e. Hierarchy. At a glance, on a phone screen, a viewer should see who the \
subject is, what is happening and why it is interesting. Secondary subjects \
may stay when they add context or depth, but must not compete with the \
primary one.
f. Choose. For each crop, weigh at least three candidates in your head: a \
tight one, a balanced one (thirds placement, breathing room, clear \
hierarchy) and an environmental one (more context or negative space). Score \
them on storytelling and emotional impact (25%), subject prominence and \
clarity (20%), thirds composition and balance (20%), removal of distractions \
(20%), and breathing room and framing (15%), and keep the best. When they \
are close, prefer the balanced crop to an overly tight one.
g. Check. Is the story immediately clear and the athlete emphasised? Does \
the subject have comfortable room, with no essential body part or the ball \
accidentally cut? Is every bit of negative space intentional? Would removing \
anything more weaken the photo? If the crop feels cramped, loosen it; if it \
feels empty or distracted, tighten it.
"""

_CLOSING = """Remove everything that weakens the photograph, keep everything that tells \
the story, and leave enough room for the composition to breathe."""

_PROMPT = _ROLE + """You only \
crop and level the photo; you never change its content.

The photo you see is {width} x {height} pixels.
{subject_line}
Do step 1 before step 2.

1. LEVEL. Look for lines that should be level or vertical: the horizon, the \
far touchline or field edge, a fence, a building, a goal post, a pole. \
Estimate how far they are tilted, then give level_degrees: the rotation that \
makes them straight. Positive turns the photo counter-clockwise, negative \
clockwise. Do not default to 0: a tilt of even 0.5 to 3 degrees is worth \
fixing, and a clearly turned photo may need more (up to 180). Use 0 only \
when the lines really are straight, or when nothing in the photo gives a \
reliable reference. Put in level_check one short sentence naming the line \
you used and how it tilts (for example "the far touchline rises about 2 \
degrees to the right").

2. CROP. Make two editorial crops of the photo, following these rules.
""" + _CROP_RULES + """
The two crops:
{formats}
Give each box as fractions of the photo as you see it, before any rotation, \
with four decimals: x1, y1 is the top-left corner and x2, y2 the \
bottom-right corner, all between 0 and 1, with x1 < x2 and y1 < y2. The box \
must stay inside the photo and have exactly the required shape in pixels \
(width = (x2 - x1) * {width}, height = (y2 - y1) * {height}). In removed, \
list in a few words what that crop cuts away; in reason, explain the \
compositional choices in one or two short sentences.

Answer with JSON only, exactly in this shape:
{{"level_check": "<one short sentence>", "level_degrees": <number>, \
{answer_shape}}}

""" + _CLOSING


@dataclass(frozen=True)
class AutoCropBox:
    box: tuple[float, float, float, float]
    reason: str = ""
    # What the crop cuts away.
    removed: str = ""


@dataclass(frozen=True)
class AutoCropSuggestion:
    level: float
    # One crop per CROP_FORMATS name.
    crops: dict[str, AutoCropBox] = field(default_factory=dict)
    # What the model looked at to level the photo.
    level_check: str = ""
    # How long the request took and what the provider has spent so far (set
    # by suggest_crop).
    elapsed: float = 0.0
    cost_usd: float = 0.0


def describe_jersey_color(value: Optional[str]) -> str:
    """A team colour as stored on albums and teams ("HUE:SHADE", e.g.
    "White:Bright", "navy:" or "Navy") in plain words ("bright white",
    "navy")."""
    hue, _, shade = (value or "").partition(":")
    return " ".join(part.strip().lower() for part in (shade, hue) if part.strip())


def _ratio_text(width: int, height: int, ratio: Optional[tuple[int, int]]) -> tuple[str, float]:
    """How the prompt names a crop's shape, and its width / height."""
    if ratio is None:
        aspect = width / height
        simple = Fraction(width, height).limit_denominator(10)
        if abs(simple - aspect) < 0.005 * aspect:
            name = f"about {simple.numerator}:{simple.denominator}"
        else:
            name = f"{width}:{height}"
        return f"the photo's own proportions, {name} (width:height)", aspect
    ratio_h, ratio_w = ratio
    return f"{ratio_w}:{ratio_h} (width:height), a portrait for Instagram", ratio_w / ratio_h


def build_prompt(width: int, height: int, jersey_color: Optional[str] = None) -> str:
    """The instructions for the model, which is shown a *width* x *height*
    photo; with *jersey_color* (a stored "HUE:SHADE" team colour) the
    primary athlete is a player in that colour."""
    subject = _subject_line(jersey_color)
    formats = []
    shapes = []
    for name, ratio in CROP_FORMATS.items():
        text, aspect = _ratio_text(width, height, ratio)
        # The largest box of that shape the photo holds, as a size reference.
        box_w = min(width, height * aspect)
        box_h = box_w / aspect
        formats.append(
            f'- "{name}": {text}, width / height = {aspect:.3f}; at most {box_w:.0f} x {box_h:.0f} pixels.'
        )
        shapes.append(
            f'"{name}": {{"crop": {{"x1": <number>, "y1": <number>, "x2": <number>, "y2": <number>}}, '
            '"removed": "<a few words>", "reason": "<one or two short sentences>"}'
        )
    return _PROMPT.format(
        width=width,
        height=height,
        subject_line=subject,
        formats="\n".join(formats) + "\n",
        answer_shape=", ".join(shapes),
    )


def _text(data: dict, name: str) -> str:
    value = data.get(name)
    return value.strip() if isinstance(value, str) else ""


def _parse_crop(name: str, data: object) -> AutoCropBox:
    if not isinstance(data, dict):
        raise ValueError(f"The model's answer has no {name} crop.")
    try:
        crop = data["crop"]
        x1, y1, x2, y2 = (float(crop[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The model's {name} crop could not be read: {exc}") from exc
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError(f"The model's {name} crop contained invalid numbers.")
    x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
    y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        raise ValueError(f"The model suggested a {name} crop that is too small.")
    return AutoCropBox(box=(x1, y1, x2, y2), reason=_text(data, "reason"), removed=_text(data, "removed"))


def parse_suggestion(raw: Optional[str]) -> AutoCropSuggestion:
    """Reads the model's JSON answer. Raises ValueError when it is not
    usable (including a missing crop); a level outside -180..180 is clamped,
    a box is clamped to the photo."""
    if not raw:
        raise ValueError("The model returned no answer.")
    try:
        data = json.loads(raw)
        level = float(data.get("level_degrees", 0.0))
    except (json.JSONDecodeError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"The model's answer could not be read: {exc}") from exc
    if not math.isfinite(level):
        raise ValueError("The model's answer contained invalid numbers.")
    crops = {name: _parse_crop(name, data.get(name)) for name in CROP_FORMATS}
    level = min(MAX_LEVEL_DEGREES, max(-MAX_LEVEL_DEGREES, level))
    return AutoCropSuggestion(level=round(level, 2) + 0.0, crops=crops, level_check=_text(data, "level_check"))


def make_provider(api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> OpenAIProvider:
    """The OpenAI provider Auto crop talks through — the one the import's burst
    ranking uses, so it shares the default model, request throttling and cost
    tracking."""
    return OpenAIProvider(api_key=api_key, model=model, timeout=AUTO_CROP_TIMEOUT_SECONDS)


def suggest_crop(
    provider: OpenAIProvider,
    jpeg_bytes: bytes,
    *,
    width: int,
    height: int,
    jersey_color: Optional[str] = None,
) -> AutoCropSuggestion:
    """Sends *jpeg_bytes* (a shrunken *width* x *height* copy of the photo)
    through *provider* and returns the model's level and crops. Raises
    ValueError for an unusable answer and lets the provider's request errors
    through."""
    raw, elapsed = provider.ask_json(
        build_prompt(width, height, jersey_color), base64.b64encode(jpeg_bytes).decode("ascii")
    )
    suggestion = parse_suggestion(raw)
    cost = provider.get_cost_summary()
    suggestion = replace(suggestion, elapsed=elapsed, cost_usd=cost.total_cost_usd)
    log.info("Auto crop (%s): level %+.2f, crops %s in %.1fs", provider.model, suggestion.level,
             {name: c.box for name, c in suggestion.crops.items()}, elapsed)
    return suggestion


# --------------------------------------------------------------------------- #
# Batches: many photos per request (Album Tools -> Batch Auto Edit)
# --------------------------------------------------------------------------- #

# Photos sent in one request, like the standalone grading batches of the
# import (algo/stages/llm_culling.py DEFAULT_SINGLETON_BATCH_SIZE): it spreads
# a request's fixed cost over several photos. The model still crops each one on
# its own, never comparing them.
AUTO_CROP_BATCH_SIZE = 6

_BATCH_PROMPT = _ROLE + """You only crop the photos; you never change their content.

You are shown {count} unrelated photos, one after another, each labelled with \
its file name and its size in pixels (W x H). Crop each photo on its own \
merits and never compare them.
{subject_line}
CROP. For every photo, make two editorial crops of it, following these rules.
""" + _CROP_RULES + """
The two crops of each photo:
{formats}
Give each box as fractions of that photo, with four decimals: x1, y1 is the \
top-left corner and x2, y2 the bottom-right corner, all between 0 and 1, with \
x1 < x2 and y1 < y2. The box must stay inside the photo and have exactly the \
required shape in pixels, using that photo's own W and H. In removed, list in \
a few words what that crop cuts away; in reason, explain the compositional \
choices in one short sentence.

Answer with JSON only, exactly in this shape, with one entry per photo and \
each file name exactly as labelled:
{{"photos": [{{"file": "<file name>", {answer_shape}}}]}}

""" + _CLOSING


@dataclass(frozen=True)
class CropPhoto:
    """One photo of a batch: the shrunken copy the model sees (base64 JPEG)
    and its size in pixels. *focus* is the normalised (x1, y1, x2, y2) box of
    the player the crop must be built around, when one was chosen."""
    file: str
    image_b64: str
    width: int
    height: int
    focus: Optional[tuple[float, float, float, float]] = None


def _subject_line(jersey_color: Optional[str]) -> str:
    colour = describe_jersey_color(jersey_color)
    if colour:
        return (
            f"The team being photographed wears {colour} jerseys: the primary athlete is a player "
            f"in {colour}.\n"
        )
    return "The primary athlete is the player at the heart of the action or moment.\n"


_FOCUS_SUBJECT_LINE = (
    "The primary athlete in each photo is the player inside the box given in its label (x1, y1, x2, y2 as "
    "fractions of that photo): build both crops around that player, whatever else is going on.\n"
)


def build_batch_prompt(count: int, jersey_color: Optional[str] = None, focus: bool = False) -> str:
    """The instructions for a request carrying *count* photos; with *focus*
    every photo's label names the box of the player to build the crops
    around."""
    formats = []
    shapes = []
    for name, ratio in CROP_FORMATS.items():
        if ratio is None:
            shape = "the photo's own proportions, so x2 - x1 = y2 - y1"
        else:
            ratio_h, ratio_w = ratio
            shape = (
                f"{ratio_w}:{ratio_h} (width:height), a portrait for Instagram, so "
                f"(x2 - x1) * W = {ratio_w / ratio_h:.3f} * (y2 - y1) * H"
            )
        formats.append(f'- "{name}": {shape}.')
        shapes.append(
            f'"{name}": {{"crop": {{"x1": <number>, "y1": <number>, "x2": <number>, "y2": <number>}}, '
            '"removed": "<a few words>", "reason": "<one short sentence>"}'
        )
    return _BATCH_PROMPT.format(
        count=count,
        subject_line=_FOCUS_SUBJECT_LINE if focus else _subject_line(jersey_color),
        formats="\n".join(formats) + "\n",
        answer_shape=", ".join(shapes),
    )


def parse_batch_suggestions(raw: Optional[str], files: list[str]) -> dict[str, dict[str, AutoCropBox]]:
    """Reads a batch answer into ``{file: {crop name: AutoCropBox}}``. Entries
    are matched to *files* by name and anything else is dropped; a crop that
    is missing or unusable is left out (a photo may keep only one of its
    two), like the standalone grade parsing. Never raises: an unreadable
    answer gives ``{}``."""
    try:
        data = json.loads(raw or "")
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning("Auto crop batch: the answer is not JSON (%s): %r", exc, raw)
        return {}
    items = data.get("photos") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return {}
    valid = set(files)
    parsed: dict[str, dict[str, AutoCropBox]] = {}
    for item in items:
        file = item.get("file") if isinstance(item, dict) else None
        if not isinstance(file, str) or file not in valid:
            continue
        crops: dict[str, AutoCropBox] = {}
        for name in CROP_FORMATS:
            try:
                crops[name] = _parse_crop(name, item.get(name))
            except ValueError as exc:
                log.warning("Auto crop batch: %s: %s", file, exc)
        if crops:
            parsed[file] = crops
    return parsed


def _batch_label(index: int, photo: CropPhoto, focus: bool) -> str:
    label = f"Image {index + 1} ({photo.file}, {photo.width} x {photo.height} pixels)"
    if focus and photo.focus:
        x1, y1, x2, y2 = photo.focus
        label += f", primary athlete in box x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}"
    return label + ":"


def suggest_crops_batch(
    provider: OpenAIProvider,
    photos: list[CropPhoto],
    *,
    jersey_color: Optional[str] = None,
) -> dict[str, dict[str, AutoCropBox]]:
    """One request for all of *photos*; returns ``{file: {crop name:
    AutoCropBox}}`` for the photos the model answered usably. Lets the
    provider's request errors through."""
    focus = any(p.focus for p in photos)
    images = [
        (_batch_label(i, p, focus), p.image_b64)
        for i, p in enumerate(photos)
    ]
    raw, elapsed = provider.ask_json_images(build_batch_prompt(len(photos), jersey_color, focus), images)
    parsed = parse_batch_suggestions(raw, [p.file for p in photos])
    log.info("Auto crop batch (%s): %d/%d photo(s) answered in %.1fs",
             provider.model, len(parsed), len(photos), elapsed)
    return parsed


def split_batches(photos: list[CropPhoto], size: int = AUTO_CROP_BATCH_SIZE) -> list[list[CropPhoto]]:
    return [photos[i:i + size] for i in range(0, len(photos), size)]


def suggest_crops_batches(
    provider: OpenAIProvider,
    batches: list[list[CropPhoto]],
    *,
    jersey_color: Optional[str] = None,
    on_batch_done=None,
) -> list[dict[str, dict[str, AutoCropBox]]]:
    """Runs *batches* concurrently (up to the provider's ``max_concurrency``;
    its limiter paces the requests) and returns the answers in the order of
    *batches*. A failed batch gives ``{}`` rather than sinking the run.
    *on_batch_done(done, total, error)* is called as each one finishes."""
    results: list[dict[str, dict[str, AutoCropBox]]] = [{} for _ in batches]
    if not batches:
        return results
    workers = max(1, min(getattr(provider, "max_concurrency", 4), len(batches)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(suggest_crops_batch, provider, batch, jersey_color=jersey_color): i
            for i, batch in enumerate(batches)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            error = None
            try:
                results[futures[future]] = future.result()
            except Exception as exc:  # noqa: BLE001 — one batch's failure must not sink the run
                error = exc
                log.warning("Auto crop batch #%d failed: %s", futures[future], exc)
            if on_batch_done is not None:
                on_batch_done(done, len(batches), error)
    return results


def fit_box_to_ratio(
    box: tuple[float, float, float, float], ratio: float, width: int, height: int,
) -> tuple[float, float, float, float]:
    """*box* (fractions of a *width* x *height* photo) reshaped to the pixel
    ratio width / height = *ratio*, keeping its centre and area and shrinking
    or shifting as needed to stay inside the photo. The model only
    approximates the shape it is asked for; the edit panel does the same
    (reshapeCrop in ai-edit.js)."""
    x1, y1, x2, y2 = box
    box_w, box_h = (x2 - x1) * width, (y2 - y1) * height
    cx, cy = (x1 + x2) / 2 * width, (y1 + y2) / 2 * height
    h = math.sqrt(box_w * box_h / ratio)
    w = h * ratio
    if w > width:
        w, h = float(width), width / ratio
    if h > height:
        h, w = float(height), height * ratio
    x = min(max(cx - w / 2, 0.0), width - w)
    y = min(max(cy - h / 2, 0.0), height - h)
    return (x / width, y / height, (x + w) / width, (y + h) / height)


def crop_ratio(name: str, width: int, height: int) -> float:
    """The pixel width / height of the crop *name* for a *width* x *height*
    photo."""
    ratio = CROP_FORMATS[name]
    return width / height if ratio is None else ratio[1] / ratio[0]
