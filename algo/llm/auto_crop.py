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

_PROMPT = """ROLE. You are an elite sports photography editor working at the \
editorial standard of ESPN and Sports Illustrated. Photography is the art of \
reduction: a photograph is complete when nothing more can be removed without \
weakening it, but a powerful photograph also needs room to breathe. Your goal \
is not the tightest crop but the most visually compelling one: intentional \
composition, balanced negative space, no unnecessary distractions. You only \
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
a. Story. Find the primary athlete, the essential action or emotional moment, \
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

Remove everything that weakens the photograph, keep everything that tells \
the story, and leave enough room for the composition to breathe."""


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
    jersey_color = describe_jersey_color(jersey_color)
    if jersey_color:
        subject = (
            f"The team being photographed wears {jersey_color} jerseys: the primary athlete is a player "
            f"in {jersey_color}.\n"
        )
    else:
        subject = "The primary athlete is the player at the heart of the action or moment.\n"
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
