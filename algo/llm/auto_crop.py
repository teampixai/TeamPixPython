"""Asks an OpenAI vision model how to level and crop one photo (the edit
panel's Auto crop).

The model sees a small copy of the photo and answers with the Level to apply
(degrees, positive = counter-clockwise) and the part of the photo to keep, as
a box in the coordinates of the image it was shown, i.e. before the level is
applied. The panel turns that into a crop on the levelled photo
(webui/culling/static/js/ai-edit.js).
"""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass, replace
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

_PROMPT = """You are a photo editor. Decide how to level this photo and how to crop it. \
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
2. CROP. The goal is to remove what does not belong, not to re-compose the \
shot. Cut away unnecessary elements near the edges: bystanders, coaches, \
parents, equipment, bags, signs, fences, parked cars, other games, partial \
people cut off by the frame, and empty foreground or sky. Keep every main \
subject (the players taking part in the action, and the ball) whole, with a \
little room around them. If nothing needs to be removed, keep the whole photo \
(0, 0, 1, 1). Give the box as fractions of the photo as you see it, before \
any rotation: x1, y1 is the top-left corner and x2, y2 the bottom-right \
corner, all between 0 and 1, with x1 < x2 and y1 < y2. In removed, list in a \
few words what the crop cuts away.
{ratio_line}
Answer with JSON only, exactly in this shape:
{{"level_check": "<one short sentence>", "level_degrees": <number>, \
"crop": {{"x1": <number>, "y1": <number>, "x2": <number>, "y2": <number>}}, \
"removed": "<a few words>", "reason": "<one short sentence summing up both edits>"}}"""


@dataclass(frozen=True)
class AutoCropSuggestion:
    level: float
    box: tuple[float, float, float, float]
    reason: str
    # What the model looked at to level the photo, and what the crop removes.
    level_check: str = ""
    removed: str = ""
    # How long the request took and what the provider has spent so far (set
    # by suggest_crop).
    elapsed: float = 0.0
    cost_usd: float = 0.0


def build_prompt(ratio_h: Optional[float] = None, ratio_w: Optional[float] = None) -> str:
    """The instructions for the model; with *ratio_h* x *ratio_w* the crop is
    asked to have that height:width shape (as it will look after the level)."""
    if ratio_h and ratio_w:
        line = (
            f"The crop must have a height to width ratio of {ratio_h:g}:{ratio_w:g} "
            f"(width / height = {ratio_w / ratio_h:.3f}) in actual pixels, "
            "whatever the photo's own proportions are.\n"
        )
    else:
        line = ""
    return _PROMPT.format(ratio_line=line)


def parse_suggestion(raw: Optional[str]) -> AutoCropSuggestion:
    """Reads the model's JSON answer. Raises ValueError when it is not
    usable; a level outside -180..180 is clamped, a box is clamped to the
    photo."""
    if not raw:
        raise ValueError("The model returned no answer.")
    try:
        data = json.loads(raw)
        level = float(data.get("level_degrees", 0.0))
        crop = data["crop"]
        x1, y1, x2, y2 = (float(crop[k]) for k in ("x1", "y1", "x2", "y2"))
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"The model's answer could not be read: {exc}") from exc
    if not all(math.isfinite(v) for v in (level, x1, y1, x2, y2)):
        raise ValueError("The model's answer contained invalid numbers.")

    x1, x2 = sorted((min(1.0, max(0.0, x1)), min(1.0, max(0.0, x2))))
    y1, y2 = sorted((min(1.0, max(0.0, y1)), min(1.0, max(0.0, y2))))
    if x2 - x1 < MIN_BOX_SIDE or y2 - y1 < MIN_BOX_SIDE:
        raise ValueError("The model suggested a crop that is too small.")
    level = min(MAX_LEVEL_DEGREES, max(-MAX_LEVEL_DEGREES, level))
    def text(name: str) -> str:
        value = data.get(name)
        return value.strip() if isinstance(value, str) else ""

    return AutoCropSuggestion(
        level=round(level, 2) + 0.0,
        box=(x1, y1, x2, y2),
        reason=text("reason"),
        level_check=text("level_check"),
        removed=text("removed"),
    )


def make_provider(api_key: str, model: str = DEFAULT_OPENAI_MODEL) -> OpenAIProvider:
    """The OpenAI provider Auto crop talks through — the one the import's burst
    ranking uses, so it shares the default model, request throttling and cost
    tracking."""
    return OpenAIProvider(api_key=api_key, model=model, timeout=AUTO_CROP_TIMEOUT_SECONDS)


def suggest_crop(
    provider: OpenAIProvider,
    jpeg_bytes: bytes,
    *,
    ratio_h: Optional[float] = None,
    ratio_w: Optional[float] = None,
) -> AutoCropSuggestion:
    """Sends *jpeg_bytes* (a shrunken copy of the photo) through *provider*
    and returns the model's level and crop. Raises ValueError for an unusable
    answer and lets the provider's request errors through."""
    raw, elapsed = provider.ask_json(
        build_prompt(ratio_h, ratio_w), base64.b64encode(jpeg_bytes).decode("ascii")
    )
    suggestion = parse_suggestion(raw)
    cost = provider.get_cost_summary()
    suggestion = replace(suggestion, elapsed=elapsed, cost_usd=cost.total_cost_usd)
    log.info("Auto crop (%s): level %+.2f, box %s in %.1fs — %s",
             provider.model, suggestion.level, suggestion.box, elapsed, suggestion.reason)
    return suggestion
