"""algo/thresholds.py — the "extended" face-sharpness threshold.

The real sharp threshold decides whether a photo/body counts as sharp. The
extended threshold sits below it (default: sharp threshold - 0.3, never below
0) and decides which bodies are *close enough* to be worth identifying by
jersey: anything whose face score is above it gets a cloth colour and a
persisted team verdict (``is_teammate``), even if it still misses the real
threshold. Verdicts (sharp/blurry, stars, previews) never depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from algo.config import app_config

# Key in an album's run_settings holding a per-album offset override.
RUN_SETTINGS_KEY = "extended_threshold_offset"


@dataclass(frozen=True)
class ExtendedThreshold:
    """``max(0, sharp - offset)``, with the comparison rule the grading gate
    itself uses (a score strictly above the threshold passes)."""

    sharp: float
    offset: float = app_config.extended_threshold_offset

    @property
    def value(self) -> float:
        # A negative offset would put the extended threshold ABOVE the sharp
        # one, which makes no sense, so it is treated as 0.
        return max(0.0, self.sharp - max(0.0, self.offset))

    def passes(self, score: float) -> bool:
        return score > self.value

    @classmethod
    def for_album(cls, sharp: float, run_settings: dict | None = None) -> "ExtendedThreshold":
        """The album's own offset (run_settings), else the app-config default.
        An unreadable stored offset falls back to the default."""
        raw = (run_settings or {}).get(RUN_SETTINGS_KEY)
        try:
            offset = float(raw) if raw is not None else app_config.extended_threshold_offset
        except (TypeError, ValueError):
            offset = app_config.extended_threshold_offset
        return cls(sharp=sharp, offset=offset)
