"""algo/star_rules.py — which LLM-graded photos earn the top star tiers.

Pure logic (no album or LLM access). Tiers are decided from the top down.
A photo earns 5 stars when it is both

1. among the top ``five_star_top_fraction`` of its pool by grade, and
2. the best photo — higher grade — of every pool photo taken within
   ``five_star_window_s`` seconds before or after it.

The 5-star photos are then removed from the pool, and 4 stars is decided on
what is left by the same two tests with ``four_star_top_fraction`` /
``four_star_window_s`` (so a 5-star photo neither takes a 4-star slot nor
blocks its neighbours in the window test). A *pool* is the set of photos
graded against each other: one burst, or all the standalone photos of an
album.

"Better" is a strict order: higher grade first, then the earlier photo, then
the key, so two tied photos never both win.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass


@dataclass(frozen=True)
class HighStarParams:
    four_star_top_fraction: float
    four_star_window_s: float
    five_star_top_fraction: float
    five_star_window_s: float


@dataclass(frozen=True)
class GradedShot:
    key: str
    timestamp: float
    grade: float

    def order_key(self) -> tuple:
        """Sorts the better shot first."""
        return (-self.grade, self.timestamp, self.key)


def _top_candidates(ranked: list[GradedShot], fraction: float) -> list[GradedShot]:
    """The best ``ceil(len * fraction)`` shots (at least one) of *ranked*;
    none when *fraction* is 0 or less, which disables the tier."""
    if fraction <= 0:
        return []
    count = max(1, math.ceil(len(ranked) * fraction - 1e-9))
    return ranked[:count]


def _winners(pool: list[GradedShot], fraction: float, window_s: float) -> list[GradedShot]:
    """The top-``fraction`` shots of *pool* that are also the best within
    +/- *window_s* seconds of themselves."""
    if not pool:
        return []
    ranked = sorted(pool, key=GradedShot.order_key)
    by_time = sorted(pool, key=lambda s: (s.timestamp, s.key))
    times = [s.timestamp for s in by_time]
    won = []
    for shot in _top_candidates(ranked, fraction):
        lo = bisect_left(times, shot.timestamp - window_s)
        hi = bisect_right(times, shot.timestamp + window_s)
        mine = shot.order_key()
        if all(other is shot or other.order_key() > mine for other in by_time[lo:hi]):
            won.append(shot)
    return won


def promote_high_stars(pool: list[GradedShot], params: HighStarParams) -> dict[str, int]:
    """``{key: 4 or 5}`` for the shots of *pool* that earn a high star tier."""
    five = _winners(pool, params.five_star_top_fraction, params.five_star_window_s)
    stars = {shot.key: 5 for shot in five}
    rest = [s for s in pool if s.key not in stars]
    stars.update({shot.key: 4 for shot in _winners(rest, params.four_star_top_fraction, params.four_star_window_s)})
    return stars
