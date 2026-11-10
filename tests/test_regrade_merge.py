"""Deep regrade (1_prep_review.py --regrade-only) re-baselines the stars of a
photo whose verdict flipped, using the extended threshold."""

from __future__ import annotations

import importlib

from algo.thresholds import ExtendedThreshold


def _merge():
    return importlib.import_module("1_prep_review")._merge_preserved_fields


def _entry(status: str, score: float, **extra) -> dict:
    return {"file": "C:/p/A.jpg", "key": "A.jpg", "status": status, "sharpness_score": score, **extra}


def test_flipped_verdicts_are_rebaselined_with_the_extended_threshold():
    ext = ExtendedThreshold(0.5, 0.2)  # 0.3
    merge = _merge()

    to_sharp = merge([_entry("sharp", 0.6)], [_entry("blurry", 0.4, stars=2)], ext)[0]
    assert (to_sharp["stars"], to_sharp["keep"]) == (3, True)

    near = merge([_entry("blurry", 0.45)], [_entry("sharp", 0.6, stars=4)], ext)[0]
    assert (near["stars"], near["keep"]) == (2, False)

    far = merge([_entry("blurry", 0.2)], [_entry("sharp", 0.6, stars=4)], ext)[0]
    assert (far["stars"], far["keep"]) == (1, False)


def test_unchanged_verdicts_and_hand_ratings_keep_their_stars():
    ext = ExtendedThreshold(0.5, 0.2)
    merge = _merge()

    same = merge([_entry("sharp", 0.6)], [_entry("sharp", 0.6, stars=5, keep=True)], ext)[0]
    assert same["stars"] == 5

    manual = merge(
        [_entry("blurry", 0.2, stars_manual=True)], [_entry("sharp", 0.6, stars=4, stars_manual=True)], ext,
    )[0]
    assert manual.get("stars") == 4  # preserved field wins; not re-baselined
