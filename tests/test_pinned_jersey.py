"""A team colour pinned as a bare hue/shade ("White", "Navy") must still anchor
the jersey check, and a forced colour must not swallow nearby dark colours."""

from __future__ import annotations

import pytest

from algo.config import app_config
from algo.models import Body
from algo.regrade import _poll_jersey
from algo.album import PersonRecord
from algo.stages.jersey_counting import (
    _allowed_reference_labs, classify_body_jersey, pinned_target_lab,
)
from algo.utils import _color_from_label
from conftest import make_body_dict

WHITE_LAB = (91.9, 5.6, -8.6)
NAVY_LAB = (14.7, 4.6, -17.1)
SHADOWED_VIOLET_LAB = (18.0, 20.0, -30.0)  # closest to the Violet reference, not to Navy


def _record(color: str, lab) -> PersonRecord:
    return PersonRecord(make_body_dict(cloth_color=color, cloth_color_detail={"mean_lab": list(lab)}))


def _body(color: str, lab) -> Body:
    body = _record(color, lab).to_body()
    body.passed, body.rejection_reason = True, ""
    return body


def _verdict(body, *, pin, candidates, forced=frozenset(), regular=frozenset({"White", "Navy"})):
    label, target, bucket = _poll_jersey(candidates, app_config, pin)
    classify_body_jersey(
        body, forced, regular, _allowed_reference_labs(forced), _allowed_reference_labs(regular),
        target, bucket, _color_from_label(label), app_config,
    )
    return body.passed


@pytest.fixture
def candidates():
    # A dark-jersey majority, like the album the bug was found in.
    return [_record("Blue:Navy", NAVY_LAB) for _ in range(5)] + [_record("White:Bright", WHITE_LAB) for _ in range(2)]


@pytest.mark.parametrize("pin", ["White", "White:Bright"])
def test_a_pinned_white_team_matches_white_and_rejects_navy(candidates, pin):
    assert _verdict(_body("White:Bright", WHITE_LAB), pin=pin, candidates=candidates)
    assert not _verdict(_body("Blue:Navy", NAVY_LAB), pin=pin, candidates=candidates)


def test_a_pinned_hue_gets_a_measured_anchor_and_bucket(candidates):
    label, target, bucket = _poll_jersey(candidates, app_config, "White")
    assert label == "White"
    assert target == pytest.approx(WHITE_LAB)
    assert bucket == "Light"  # not the Dark majority


def test_a_pinned_shade_is_also_resolved(candidates):
    assert pinned_target_lab("Navy", {"Blue:Navy": [NAVY_LAB]}) == pytest.approx(NAVY_LAB)
    assert _verdict(_body("Blue:Navy", NAVY_LAB), pin="Navy", candidates=candidates)
    assert not _verdict(_body("White:Bright", WHITE_LAB), pin="Navy", candidates=candidates)


def test_a_pin_with_no_measured_bodies_uses_the_reference_colour():
    assert pinned_target_lab("White", {}) is not None
    assert pinned_target_lab("NotAColour", {"White:Bright": [WHITE_LAB]}) is None


def test_an_unresolvable_pin_does_not_fall_back_to_the_polled_majority(candidates):
    _, target, bucket = _poll_jersey(candidates, app_config, "NotAColour")
    assert target is None and bucket is None


def test_a_forced_colour_does_not_let_a_nearby_dark_colour_through(candidates):
    forced = frozenset({"Violet"})
    navy = _body("Blue:Navy", NAVY_LAB)  # well within tolerance of Violet, but closest to Navy
    assert not _verdict(navy, pin="White", candidates=candidates, forced=forced)


def test_a_forced_colour_is_still_accepted_by_label_or_when_it_is_the_closest():
    candidates = [_record("White:Bright", WHITE_LAB)]
    forced = frozenset({"Violet"})
    assert _verdict(_body("Purple:Violet", SHADOWED_VIOLET_LAB), pin="White", candidates=candidates, forced=forced)
    assert _verdict(_body("Purple:Violet", NAVY_LAB), pin="White", candidates=candidates, forced=forced)
