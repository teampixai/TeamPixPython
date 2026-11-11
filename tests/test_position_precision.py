"""Positions are persisted as fractions with at most 5 decimals (0.12345)."""

from __future__ import annotations

from algo.album import _person_wire_in_crop
from algo.facereco_provider import Box as ProviderBox
from algo.models import Box, Point, PredictedKeyPoint


def test_box_wire_keeps_five_decimals():
    wire = Box(0.48241875542534723, 0.2095703125, 0.6555701361762153, 0.89366943359375).to_wire()
    assert wire == {"x1": 0.48242, "y1": 0.20957, "x2": 0.65557, "y2": 0.89367}


def test_provider_box_wire_keeps_five_decimals():
    assert ProviderBox(0.123456789, 0.5, 0.987654321, 1.0).to_wire() == {
        "x1": 0.12346, "y1": 0.5, "x2": 0.98765, "y2": 1.0,
    }


def test_keypoint_wire_rounds_position_but_not_confidence():
    wire = PredictedKeyPoint(Point(0.5483832281827926, 0.2795861729234457), 0.9999566078186035).to_wire()
    assert (wire["x"], wire["y"]) == (0.54838, 0.27959)
    assert wire["conf"] == 0.9999566078186035


def test_reframing_to_a_crop_keeps_five_decimals():
    person = {
        "body_bbox": {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.9},
        "body_keypoints": [{"x": 0.4, "y": 0.5, "conf": 0.9, "passed": True}],
    }
    out = _person_wire_in_crop(person, (0.1, 0.1, 0.7, 0.7))
    assert out["body_bbox"] == {"x1": 0.33333, "y1": 0.33333, "x2": 0.83333, "y2": 1.0}
    assert (out["body_keypoints"][0]["x"], out["body_keypoints"][0]["y"]) == (0.5, 0.66667)
