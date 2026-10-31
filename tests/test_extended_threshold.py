"""The extended sharpness threshold and the per-body is_teammate verdict:
the threshold arithmetic, who gets a colour at import, who is flagged as a
teammate (without touching the blurry/sharp verdict), and the same on a
quick regrade of an existing album."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import make_body_dict, make_entry, write_album

from algo.album import Album, PersonRecord
from algo.config import app_config
from algo.frame import Frame
from algo.models import Body, Box
from algo.regrade import regrade_sensitivity
from algo.stages import grading
from algo.stages.grading import GradingStage
from algo.stages.jersey_counting import JerseyCountingStage
from algo.thresholds import RUN_SETTINGS_KEY, ExtendedThreshold

WHITE_LAB = [95.0, 0.0, 0.0]
PINK_LAB = [28.0, 48.0, -12.0]


# --------------------------------------------------------------------------- #
# ExtendedThreshold
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sharp,offset,expected", [
    (0.5, 0.3, 0.2),
    (0.62, 0.3, 0.32),
    (0.2, 0.3, 0.0),     # never below zero
    (0.0, 0.3, 0.0),
    (0.5, 0.0, 0.5),
    (0.5, -0.1, 0.5),    # a negative offset must not push it above the sharp threshold
])
def test_extended_threshold_value(sharp, offset, expected):
    assert ExtendedThreshold(sharp, offset).value == pytest.approx(expected)


def test_extended_threshold_passes_is_strictly_greater_than():
    ext = ExtendedThreshold(0.5, 0.2)  # 0.3
    assert ext.passes(0.31)
    assert not ext.passes(0.30)
    assert not ext.passes(0.0)
    assert ExtendedThreshold(0.1, 0.3).passes(0.01)  # threshold floored at 0


def test_extended_threshold_defaults_to_the_app_config_offset():
    assert ExtendedThreshold(0.5).offset == app_config.extended_threshold_offset == 0.3
    assert ExtendedThreshold.for_album(0.5).value == pytest.approx(0.2)


@pytest.mark.parametrize("settings,expected_offset", [
    ({RUN_SETTINGS_KEY: 0.1}, 0.1),
    ({RUN_SETTINGS_KEY: "0.25"}, 0.25),
    ({RUN_SETTINGS_KEY: "garbage"}, 0.3),
    ({RUN_SETTINGS_KEY: None}, 0.3),
    ({}, 0.3),
    (None, 0.3),
])
def test_extended_threshold_for_album_reads_the_per_album_offset(settings, expected_offset):
    assert ExtendedThreshold.for_album(0.5, settings).offset == pytest.approx(expected_offset)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_is_teammate_is_false_when_absent_and_persists_when_set():
    record = PersonRecord({})
    assert record.is_teammate is False
    record.is_teammate = True
    assert record.wire["is_teammate"] is True
    assert record.to_body().is_teammate is True
    assert PersonRecord.from_body(record.to_body()).is_teammate is True


# --------------------------------------------------------------------------- #
# GradingStage: who gets a colour
# --------------------------------------------------------------------------- #
class _PassThroughScorer:
    """Stands in for BodyArrayScorer: the bodies are pre-graded by the test."""

    def __init__(self, scorers):
        pass

    def process(self, img_w, img_h, bodies):
        return bodies


def _body(*, passed=False, reason="", score=0.0, color="N/A", lab=None) -> Body:
    return Body(
        crop=np.zeros((1, 1, 3), dtype=np.uint8), bbox=Box(0.1, 0.1, 0.3, 0.9),
        faces=[], keypoints=[], passed=passed, rejection_reason=reason,
        sharpness_score=score, cloth_color=color,
        cloth_color_detail={"mean_lab": lab} if lab else {},
    )


def _sharpness_reason(score: float, threshold: float = 0.5) -> str:
    return f"sharpness score {score:.4f} <= threshold {threshold:.2f}"


def _frame(*bodies: Body) -> Frame:
    return Frame(path=Path("IMG_0001.jpg"), bodies=list(bodies), img_w=100, img_h=100)


def test_grading_measures_colour_for_passed_and_extended_range_bodies_only(monkeypatch):
    monkeypatch.setattr(grading, "BodyArrayScorer", _PassThroughScorer)
    measured = []

    def fake_predict(body):
        measured.append(body)
        return "White:Bright", {"mean_lab": WHITE_LAB}

    monkeypatch.setattr(grading.cloth_color_predictor, "predict", fake_predict)

    passed = _body(passed=True, score=0.8)
    near = _body(reason=_sharpness_reason(0.35), score=0.35)
    far = _body(reason=_sharpness_reason(0.10), score=0.10)
    no_face = _body(reason="no matched face", score=0.0)
    wrong_gate_high_score = _body(reason="all faces too small", score=0.9)
    frame = _frame(passed, near, far, no_face, wrong_gate_high_score)

    GradingStage(0.5, ExtendedThreshold(0.5, 0.3)).process([frame], app_config)

    assert {id(b) for b in measured} == {id(passed), id(near)}
    assert near.cloth_color == "White:Bright"
    assert far.cloth_color == no_face.cloth_color == wrong_gate_high_score.cloth_color == "N/A"
    # The blurry verdict of the near-threshold body is untouched.
    assert near.passed is False and near.rejection_reason == _sharpness_reason(0.35)


def test_grading_without_an_extended_threshold_behaves_as_before(monkeypatch):
    monkeypatch.setattr(grading, "BodyArrayScorer", _PassThroughScorer)
    monkeypatch.setattr(grading.cloth_color_predictor, "predict",
                        lambda body: ("White:Bright", {"mean_lab": WHITE_LAB}))
    near = _body(reason=_sharpness_reason(0.35), score=0.35)
    GradingStage(0.5).process([_frame(near)], app_config)
    assert near.cloth_color == "N/A"


# --------------------------------------------------------------------------- #
# JerseyCountingStage: who is a teammate
# --------------------------------------------------------------------------- #
def _coloured(color, lab, **kw) -> Body:
    return _body(color=color, lab=lab, **kw)


def _team_frame():
    sharp_white = _coloured("White:Bright", WHITE_LAB, passed=True, score=0.8)
    sharp_pink = _coloured("Pink:Deep Magenta", PINK_LAB, passed=True, score=0.8)
    near_white = _coloured("White:Bright", WHITE_LAB, reason=_sharpness_reason(0.35), score=0.35)
    near_pink = _coloured("Pink:Deep Magenta", PINK_LAB, reason=_sharpness_reason(0.35), score=0.35)
    far = _body(reason=_sharpness_reason(0.1), score=0.1)  # never measured
    no_face = _body(reason="no matched face")
    frame = _frame(sharp_white, sharp_pink, near_white, near_pink, far, no_face)
    # Extra white sharp players so white is clearly the team colour of the day.
    extras = [_coloured("White:Bright", WHITE_LAB, passed=True, score=0.9) for _ in range(4)]
    frame.bodies.extend(extras)
    return frame, dict(sharp_white=sharp_white, sharp_pink=sharp_pink, near_white=near_white,
                       near_pink=near_pink, far=far, no_face=no_face, extras=extras)


def test_jersey_stage_flags_teammates_without_changing_blurry_verdicts():
    frame, b = _team_frame()
    JerseyCountingStage(frozenset(), frozenset({"White"})).process([frame], app_config)

    assert b["sharp_white"].is_teammate and b["sharp_white"].passed
    assert all(e.is_teammate and e.passed for e in b["extras"])
    assert not b["sharp_pink"].is_teammate and not b["sharp_pink"].passed
    assert b["sharp_pink"].rejection_reason.startswith("jersey")

    # Near-threshold: flagged by colour, verdict left exactly as the scorers set it.
    assert b["near_white"].is_teammate
    assert not b["near_pink"].is_teammate
    for key in ("near_white", "near_pink"):
        assert b[key].passed is False
        assert b[key].rejection_reason == _sharpness_reason(0.35)

    assert not b["far"].is_teammate
    assert not b["no_face"].is_teammate


def test_jersey_stage_polls_the_team_colour_from_extended_range_bodies_too():
    # Only one sharp player (pink) but several near-threshold white ones: with the
    # extended population voting, white is the team colour of the day.
    sharp_pink = _coloured("Pink:Deep Magenta", PINK_LAB, passed=True, score=0.8)
    near_whites = [
        _coloured("White:Bright", WHITE_LAB, reason=_sharpness_reason(0.35), score=0.35)
        for _ in range(3)
    ]
    stage = JerseyCountingStage(frozenset(), frozenset({"White", "Pink"}))
    stage.process([_frame(sharp_pink, *near_whites)], app_config)
    assert stage.our_color is not None and stage.our_color.label.startswith("White")
    assert all(b.is_teammate for b in near_whites)
    assert not sharp_pink.is_teammate


def test_jersey_stage_noteam_flags_every_measured_body():
    near = _coloured("Pink:Deep Magenta", PINK_LAB, reason=_sharpness_reason(0.35), score=0.35)
    sharp = _coloured("White:Bright", WHITE_LAB, passed=True, score=0.8)
    far = _body(reason=_sharpness_reason(0.1), score=0.1)
    JerseyCountingStage(frozenset(), frozenset(), no_team=True).process([_frame(near, sharp, far)], app_config)
    assert near.is_teammate and sharp.is_teammate
    assert not far.is_teammate
    assert near.passed is False  # verdict untouched


# --------------------------------------------------------------------------- #
# Quick regrade of an existing album
# --------------------------------------------------------------------------- #
def _album_body(*, score, blurry, reason="", color="N/A", lab=None, **extra) -> dict:
    return make_body_dict(
        sharpness_score=score, is_blurry=blurry, rejection_reason=reason,
        cloth_color=color, cloth_color_detail={"mean_lab": lab} if lab else {}, **extra,
    )


def _write_regrade_album(album_dir: Path, *, our_color: str = "White:Bright", **run_settings) -> Path:
    bodies = [
        _album_body(score=0.80, blurry=False, color="White:Bright", lab=WHITE_LAB),            # 0 sharp teammate
        _album_body(score=0.80, blurry=False, color="White:Bright", lab=WHITE_LAB),            # 1 sharp teammate
        _album_body(score=0.35, blurry=True, reason=_sharpness_reason(0.35),
                    color="White:Bright", lab=WHITE_LAB),                                       # 2 near, teammate
        _album_body(score=0.35, blurry=True, reason=_sharpness_reason(0.35),
                    color="Pink:Deep Magenta", lab=PINK_LAB),                                   # 3 near, opponent
        _album_body(score=0.10, blurry=True, reason=_sharpness_reason(0.10),
                    color="White:Bright", lab=WHITE_LAB),                                       # 4 below extended
        _album_body(score=0.0, blurry=True, reason="no matched face"),                          # 5 no face
    ]
    entry = make_entry(annotation_data={
        "processing_shape": [1080, 1920], "overall_blurry": False, "evaluated": bodies,
    })
    settings = {"sensitivity": "0.5", "jerseycolor": "White", **run_settings}
    write_album(album_dir, [entry], run_settings=settings, our_jersey_color=our_color)
    return album_dir


def test_regrade_writes_is_teammate_and_keeps_verdicts(album_dir):
    _write_regrade_album(album_dir)
    regrade_sensitivity(album_dir, 0.5)

    bodies = Album(album_dir).images["IMG_0001.jpg"].bodies
    assert [b.is_teammate for b in bodies] == [True, True, True, False, False, False]
    assert [b.passed for b in bodies] == [True, True, False, False, False, False]
    assert bodies[2].rejection_reason == _sharpness_reason(0.35)
    assert Album(album_dir).run_settings[RUN_SETTINGS_KEY] == 0.3


def test_regrade_honours_a_per_album_offset(album_dir):
    # Offset 0.45 -> extended threshold 0.05: the 0.10-score body is now in range.
    _write_regrade_album(album_dir, **{RUN_SETTINGS_KEY: 0.45})
    regrade_sensitivity(album_dir, 0.5)
    bodies = Album(album_dir).images["IMG_0001.jpg"].bodies
    assert bodies[4].is_teammate is True
    assert bodies[5].is_teammate is False  # failed an earlier gate: never in range
    assert Album(album_dir).run_settings[RUN_SETTINGS_KEY] == 0.45


def test_regrade_noteam_flags_every_body_in_range(album_dir):
    _write_regrade_album(album_dir, noteam=True)
    regrade_sensitivity(album_dir, 0.5)
    bodies = Album(album_dir).images["IMG_0001.jpg"].bodies
    assert [b.is_teammate for b in bodies] == [True, True, True, True, False, False]


def _regrade_log(album_dir, caplog, threshold=0.5):
    import logging

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="BlurPictureDetector"):
        summary = regrade_sensitivity(album_dir, threshold)
    return summary, caplog.text


def test_regrade_logs_every_is_teammate_change(album_dir, caplog):
    _write_regrade_album(album_dir)
    summary, text = _regrade_log(album_dir, caplog)

    # Bodies 1-3 become teammates (the flag was absent); 4-6 stay False, which is no change.
    for index, cloth in ((1, "White:Bright"), (2, "White:Bright"), (3, "White:Bright")):
        assert f"IMG_0001.jpg body {index}/6: is_teammate False -> True" in text
        assert f"cloth {cloth}" in text
    assert "body 4/6: is_teammate" not in text
    assert "body 6/6: is_teammate" not in text
    assert "teammates: 3 total (3 newly flagged, 0 no longer teammates" in text
    assert (summary.teammates, summary.teammates_added, summary.teammates_removed) == (3, 3, 0)


def test_regrade_logs_a_body_that_stops_being_a_teammate(album_dir, caplog):
    _write_regrade_album(album_dir)
    regrade_sensitivity(album_dir, 0.5)
    # The opponent-coloured body was wrongly flagged earlier; the regrade must say it fixed it.
    from algo.album import Album as _Album

    album = _Album(album_dir)
    album.images["IMG_0001.jpg"].bodies[3].is_teammate = True
    album.save()

    summary, text = _regrade_log(album_dir, caplog)
    assert "body 4/6: is_teammate True -> False" in text
    assert (summary.teammates_added, summary.teammates_removed) == (0, 1)


def test_regrade_logs_each_setting_it_changes(album_dir, caplog):
    _write_regrade_album(album_dir, our_color="Blue:Royal")
    _summary, text = _regrade_log(album_dir, caplog, threshold=0.55)

    assert "setting changed - sensitivity: '0.5' -> '0.55'" in text
    assert f"setting changed - {RUN_SETTINGS_KEY}: (not set) -> 0.3" in text
    assert "setting changed - our_jersey_color: 'Blue:Royal' -> 'White:Bright'" in text
    assert "team_color_override" not in text  # '' -> '' is not a change


def test_regrade_says_so_when_no_setting_changed(album_dir, caplog):
    _write_regrade_album(album_dir)
    regrade_sensitivity(album_dir, 0.5)
    _summary, text = _regrade_log(album_dir, caplog)
    assert "no settings changed" in text
    assert "is_teammate" not in text.replace("teammates:", "")  # nothing flipped on the second run


def test_regrade_notes_when_a_preset_name_becomes_a_number(album_dir, caplog):
    _write_regrade_album(album_dir, sensitivity="medium")
    _summary, text = _regrade_log(album_dir, caplog)
    assert "sensitivity: 'medium' -> '0.5' (the preset name is now stored as its numeric value)" in text


def test_regrade_logs_a_pinned_colour_change(album_dir, caplog):
    import logging

    _write_regrade_album(album_dir)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="BlurPictureDetector"):
        regrade_sensitivity(album_dir, 0.5, team_color_override="White:Bright")
    assert "setting changed - team_color_override: '' -> 'White:Bright'" in caplog.text
