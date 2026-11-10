"""LLMCullingStage: base stars, which photos are graded, and the 4/5-star
promotion (top grade percentile AND best inside a time window)."""

from __future__ import annotations

from pathlib import Path

import pytest

from algo.album import Album
from algo.llm.culling_provider import (
    BurstFrameInput, BurstRankingResult, CullingProvider, RankedFrame,
)
from algo.stages import llm_culling
from algo.stages.llm_culling import LLMCullingStage
from algo.star_rules import HighStarParams
from algo.thresholds import ExtendedThreshold
from conftest import make_entry, write_album

T0 = 5000.15  # offsets below never land on a whole second, which would be an invalid capture_ts
GRADES = {
    "A": 0.80, "B": 0.78, "C": 0.85, "D": 0.60, "F": 0.82,   # the clustered burst
    "S1": 0.90, "S2": 0.50,                                    # standalone
}


class FakeProvider(CullingProvider):
    model = "fake-model"

    def __init__(self) -> None:
        super().__init__()
        self.ranked: list[list[str]] = []
        self.graded: list[list[str]] = []

    def rank_burst(self, frames):
        names = [f.file for f in frames]
        self.ranked.append(names)
        best = max(names, key=lambda n: GRADES[n])
        return BurstRankingResult(
            rankings=[RankedFrame(best, 1, "best")],
            grades={n: GRADES[n] for n in names}, caption="cap",
        )

    def grade_images(self, frames):
        names = [f.file for f in frames]
        self.graded.append(names)
        return {n: GRADES[n] for n in names}


def _entry(key: str, offset: float, status: str = "sharp", score: float = 0.62, **extra) -> dict:
    return make_entry(key, status=status, sharpness_score=score, capture_ts=T0 + offset, **extra)


def _build_album(album_dir: Path) -> Path:
    entries = [
        # One burst (gaps <= 1s, spans 2s): A B C D(blurry, 2 star) F; E is blurry below the extended threshold.
        _entry("A", 0.00), _entry("B", 0.20), _entry("C", 0.30),
        _entry("D", 0.90, "blurry", 0.45), _entry("E", 1.10, "blurry", 0.20), _entry("F", 1.80),
        # Standalone photos, far apart from each other.
        _entry("S1", 1000.0), _entry("S2", 2000.0, "blurry", 0.50), _entry("S3", 3000.0, "blurry", 0.10),
        _entry("NOFACE", 4000.0, "skipped", 0.0),
        # Rated by hand: must not be touched.
        _entry("M", 5000.0, stars=1, stars_manual=True, keep=False),
    ]
    return write_album(album_dir, entries, run_settings={"sensitivity": "0.5"})


@pytest.fixture
def stage_env(monkeypatch):
    monkeypatch.setattr(
        LLMCullingStage, "_build_frame_input",
        lambda self, image: BurstFrameInput(image.key, "", None, 0.0),
    )
    # "M" is sharp, so it is graded too (low, so it never wins the standalone pool).
    GRADES.setdefault("M", 0.10)
    yield


def _run(album_dir: Path, **kwargs):
    provider = FakeProvider()
    stage = LLMCullingStage(
        album_dir, provider=provider, threshold=0.5, burst_gap_seconds=1.0, **kwargs,
    )
    stage.process([], None)
    return provider, Album(album_dir)


def test_only_sharp_and_blurry_above_the_extended_threshold_are_graded(album_dir, stage_env):
    _build_album(album_dir)
    provider, _ = _run(album_dir)

    assert sorted(n for batch in provider.ranked for n in batch) == ["A", "B", "C", "D", "F"]
    assert sorted(n for batch in provider.graded for n in batch) == ["M", "S1", "S2"]


def test_stars_and_keep_follow_the_new_rules(album_dir, stage_env):
    _build_album(album_dir)
    _, album = _run(
        album_dir,
        clustered_star_params=HighStarParams(0.5, 0.5, 0.05, 1.5),
    )
    stars = {i.key: i.stars for i in album.image_list}

    assert stars["C"] == 5 and stars["F"] == 4     # 5 stars first; F is then the best of the rest
    assert stars["A"] == 4                         # C (5 stars) is out of the 4-star pool, so no longer blocks A
    assert stars["B"] == 3                         # beaten by A within 0.5s: stays at the sharp base
    assert stars["D"] == 2                         # blurry above the extended threshold, not promoted
    assert stars["E"] == 1                         # blurry below it: never graded
    assert stars["S1"] == 5 and stars["S2"] == 4   # standalone pool: S1 first, then S2 from the rest
    assert stars["S3"] == 1 and stars["NOFACE"] == 1
    assert stars["M"] == 1                          # hand-rated photo untouched

    keep = {i.key: i.keep for i in album.image_list}
    assert keep["C"] and keep["F"] and keep["A"] and keep["B"] and keep["S1"] and keep["S2"]
    assert not keep["D"] and not keep["E"] and not keep["M"]


def test_default_params_give_a_five_star_then_a_four_star_pick_for_a_small_burst(album_dir, stage_env):
    _build_album(album_dir)
    _, album = _run(album_dir)
    stars = {i.key: i.stars for i in album.image_list}
    # Pool of 5 graded frames: C is the 5-star pick; F is the best of the remaining 4 (ceil(4 * 0.2) = 1).
    assert [stars[k] for k in "ABCDF"] == [3, 3, 5, 2, 4]


def test_a_two_star_photo_can_be_promoted(album_dir, stage_env, monkeypatch):
    _build_album(album_dir)
    monkeypatch.setitem(GRADES, "D", 0.99)
    _, album = _run(album_dir)
    assert album.image("D").stars == 5 and album.image("D").keep is True


def test_ungraded_photos_keep_their_base_stars(album_dir, stage_env):
    _build_album(album_dir)

    class Failing(FakeProvider):
        def rank_burst(self, frames):
            return BurstRankingResult(rankings=[], grades={}, caption="")

        def grade_images(self, frames):
            return {}

    stage = LLMCullingStage(album_dir, provider=Failing(), threshold=0.5)
    stage.process([], None)
    stars = {i.key: i.stars for i in Album(album_dir).image_list}
    assert [stars[k] for k in ("A", "C", "F", "S1")] == [3, 3, 3, 3]
    assert stars["D"] == 2 and stars["S2"] == 2 and stars["E"] == 1


def test_model_and_cost_are_recorded(album_dir, stage_env):
    _build_album(album_dir)
    _, album = _run(album_dir)
    assert album.llm_model == "fake-model"


def test_whole_second_capture_times_are_refreshed_from_exif(album_dir, stage_env, monkeypatch):
    entries = [make_entry("A", capture_ts=1000.0), make_entry("B", capture_ts=1000.25)]
    write_album(album_dir, entries, run_settings={"sensitivity": "0.5"})
    reads: list[str] = []

    def fake_read(path):
        reads.append(Path(path).name)
        return 7777.5

    monkeypatch.setattr(llm_culling, "image_capture_timestamp", fake_read)
    GRADES.setdefault("A", 0.8)
    _run(album_dir)

    assert reads == ["A"]  # only the whole-second one is invalid
    saved = Album(album_dir)
    assert saved.image("A").capture_ts == 7777.5 and saved.image("B").capture_ts == 1000.25


def test_extended_threshold_is_derived_from_the_album_unless_given(album_dir, stage_env):
    # Sensitivity 0.5 / default offset 0.2 / floor 0.3: a 0.31 blurry photo is graded, a 0.30 one is not.
    entries = [
        _entry("D", 0.0, "blurry", 0.31), _entry("E", 500.0, "blurry", 0.30),
    ]
    write_album(album_dir, entries, run_settings={"sensitivity": "0.5"})
    GRADES.setdefault("E", 0.5)
    GRADES["D"] = 0.6
    provider, album = _run(album_dir)
    assert [n for b in provider.graded for n in b] == ["D"]
    assert album.image("D").stars == 5 and album.image("E").stars == 1

    # An explicit extended threshold wins.
    provider2, _ = _run(album_dir, extended=ExtendedThreshold(0.5, 0.5, 0.0))  # 0.0 -> E now graded
    assert sorted(n for b in provider2.graded for n in b) == ["D", "E"]
