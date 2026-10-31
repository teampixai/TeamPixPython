"""The "tone" engine shared by autoedit.py's tone pipeline and the culling
app's edit-mode preview."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import autoedit  # noqa: E402
from autoedit import ToneParams, apply_tone, propose_tone, tone_curve  # noqa: E402


def _gradient_image(width: int = 256, height: int = 32) -> np.ndarray:
    """Grey ramp with a coloured band, so hue preservation is checkable."""
    ramp = np.tile(np.linspace(0, 255, width, dtype=np.float32), (height, 1))
    image = np.stack([ramp, ramp, ramp], axis=-1)
    image[: height // 2, :, 0] *= 0.4  # warmer top half (less blue)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def _mean_lum(image: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> float:
    lum = autoedit._luminance(image)
    mask = (lum >= lo) & (lum <= hi)
    return float(lum[mask].mean())


def test_identity_params_leave_the_image_untouched():
    image = _gradient_image()
    assert apply_tone(image, ToneParams()) is image


def test_tone_params_are_clamped():
    p = ToneParams(highlights=5, midtones=-9, shadows=2.5, contrast=-300).clamped()
    assert p == ToneParams(highlights=2.0, midtones=-2.0, shadows=2.0, contrast=-100.0)


@pytest.mark.parametrize("zone, lo, hi", [
    ("shadows", 0.02, 0.25),
    ("midtones", 0.35, 0.65),
    ("highlights", 0.75, 0.95),
])
def test_each_zone_moves_its_own_tones_most(zone, lo, hi):
    image = _gradient_image()
    lum_before = autoedit._luminance(image)
    for ev, sign in ((1.0, 1), (-1.0, -1)):
        out = apply_tone(image, ToneParams(**{zone: ev}))
        delta = autoedit._luminance(out) - lum_before
        in_zone = (lum_before >= lo) & (lum_before <= hi)
        assert sign * delta[in_zone].mean() > 0.02
        # Strongest effect lands in the zone it is named after.
        others = ~in_zone & (lum_before > 0.02) & (lum_before < 0.98)
        assert abs(delta[in_zone].mean()) > abs(delta[others].mean())


def test_positive_contrast_spreads_and_negative_contrast_flattens():
    image = _gradient_image()
    def spread(img):
        return _mean_lum(img, 0.7, 1.0) - _mean_lum(img, 0.0, 0.3)
    # Measured on the source pixels' own positions, so compare the same columns.
    base = spread(image)
    assert spread(apply_tone(image, ToneParams(contrast=60))) > base
    assert spread(apply_tone(image, ToneParams(contrast=-60))) < base


def test_tone_curve_stays_monotonic_with_opposing_zones():
    lut = autoedit._tone_lut(ToneParams(shadows=2, midtones=-2, highlights=2, contrast=100))
    assert np.all(np.diff(lut) >= 0)
    assert lut.min() >= 0.0 and lut.max() <= 1.0


@pytest.mark.parametrize("params", [ToneParams(midtones=1.0), ToneParams(highlights=2.0), ToneParams(highlights=-1.0)])
def test_tone_curve_keeps_black_fixed_and_white_from_dimming(params):
    curve = tone_curve(params, np.array([0.0, 0.5, 1.0]))
    assert curve[0] == pytest.approx(0.0, abs=1e-6)
    if params.highlights >= 0:
        assert curve[2] == pytest.approx(1.0, abs=1e-6)
    assert curve[0] < curve[1] < curve[2]


def test_apply_tone_preserves_hue():
    image = _gradient_image()
    out = apply_tone(image, ToneParams(midtones=0.7))
    row = image.shape[0] // 4
    cols = slice(60, 180)  # avoid clipped ends
    before = image[row, cols].astype(np.float64)
    after = out[row, cols].astype(np.float64)
    # Blue:green ratio of the warm band is unchanged (within rounding).
    np.testing.assert_allclose(after[:, 0] / after[:, 1], before[:, 0] / before[:, 1], atol=0.03)


def test_apply_tone_is_resolution_independent():
    rng = np.random.default_rng(0)
    big = cv2.resize(rng.integers(0, 256, (40, 60, 3), dtype=np.uint8), (600, 400), interpolation=cv2.INTER_CUBIC)
    small = cv2.resize(big, (150, 100), interpolation=cv2.INTER_AREA)
    params = ToneParams(highlights=-0.5, midtones=0.4, shadows=0.8, contrast=25)
    big_then_small = cv2.resize(apply_tone(big, params), (150, 100), interpolation=cv2.INTER_AREA)
    small_out = apply_tone(small, params)
    assert np.abs(big_then_small.astype(int) - small_out.astype(int)).mean() < 2.0


def test_negative_contrast_undoes_positive_contrast():
    ramp = np.linspace(0.0, 1.0, 101)
    up = autoedit._contrast_curve(ramp, 0.6)
    np.testing.assert_allclose(autoedit._contrast_curve(up, -0.6), ramp, atol=1e-9)


def test_propose_tone_approximates_auto_level_and_brightness():
    rng = np.random.default_rng(3)
    base = cv2.resize(rng.integers(0, 256, (12, 18, 3), dtype=np.uint8), (360, 240), interpolation=cv2.INTER_CUBIC)
    for image in (
        (base * 0.7).astype(np.uint8),  # underexposed
        np.rint(255.0 * (base / 255.0) ** 0.7).astype(np.uint8),  # overexposed
    ):
        expected = autoedit._luminance(autoedit.auto_brightness(autoedit.auto_level(image)))
        got = autoedit._luminance(apply_tone(image, propose_tone(image)))
        assert np.abs(expected - got).mean() < 0.03


def test_propose_tone_brightens_a_dark_frame_and_darkens_a_bright_one():
    image = _gradient_image(512, 256)
    dark = (image * 0.35).astype(np.uint8)
    bright = np.clip(image.astype(np.int32) + 120, 0, 255).astype(np.uint8)
    assert propose_tone(dark).midtones > 0
    assert propose_tone(bright).midtones < 0


def test_propose_tone_values_are_in_range():
    rng = np.random.default_rng(1)
    for _ in range(5):
        image = rng.integers(0, 256, (120, 180, 3), dtype=np.uint8)
        p = propose_tone(image)
        assert p == p.clamped()
        assert p.highlights <= autoedit._PROPOSE_HIGHLIGHT_MAX_EV


def test_all_does_not_include_tone():
    names = [p.name for p in autoedit.resolve_pipelines(["all"])]
    assert "tone" not in names
    assert [p.name for p in autoedit.resolve_pipelines(["tone aisharpen"])] == ["tone", "aisharpen"]


def test_cli_tone_output_matches_apply_tone(tmp_path):
    rng = np.random.default_rng(2)
    image = cv2.resize(rng.integers(0, 256, (12, 16, 3), dtype=np.uint8), (160, 120), interpolation=cv2.INTER_CUBIC)
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), image)
    out = tmp_path / "out.png"
    subprocess.run(
        [sys.executable, str(REPO / "autoedit.py"), str(src), "-pipelines", "tone",
         "--out-file", str(out), "-ext", "png",
         "--highlights", "-0.5", "--midtones", "0.6", "--shadows", "1.2", "--contrast", "30"],
        check=True, cwd=str(REPO), capture_output=True,
    )
    expected = apply_tone(image, ToneParams(highlights=-0.5, midtones=0.6, shadows=1.2, contrast=30))
    np.testing.assert_array_equal(cv2.imread(str(out)), expected)
