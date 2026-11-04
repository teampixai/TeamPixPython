"""Body-only sharpening: which players get sharpened, and that the sharpened
region stays clipped to the target's own box."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import autoedit  # noqa: E402
import culling_app  # noqa: E402
from algo.album import PersonRecord  # noqa: E402


def _person(bbox: dict | None, *, teammate: bool | None = None) -> PersonRecord:
    wire = {"body_bbox": bbox}
    if teammate is not None:  # None = the key is absent, as in an older album
        wire["is_teammate"] = teammate
    return PersonRecord(wire)


def _box(x1, y1, x2, y2) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_edit_session_bodies_lists_every_boxed_body_with_its_index_and_flag():
    bodies = [
        _person(_box(0.1, 0.1, 0.3, 0.9), teammate=True),
        _person(None, teammate=True),
        _person(_box(0.5, 0.1, 0.7, 0.9), teammate=False),
        _person(_box(0.8, 0.2, 0.9, 0.8)),  # older album: no flag
    ]
    assert culling_app._edit_session_bodies(bodies) == [
        {"id": 0, "box": [0.1, 0.1, 0.3, 0.9], "isTeammate": True},
        {"id": 2, "box": [0.5, 0.1, 0.7, 0.9], "isTeammate": False},
        {"id": 3, "box": [0.8, 0.2, 0.9, 0.8], "isTeammate": False},
    ]


def test_selected_body_entries_keeps_only_selected_bodies():
    bodies = [
        _person(_box(0.1, 0.1, 0.3, 0.9), teammate=True),
        _person(_box(0.5, 0.1, 0.7, 0.9), teammate=False),
        _person(_box(0.8, 0.2, 0.9, 0.8), teammate=True),
    ]
    # Selection, not the teammate flag, decides — and order/duplicates don't matter.
    assert culling_app._selected_body_entries(bodies, [2, 1, 2]) == [
        {"box": [0.5, 0.1, 0.7, 0.9], "keypoints": []},
        {"box": [0.8, 0.2, 0.9, 0.8], "keypoints": []},
    ]
    assert culling_app._selected_body_entries(bodies, []) == []


def test_selected_body_entries_skips_unknown_ids_and_bodies_without_a_box():
    bodies = [_person(None, teammate=True), _person(_box(0.1, 0.1, 0.3, 0.9))]
    assert culling_app._selected_body_entries(bodies, [-1, 0, 5]) == []
    assert culling_app._selected_body_entries(bodies, [1]) == [{"box": [0.1, 0.1, 0.3, 0.9], "keypoints": []}]


def test_selected_body_entries_includes_keypoints_when_stored():
    wire = {
        "body_bbox": _box(0.1, 0.1, 0.3, 0.9),
        "is_teammate": True,
        "body_keypoints": [{"x": 0.2, "y": 0.2, "conf": 0.9}],
    }
    assert culling_app._selected_body_entries([PersonRecord(wire)], [0]) == [
        {"box": [0.1, 0.1, 0.3, 0.9], "keypoints": [[0.2, 0.2, 0.9]]},
    ]


@pytest.mark.parametrize("settings,expected", [
    ({"sensitivity": "medium"}, 0.5),
    ({"sensitivity": "low"}, 0.4),
    ({"sensitivity": "0.55"}, 0.55),
    ({"sensitivity": "garbage"}, 0.5),
    ({}, 0.5),
])
def test_album_sharpness_threshold(settings, expected):
    assert culling_app._album_sharpness_threshold(settings) == expected


class _FakeDispatcher:
    """Stands in for SharpenDispatcher: 'sharpening' paints the crop white."""

    def submit(self, crop):
        return np.full_like(crop, 255)


class _EverythingIsPerson:
    """Segmenter that (like the real one) marks every pixel in the crop as a
    person, standing in for a neighbour inside the padded crop."""

    def segment(self, image_bgr):
        return np.ones(image_bgr.shape[:2], dtype=np.float32)


def test_sharpen_bodies_leaves_neighbour_outside_target_box_untouched(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    # Target box covers x 400..500; its 20% padded crop reaches x 380..520.
    target = (0.40, 0.25, 0.50, 0.75)

    result = autoedit.sharpen_bodies(image, [target], _FakeDispatcher())

    inside = result[200, 450]      # centre of the target
    neighbour = result[200, 515]   # inside the padded crop, outside the box
    far = result[200, 100]         # nowhere near it
    assert inside.min() > 240
    assert neighbour.max() < 40
    assert far.max() == 0


def test_sharpen_bodies_no_boxes_returns_identical_image(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    image = np.random.default_rng(0).integers(0, 255, (50, 80, 3), dtype=np.uint8)
    result = autoedit.sharpen_bodies(image, [], _FakeDispatcher())
    assert np.array_equal(result, image)


class _RaisingSegmenter:
    def segment(self, image_bgr):
        raise RuntimeError("model exploded")


class _NobodyIsPerson:
    def segment(self, image_bgr):
        return np.zeros(image_bgr.shape[:2], dtype=np.float32)


class _CountingDispatcher(_FakeDispatcher):
    def __init__(self):
        self.calls = 0

    def submit(self, crop):
        self.calls += 1
        return super().submit(crop)


@pytest.mark.parametrize("segmenter,expected_text", [
    (_RaisingSegmenter(), "segmentation FAILED (RuntimeError: model exploded)"),
    (_NobodyIsPerson(), "segmentation FAILED - no person found"),
])
def test_sharpen_bodies_logs_failed_segmentation_and_skips_box(monkeypatch, caplog, segmenter, expected_text):
    import logging
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: segmenter)
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    dispatcher = _CountingDispatcher()

    with caplog.at_level(logging.INFO, logger="autoedit"):
        result = autoedit.sharpen_bodies(image, [(0.40, 0.25, 0.50, 0.75)], dispatcher, label="photo.jpg")

    assert np.array_equal(result, image)
    assert dispatcher.calls == 0  # segmentation fails first, so no wasted sharpen pass
    text = caplog.text
    assert "[photo.jpg] Body sharpen: 1 box(es) passed in" in text
    assert "norm=(0.400, 0.250, 0.500, 0.750)" in text
    assert "body_px=(400, 100)-(500, 300)" in text
    assert expected_text in text
    assert "0/1 box(es) sharpened, 1 skipped" in text


def test_sharpen_bodies_logs_successful_segmentation(monkeypatch, caplog):
    import logging
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    image = np.zeros((400, 1000, 3), dtype=np.uint8)

    with caplog.at_level(logging.INFO, logger="autoedit"):
        autoedit.sharpen_bodies(image, [(0.40, 0.25, 0.50, 0.75)], _FakeDispatcher(), label="photo.jpg")

    assert "box 1/1" in caplog.text
    assert "segmentation OK" in caplog.text
    assert "1/1 box(es) sharpened, 0 skipped" in caplog.text


# --- keypoint-assisted component selection (_select_body_component) ------- #

def test_select_body_component_keeps_only_the_voted_blob():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 1:4] = 0.9  # blob A
    mask[2:5, 6:9] = 0.9  # blob B, disconnected from A

    kept, info = autoedit._select_body_component(mask, [(2, 3, 0.9)])  # inside blob A only

    assert info == {"n_keypoints": 1, "n_components": 2, "n_kept": 1}
    assert kept[3, 2] == pytest.approx(0.9)  # blob A kept
    assert kept[3, 7] == 0.0  # blob B zeroed out


def test_select_body_component_falls_back_when_no_confident_keypoints():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 1:4] = 0.9

    kept, info = autoedit._select_body_component(mask, [(2, 3, 0.1)])  # below conf threshold

    assert info is None
    assert np.array_equal(kept, mask)


def test_select_body_component_falls_back_when_keypoints_miss_every_blob():
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 1:4] = 0.9

    kept, info = autoedit._select_body_component(mask, [(9, 9, 0.9)])  # lands on background

    assert info is None
    assert np.array_equal(kept, mask)


class _TwoBlobsSegmenter:
    """Returns two disconnected 'person' blobs inside the crop, both within
    the target box's feathered box-clip region (see
    BODY_SHARPEN_CLIP_MARGIN_PCT): crop-local columns 20-60 (the target's
    own silhouette) and 90-120 (a neighbour standing close enough to also
    fall inside the same padded crop and box-clip)."""

    def segment(self, image_bgr):
        mask = np.zeros(image_bgr.shape[:2], dtype=np.float32)
        mask[:, 20:60] = 1.0
        mask[:, 90:120] = 1.0
        return mask


def test_sharpen_bodies_without_keypoints_neighbour_blob_inside_box_bleeds_through(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _TwoBlobsSegmenter())
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    target = (0.40, 0.25, 0.50, 0.75)

    result = autoedit.sharpen_bodies(image, [target], _FakeDispatcher())

    # Neighbour's blob (crop-local cols 90-120 -> full-image cols 470-500)
    # sits inside the box-clip's feathered region, so with no keypoints to
    # tell the two blobs apart it bleeds through and gets sharpened too.
    neighbour_px = result[200, 485]
    assert neighbour_px.min() > 100


def test_sharpen_bodies_keypoints_exclude_neighbour_blob_inside_the_box(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _TwoBlobsSegmenter())
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    target = (0.40, 0.25, 0.50, 0.75)
    # One confident keypoint inside the target's own blob (crop-local col 40,
    # row 150 -> full-image (420, 210)); nowhere near the neighbour's blob.
    keypoints = [[420 / 1000, 210 / 400, 0.9]]

    result = autoedit.sharpen_bodies(image, [target], _FakeDispatcher(), keypoints_norm=[keypoints])

    target_px = result[210, 420]
    neighbour_px = result[200, 485]
    assert target_px.min() > 240  # target still fully sharpened
    assert neighbour_px.max() == 0  # neighbour's blob now fully excluded


def test_sharpen_bodies_fills_debug_layers_with_silhouette_and_sharpened(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _TwoBlobsSegmenter())
    image = np.full((400, 1000, 3), 7, dtype=np.uint8)
    target = (0.40, 0.25, 0.50, 0.75)
    keypoints = [[420 / 1000, 210 / 400, 0.9]]  # on the target's own blob only
    debug = {}

    result = autoedit.sharpen_bodies(image, [target], _FakeDispatcher(), keypoints_norm=[keypoints], debug_layers=debug)

    assert set(debug) == {"silhouette", "sharpened"}
    silhouette, sharpened = debug["silhouette"], debug["sharpened"]
    assert silhouette.shape == image.shape[:2] and silhouette.dtype == np.float32
    assert sharpened.shape == image.shape and sharpened.dtype == np.uint8
    assert silhouette[200, 485] == 0.0   # the neighbour's blob isn't in the blend mask
    assert silhouette[210, 420] > 0.9
    assert silhouette[0, 0] == 0.0       # outside every crop
    # The sharpened layer holds the unmasked sharpened crop (the fake paints it white),
    # and the untouched original everywhere else...
    assert sharpened[200, 485].min() == 255
    assert np.array_equal(sharpened[0, 0], image[0, 0])
    # ...and blending it over the original with the silhouette reproduces the result.
    alpha = np.clip(silhouette, 0, 1)[..., None]
    rebuilt = sharpened.astype(np.float32) * alpha + image.astype(np.float32) * (1 - alpha)
    np.testing.assert_allclose(rebuilt, result, atol=1.0)


def test_sharpen_bodies_without_debug_layers_is_unchanged(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    image = np.zeros((400, 1000, 3), dtype=np.uint8)
    a = autoedit.sharpen_bodies(image, [(0.40, 0.25, 0.50, 0.75)], _FakeDispatcher())
    b = autoedit.sharpen_bodies(image, [(0.40, 0.25, 0.50, 0.75)], _FakeDispatcher(), debug_layers={})
    assert np.array_equal(a, b)


def test_process_one_saves_the_four_debug_images_at_full_resolution(monkeypatch, tmp_path):
    import cv2
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    src = tmp_path / "IMG_0001.png"
    original = np.full((400, 1000, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(src), original)
    debug_dir = tmp_path / "debug" / "esrgan_silhouette"

    out = autoedit.process_one(
        src, [autoedit.AiSharpenPipeline()], tmp_path / "out", "png", 95,
        _FakeDispatcher(), None, out_file=tmp_path / "result.png",
        body_boxes=[(0.40, 0.25, 0.50, 0.75)], silhouette_dir=debug_dir,
    )

    assert out is not None
    assert sorted(p.name for p in debug_dir.iterdir()) == [
        "IMG_0001_final.jpg", "IMG_0001_orig.jpg", "IMG_0001_sharpened.jpg", "IMG_0001_silhouette.png",
    ]
    for name in ("IMG_0001_final.jpg", "IMG_0001_orig.jpg", "IMG_0001_sharpened.jpg", "IMG_0001_silhouette.png"):
        assert cv2.imread(str(debug_dir / name), cv2.IMREAD_UNCHANGED).shape[:2] == (400, 1000)
    assert abs(int(cv2.imread(str(debug_dir / "IMG_0001_orig.jpg"))[10, 10, 0]) - 40) <= 1
    assert cv2.imread(str(debug_dir / "IMG_0001_silhouette.png"), cv2.IMREAD_UNCHANGED)[200, 450] == 255
    assert cv2.imread(str(debug_dir / "IMG_0001_final.jpg"))[200, 450].min() > 240


def test_process_one_saves_nothing_without_silhouette_dir_or_when_nothing_is_sharpened(monkeypatch, tmp_path):
    import cv2
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _NobodyIsPerson())
    src = tmp_path / "IMG_0002.png"
    cv2.imwrite(str(src), np.full((100, 200, 3), 40, dtype=np.uint8))
    debug_dir = tmp_path / "debug"

    autoedit.process_one(src, [autoedit.AiSharpenPipeline()], tmp_path / "out", "png", 95, _FakeDispatcher(), None,
                         out_file=tmp_path / "r.png", body_boxes=[(0.4, 0.25, 0.5, 0.75)], silhouette_dir=debug_dir)
    assert not debug_dir.exists()  # segmentation found nobody -> nothing sharpened -> nothing saved


# --- unselected ("protected") players -------------------------------------- #

def _standing_keypoints(cx: float, top: float, bottom: float, w: int, h: int) -> list[list[float]]:
    """17 confident COCO keypoints of a person standing upright at column
    *cx* (px) between rows *top* and *bottom* (px), normalised to w x h."""
    ys = np.linspace(top, bottom, 17)
    return [[cx / w, y / h, 0.95] for y in ys]


def test_sharpen_bodies_leaves_protected_neighbour_inside_the_box_untouched(monkeypatch):
    import algo.segmentation as segmentation

    # Everything is "person": a single merged blob, like two touching players.
    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    w, h = 1000, 400
    image = np.zeros((h, w, 3), dtype=np.uint8)
    target = (0.40, 0.25, 0.60, 0.75)            # x 400..600
    neighbour = (0.52, 0.30, 0.70, 0.75)         # x 520..700, overlaps the target's right side
    target_kps = _standing_keypoints(440, 110, 290, w, h)
    neighbour_kps = _standing_keypoints(600, 130, 290, w, h)

    result = autoedit.sharpen_bodies(
        image, [target], _FakeDispatcher(), keypoints_norm=[target_kps],
        protected_norm=[(neighbour, neighbour_kps)],
    )

    assert result[200, 440].min() > 240   # the selected player is sharpened
    assert result[200, 590].max() < 20    # neighbour's body, inside the target box: untouched
    # Without protection the same pixel would have been sharpened.
    unprotected = autoedit.sharpen_bodies(image, [target], _FakeDispatcher(), keypoints_norm=[target_kps])
    assert unprotected[200, 590].min() > 240


def _person_keypoints(cx: float, top: float, height: float, w: int, h: int) -> list[list[float]]:
    """17 confident COCO keypoints of a person standing facing the camera,
    horizontally centred on *cx* (px), spanning *height* px from *top*,
    normalised to a w x h image."""
    layout = [  # (dx, y), as fractions of the height
        (0, .07), (-.02, .06), (.02, .06), (-.04, .07), (.04, .07),
        (-.11, .20), (.11, .20), (-.14, .35), (.14, .35), (-.15, .48), (.15, .48),
        (-.07, .50), (.07, .50), (-.07, .72), (.07, .72), (-.07, .93), (.07, .93),
    ]
    return [[(cx + dx * height) / w, (top + y * height) / h, 0.95] for dx, y in layout]


def test_same_person_tells_a_duplicate_from_a_neighbour():
    mine = [(x * 1000, y * 400, c) for x, y, c in _person_keypoints(460, 40, 320, 1000, 400)]
    duplicate = [(x * 1000, y * 400, c) for x, y, c in _person_keypoints(480, 45, 320, 1000, 400)]
    neighbour = [(x * 1000, y * 400, c) for x, y, c in _person_keypoints(560, 40, 320, 1000, 400)]
    assert autoedit._same_person(mine, duplicate, 320)
    assert not autoedit._same_person(mine, neighbour, 320)
    assert not autoedit._same_person(mine, [], 320)


def test_protected_duplicate_detection_of_the_target_does_not_carve_it_up(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    w, h = 1000, 400
    image = np.zeros((h, w, 3), dtype=np.uint8)
    target = (0.40, 0.10, 0.52, 0.90)
    duplicate = (0.42, 0.11, 0.54, 0.91)  # the same player detected twice, shifted 20 px
    kps = _person_keypoints(460, 40, 320, w, h)
    dup_kps = _person_keypoints(480, 44, 320, w, h)

    result = autoedit.sharpen_bodies(
        image, [target], _FakeDispatcher(), keypoints_norm=[kps], protected_norm=[(duplicate, dup_kps)],
    )

    # Across the whole torso, including the side the duplicate is deeper on.
    for x in (430, 460, 495):
        assert result[150, x].min() > 240, x


def test_sharpening_is_limited_to_the_targets_own_body_when_others_are_nearby(monkeypatch):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name=None: _EverythingIsPerson())
    w, h = 1000, 400
    image = np.zeros((h, w, 3), dtype=np.uint8)
    target = (0.38, 0.10, 0.54, 0.90)
    neighbour = (0.52, 0.10, 0.64, 0.90)
    kps = _person_keypoints(460, 40, 320, w, h)
    other_kps = _person_keypoints(580, 40, 320, w, h)

    result = autoedit.sharpen_bodies(
        image, [target], _FakeDispatcher(), keypoints_norm=[kps], protected_norm=[(neighbour, other_kps)],
    )

    assert result[62, 460].min() > 240   # the target's head
    assert result[150, 460].min() > 240  # torso
    # A corner of the target's box that its body doesn't reach (and the segmenter
    # called "person" anyway, as it would for a merged neighbour) stays untouched.
    assert result[60, 385].max() < 40


def test_protected_keep_mask_is_none_when_nobody_overlaps():
    keep = autoedit._protected_keep_mask(
        (100, 100), (10, 10, 50, 90), [], [((200, 10, 260, 90), [])],
    )
    assert keep is None


def test_parse_boxes_file_splits_protected_entries():
    raw = {"a.jpg": [
        [0.1, 0.1, 0.2, 0.2],
        {"box": [0.3, 0.3, 0.4, 0.4], "keypoints": [[0.35, 0.35, 0.9]]},
        {"box": [0.5, 0.5, 0.6, 0.6], "keypoints": [], "protect": True},
    ]}
    boxes, kps, protected = autoedit._parse_boxes_file(raw)
    assert boxes == {"a.jpg": [(0.1, 0.1, 0.2, 0.2), (0.3, 0.3, 0.4, 0.4)]}
    assert kps == {"a.jpg": [[], [(0.35, 0.35, 0.9)]]}
    assert protected == {"a.jpg": [((0.5, 0.5, 0.6, 0.6), [])]}


def test_protected_body_entries_lists_every_unselected_boxed_body():
    bodies = [
        _person(_box(0.1, 0.1, 0.3, 0.9), teammate=True),
        _person(None),
        _person(_box(0.5, 0.1, 0.7, 0.9), teammate=True),
        _person(_box(0.8, 0.2, 0.9, 0.8)),
    ]
    assert culling_app._protected_body_entries(bodies, [0]) == [
        {"box": [0.5, 0.1, 0.7, 0.9], "keypoints": [], "protect": True},
        {"box": [0.8, 0.2, 0.9, 0.8], "keypoints": [], "protect": True},
    ]
