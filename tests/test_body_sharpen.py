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
