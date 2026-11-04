"""Edit window: cropping (autoedit.py --crop, crop bookkeeping on the album
entry, remapping stored detections onto a cropped edit) and the Other tab's
Revert / Edit-with-custom-app endpoints."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import autoedit
import culling_app
from algo import album as album_module
from algo.album import PersonRecord, album_for
from conftest import make_entry, write_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient

KEY = "IMG_0001.jpg"


# ---- autoedit.py ------------------------------------------------------------ #

def test_parse_crop_reads_four_numbers():
    assert autoedit.parse_crop("10, 20,300,400") == (10, 20, 300, 400)
    with pytest.raises(Exception):
        autoedit.parse_crop("1,2,3")
    with pytest.raises(Exception):
        autoedit.parse_crop("1,2,0,4")


def test_crop_image_cuts_the_rectangle():
    img = np.arange(10 * 20 * 3, dtype=np.uint8).reshape(10, 20, 3)
    out = autoedit.crop_image(img, (5, 2, 8, 6))
    assert out.shape == (6, 8, 3)
    assert np.array_equal(out, img[2:8, 5:13])


def test_crop_image_clamps_to_the_image():
    img = np.zeros((10, 20, 3), np.uint8)
    assert autoedit.crop_image(img, (15, 8, 100, 100)).shape == (2, 5, 3)


def test_crop_image_scales_when_a_pipeline_resized_the_image():
    img = np.zeros((20, 40, 3), np.uint8)  # source was 20x10, now 2x
    assert autoedit.crop_image(img, (5, 2, 8, 6), source_size=(20, 10)).shape == (12, 16, 3)


def test_process_one_crops_after_the_pipelines(tmp_path):
    src = tmp_path / "in.jpg"
    assert cv2.imwrite(str(src), np.full((100, 200, 3), 128, np.uint8))
    out = tmp_path / "out.jpg"
    result = autoedit.process_one(
        src, [autoedit.TonePipeline()], tmp_path, "jpg", 95, None, None,
        out_file=out, tone=autoedit.ToneParams(), crop=(20, 10, 50, 40),
    )
    assert result == out
    assert cv2.imread(str(out)).shape[:2] == (40, 50)


# ---- crop geometry ---------------------------------------------------------- #

def test_compose_crop_nests_inner_inside_outer():
    assert culling_app._compose_crop(None, None) is None
    assert culling_app._compose_crop((0.1, 0.2, 0.5, 0.6), None) == (0.1, 0.2, 0.5, 0.6)
    assert culling_app._compose_crop(None, (0.1, 0.2, 0.5, 0.6)) == (0.1, 0.2, 0.5, 0.6)
    composed = culling_app._compose_crop((0.2, 0.0, 0.6, 0.5), (0.5, 0.5, 1.0, 1.0))
    assert composed == pytest.approx((0.4, 0.25, 0.6, 0.5))


def test_box_in_crop_maps_and_clips():
    crop = (0.5, 0.0, 1.0, 0.5)
    assert culling_app._box_in_crop((0.6, 0.1, 0.8, 0.3), crop) == pytest.approx((0.2, 0.2, 0.6, 0.6))
    # Partly outside: clipped to the crop's edge.
    assert culling_app._box_in_crop((0.4, 0.1, 0.6, 0.3), crop) == pytest.approx((0.0, 0.2, 0.2, 0.6))
    # Entirely outside: gone.
    assert culling_app._box_in_crop((0.1, 0.1, 0.3, 0.3), crop) is None


def _person(box, kps=()):
    wire = {"body_bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]}, "is_teammate": True,
            "body_keypoints": [{"x": x, "y": y, "conf": c} for x, y, c in kps]}
    return PersonRecord(wire)


def test_body_helpers_map_onto_a_cropped_edit():
    crop = (0.5, 0.0, 1.0, 1.0)
    bodies = [
        _person((0.1, 0.1, 0.3, 0.9)),                              # cropped out
        _person((0.6, 0.1, 0.8, 0.9), [(0.7, 0.5, 0.9), (0.2, 0.5, 0.8)]),
        _person((0.75, 0.1, 0.95, 0.9)),
    ]
    sessions = culling_app._edit_session_bodies(bodies, crop)
    assert [b["id"] for b in sessions] == [1, 2]
    assert sessions[0]["box"] == pytest.approx([0.2, 0.1, 0.6, 0.9])

    selected = culling_app._selected_body_entries(bodies, [0, 1], crop)
    assert len(selected) == 1
    assert selected[0]["box"] == pytest.approx([0.2, 0.1, 0.6, 0.9])
    # The keypoint left of the crop is dropped, the other remapped.
    assert selected[0]["keypoints"] == [pytest.approx([0.4, 0.5, 0.9])]

    protected = culling_app._protected_body_entries(bodies, [0, 1], crop)
    assert [p["box"] for p in protected] == [pytest.approx([0.5, 0.1, 0.9, 0.9])]


def test_clamp_crop_px():
    rect = culling_app.CropRect
    assert culling_app._clamp_crop_px(None, 100, 50) is None
    assert culling_app._clamp_crop_px(rect(x=0, y=0, w=100, h=50), 100, 50) is None
    assert culling_app._clamp_crop_px(rect(x=-5, y=0, w=200, h=50), 100, 50) is None
    assert culling_app._clamp_crop_px(rect(x=10, y=5, w=200, h=20), 100, 50) == (10, 5, 90, 20)


# ---- album bookkeeping ------------------------------------------------------ #

def _make_source(root: Path, name: str = KEY, shape=(60, 80)) -> Path:
    fp = root / "src" / name
    fp.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((*shape, 3), np.uint8)
    img[:] = (60, 120, 180)
    assert cv2.imwrite(str(fp), img)
    return fp


def _with_edit(album_dir: Path, tmp_path: Path, crop=None) -> Path:
    edited = album_dir / "editedimages" / "IMG_0001.jpg"
    entry = make_entry(KEY, file=str(_make_source(tmp_path)), edited_image="editedimages/IMG_0001.jpg")
    if crop is not None:
        entry["edited_crop"] = list(crop)
    write_album(album_dir, [entry])
    edited.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(edited), np.zeros((30, 40, 3), np.uint8))
    album_module.invalidate_all()
    return edited


def test_edited_crop_is_only_reported_while_the_edit_exists(album_dir, tmp_path):
    edited = _with_edit(album_dir, tmp_path, crop=(0.1, 0.2, 0.6, 0.7))
    image = album_for(album_dir).image(KEY)
    assert image.edited_crop == pytest.approx((0.1, 0.2, 0.6, 0.7))
    edited.unlink()
    assert image.edited_crop is None


def test_set_and_clear_edited(album_dir, tmp_path):
    edited = _with_edit(album_dir, tmp_path)
    album = album_for(album_dir)
    image = album.image(KEY)
    assert image.edited_crop is None
    image.set_edited_crop((0.0, 0.0, 0.5, 0.5))
    album.save()
    assert album_for(album_dir).image(KEY).edited_crop == (0.0, 0.0, 0.5, 0.5)

    assert image.clear_edited() == edited
    album.save()
    image = album_for(album_dir).image(KEY)
    assert image.edited_path is None and image.edited_crop is None


# ---- endpoints -------------------------------------------------------------- #

@pytest.fixture
def client(album_dir, monkeypatch):
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    yield TestClient(culling_app.app)
    culling_app.edit_session_state.clear()
    culling_app.ai_edit_state.clear()


def test_session_maps_bodies_onto_a_cropped_edit(album_dir, tmp_path, client):
    _with_edit(album_dir, tmp_path, crop=(0.0, 0.0, 0.5, 1.0))
    data = client.post("/api/edit/ai-edit/session", json={"file": KEY}).json()
    assert data["edited"] is True
    assert (data["width"], data["height"]) == (40, 30)
    # conftest's body sits at x 0.1..0.3 of the original -> 0.2..0.6 of the left half.
    assert data["bodies"][0]["box"] == pytest.approx([0.2, 0.2, 0.6, 0.4])


def test_accept_records_the_composed_crop(album_dir, tmp_path, client):
    _with_edit(album_dir, tmp_path, crop=(0.0, 0.0, 0.5, 1.0))
    pending = album_dir / "editedimages" / ".pending" / "IMG_0001.jpg"
    pending.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(pending), np.zeros((10, 10, 3), np.uint8))
    culling_app.ai_edit_state.set_pending(
        album_dir, KEY, pending, pending, album_dir / "editedimages" / "IMG_0001.jpg",
        crop=(0.0, 0.0, 0.25, 0.5),
    )
    assert client.post("/api/edit/ai-edit/accept").status_code == 200
    image = album_for(album_dir).image(KEY)
    assert image.edited_crop == (0.0, 0.0, 0.25, 0.5)


def test_revert_deletes_the_edit_and_forgets_it(album_dir, tmp_path, client):
    edited = _with_edit(album_dir, tmp_path, crop=(0.0, 0.0, 0.5, 1.0))
    response = client.post("/api/edit/revert", json={"file": KEY})
    assert response.status_code == 200 and response.json()["deleted"] is True
    assert not edited.exists()
    image = album_for(album_dir).image(KEY)
    assert image.edited_path is None and image.edited_crop is None
    assert image.original_path.is_file()


def test_revert_refused_while_processing(album_dir, tmp_path, client):
    edited = _with_edit(album_dir, tmp_path)
    assert culling_app.processing_state.claim()
    try:
        assert client.post("/api/edit/revert", json={"file": KEY}).status_code == 409
    finally:
        culling_app.processing_state.mark_stopped()
    assert edited.exists()


def test_open_external_creates_the_edited_copy_first(album_dir, tmp_path, client, monkeypatch):
    source = _make_source(tmp_path)
    write_album(album_dir, [make_entry(KEY, file=str(source))])
    album_module.invalidate_all()
    opened = []
    monkeypatch.setattr(culling_app.sys, "platform", "win32")
    monkeypatch.setattr(culling_app, "_open_with_chooser", opened.append)

    data = client.post("/api/edit/open-external", json={"file": KEY}).json()
    edited = album_dir / "editedimages" / "IMG_0001.jpg"
    assert data["created"] is True and opened == [edited]
    assert edited.read_bytes() == source.read_bytes()
    assert album_for(album_dir).image(KEY).edited_path == edited

    # Second time round the existing edit is opened as is.
    data = client.post("/api/edit/open-external", json={"file": KEY}).json()
    assert data["created"] is False and opened == [edited, edited]


# ---- edited vs original image consumers ------------------------------------- #

def test_api_original_serves_the_edit_and_revalidates(album_dir, tmp_path, client):
    edited = _with_edit(album_dir, tmp_path)
    response = client.get(f"/api/original/{KEY}")
    assert response.status_code == 200
    assert response.content == edited.read_bytes()
    assert response.headers["cache-control"] == "no-cache"


def _two_photos_with_one_edit(album_dir: Path, tmp_path: Path):
    edited = _with_edit(album_dir, tmp_path)  # IMG_0001: has an edit
    other = _make_source(tmp_path, "IMG_0002.jpg")
    entries = [
        make_entry(KEY, file=str(tmp_path / "src" / KEY), edited_image="editedimages/IMG_0001.jpg"),
        make_entry("IMG_0002.jpg", file=str(other)),
    ]
    write_album(album_dir, entries)
    album_module.invalidate_all()
    return edited, tmp_path / "src" / KEY, other


def test_burst_frames_are_always_the_originals(album_dir, tmp_path, monkeypatch):
    from algo import burst_video

    edited, original, other = _two_photos_with_one_edit(album_dir, tmp_path)
    monkeypatch.setattr(burst_video, "image_capture_timestamp", lambda path: 100.0)
    frames = [f for b in burst_video.detect_bursts(album_dir) for f in b.frames]
    assert sorted(f.path for f in frames) == sorted([original, other])
    assert edited not in [f.path for f in frames]


def test_player_videos_use_the_originals(album_dir, tmp_path, monkeypatch):
    edited, original, _other = _two_photos_with_one_edit(album_dir, tmp_path)
    fr = album_dir / ".FaceReco" / "1"
    fr.mkdir(parents=True)
    monkeypatch.setattr(culling_app, "_facereco_dir", lambda p: album_dir / ".FaceReco")
    monkeypatch.setattr(culling_app, "_load_face_json", lambda d: {"name": "Sam", "faces": [{"origFilename": KEY}]})
    monkeypatch.setattr(culling_app, "_is_pending_cluster", lambda n: False)
    assert culling_app._collect_player_frame_paths(album_dir) == {"Sam": [original]}


# ---- crop recorded as pixels; cropped-out area shaded in the preview -------- #

def test_crop_to_px_in_the_originals_frame():
    box = (0.25, 0.5, 0.75, 1.0)
    assert culling_app._crop_to_px(box, (400, 200), None) == {"x": 100, "y": 100, "h": 100, "w": 200}
    # Edited file is the left half of a 800x200 original: its 400 px are 0.5 of it.
    nested = culling_app._compose_crop((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0))
    assert culling_app._crop_to_px(nested, (400, 200), (0.0, 0.0, 0.5, 1.0)) == {"x": 200, "y": 0, "h": 200, "w": 200}


def test_darken_outside_crop_blackens_80_percent_outside_only():
    from algo.stages.annotation import darken_outside_crop

    img = np.full((100, 200, 3), 200, np.uint8)
    assert darken_outside_crop(img, None) is img
    assert darken_outside_crop(img, (0.0, 0.0, 1.0, 1.0)) is img
    out = darken_outside_crop(img, (0.25, 0.25, 0.75, 0.75))
    assert (out[50, 100] == 200).all()                    # inside: untouched
    assert (out[10, 10] == 40).all()                      # corner: 20% left
    assert (out[50, 10] == 40).all() and (out[90, 100] == 40).all()
    assert (img == 200).all()                             # input not modified


def _no_body_entry(tmp_path: Path, **extra):
    return make_entry(
        KEY, file=str(_make_source(tmp_path, shape=(100, 200))), **extra,
        annotation_data={"processing_shape": [100, 200], "overall_blurry": False, "evaluated": []},
    )


def test_preview_shades_what_the_accepted_crop_removed(album_dir, tmp_path):
    from algo.stages.annotation import render_album_preview

    edited = album_dir / "editedimages" / "IMG_0001.jpg"
    edited.parent.mkdir(parents=True)
    assert cv2.imwrite(str(edited), np.zeros((50, 100, 3), np.uint8))
    write_album(album_dir, [_no_body_entry(
        tmp_path, edited_image="editedimages/IMG_0001.jpg", edited_crop=[0.25, 0.25, 0.75, 0.75])])
    album_module.invalidate_all()

    image = album_for(album_dir).image(KEY)
    assert render_album_preview(image)
    preview = cv2.imread(str(image.preview_path))
    inside, outside = preview[50, 100].mean(), preview[5, 5].mean()
    assert outside < inside * 0.3

    # No crop recorded -> an unshaded preview.
    write_album(album_dir, [_no_body_entry(tmp_path, edited_image="editedimages/IMG_0001.jpg")])
    album_module.invalidate_all()
    image = album_for(album_dir).image(KEY)
    assert render_album_preview(image)
    preview = cv2.imread(str(image.preview_path))
    assert abs(float(preview[5, 5].mean()) - float(preview[50, 100].mean())) < 10


def test_accept_writes_the_pixel_crop_and_redraws_the_preview(album_dir, tmp_path, client):
    write_album(album_dir, [_no_body_entry(tmp_path)])
    album_module.invalidate_all()
    preview = album_for(album_dir).image(KEY).preview_path
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"stale")
    pending = album_dir / "editedimages" / ".pending" / "IMG_0001.jpg"
    pending.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(pending), np.zeros((50, 100, 3), np.uint8))
    culling_app.ai_edit_state.set_pending(
        album_dir, KEY, pending, pending, album_dir / "editedimages" / "IMG_0001.jpg",
        crop=(0.25, 0.25, 0.75, 0.75), crop_px={"x": 50, "y": 25, "h": 50, "w": 100},
    )
    assert client.post("/api/edit/ai-edit/accept").status_code == 200
    image = album_for(album_dir).image(KEY)
    assert image.edited_crop_px == {"x": 50, "y": 25, "h": 50, "w": 100}
    assert not preview.exists(), "the cached preview must be dropped so it is redrawn with the shading"


def test_revert_removes_the_pixel_crop_and_redraws_the_preview(album_dir, tmp_path, client):
    _with_edit(album_dir, tmp_path, crop=(0.0, 0.0, 0.5, 1.0))
    album = album_for(album_dir)
    album.image(KEY).set_edited_crop((0.0, 0.0, 0.5, 1.0), {"x": 0, "y": 0, "h": 60, "w": 40})
    album.save()
    preview = album.image(KEY).preview_path
    preview.parent.mkdir(parents=True, exist_ok=True)
    preview.write_bytes(b"shaded")

    assert client.post("/api/edit/revert", json={"file": KEY}).status_code == 200
    image = album_for(album_dir).image(KEY)
    assert image.edited_crop is None and image.edited_crop_px is None
    assert "edited_crop" not in image._d and "edited_crop_px" not in image._d
    assert not preview.exists()


# ---- Save Copy --------------------------------------------------------------- #

def _copy_person(box, kps=(), face_box=None):
    person = {
        "body_bbox": {"x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]},
        "body_keypoints": [{"x": x, "y": y, "conf": 0.9, "passed": True} for x, y in kps],
        "face_bbox": None, "narrow_face_bbox": None, "face_kps": None,
        "sharpness_score": 0.5,
    }
    if face_box:
        person["face_bbox"] = {"x1": face_box[0], "y1": face_box[1], "x2": face_box[2], "y2": face_box[3]}
        person["face_kps"] = {
            "bbox": dict(person["face_bbox"]), "confidence": 0.8, "passed": True,
            "landmarks": [{"x": face_box[0], "y": face_box[1], "conf": 0.7, "passed": True}],
        }
    return person


def test_copy_entry_reframes_detections_and_drops_edit_state(album_dir, tmp_path):
    people = [
        _copy_person((0.1, 0.1, 0.3, 0.9)),                                   # cropped out
        _copy_person((0.5, 0.2, 0.9, 0.8), kps=[(0.6, 0.5), (0.2, 0.5)], face_box=(0.6, 0.2, 0.7, 0.3)),
    ]
    entry = make_entry(KEY, file=str(_make_source(tmp_path)), stars=4, keep=True,
                       edited_image="editedimages/IMG_0001.jpg", edited_crop=[0, 0, 1, 1],
                       edited_crop_px={"x": 0, "y": 0, "h": 1, "w": 1},
                       annotation_data={"processing_shape": [1000, 2000], "overall_blurry": False, "evaluated": people})
    write_album(album_dir, [entry])
    album_module.invalidate_all()
    original = album_for(album_dir).image(KEY)

    copy_entry = original.copy_entry(file=album_dir / "SaveCopy" / "IMG_0001_1.jpg", key="IMG_0001_1.jpg",
                                     crop=(0.5, 0.0, 1.0, 1.0))
    copied = album_module.AlbumImage(album_for(album_dir), copy_entry)
    assert copied.key == "IMG_0001_1.jpg"
    assert copied.source_file.endswith("IMG_0001_1.jpg")
    assert copy_entry["preview_path"] == "previews/IMG_0001_1.jpg.jpg"  # importer convention: <key>.jpg
    assert copied.stars == 4 and copied.keep is True
    assert copied.auto_adjustment is None
    assert copied.edited_path is None and copied.edited_crop is None
    assert not {"edited_image", "edited_crop", "edited_crop_px"} & set(copy_entry)
    assert copied.processing_shape == (1000, 1000)
    assert len(copied.bodies) == 1
    kept = copied.bodies[0]
    assert kept.body_bbox.to_wire() == pytest.approx({"x1": 0.0, "y1": 0.2, "x2": 0.8, "y2": 0.8})
    assert kept.face.bbox.to_wire() == pytest.approx({"x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.3})
    first_kp, second_kp = kept.body_keypoints
    assert (first_kp.point.x, first_kp.point.y) == pytest.approx((0.2, 0.5))
    # The keypoint left of the crop is pinned to the edge and disqualified.
    assert second_kp.point.x == 0.0 and second_kp.passed is False
    # The original's own entry is untouched.
    assert len(original.bodies) == 2
    assert original.edited_path is not None or original.entry["edited_image"]


def test_copy_entry_without_people_left_becomes_skipped(album_dir, tmp_path):
    entry = make_entry(KEY, file=str(_make_source(tmp_path)), annotation_data={
        "processing_shape": [100, 100], "overall_blurry": False,
        "evaluated": [_copy_person((0.1, 0.1, 0.3, 0.9))]})
    write_album(album_dir, [entry])
    album_module.invalidate_all()
    copy_entry = album_for(album_dir).image(KEY).copy_entry(
        file=Path("x.jpg"), key="x.jpg", crop=(0.5, 0.0, 1.0, 1.0))
    copied = album_module.AlbumImage(album_for(album_dir), copy_entry)
    assert copied.status == "skipped" and copied.bodies == []


def _album_with_info(album_dir: Path, tmp_path: Path):
    import json

    source = _make_source(tmp_path)
    write_album(album_dir, [make_entry(KEY, file=str(source), annotation_data={
        "processing_shape": [60, 80], "overall_blurry": False,
        "evaluated": [_copy_person((0.55, 0.2, 0.9, 0.8))]})])
    (album_dir / "info.json").write_text(json.dumps({
        "SrcDir": str(source.parent), "Anno_Sharp": [{"src": KEY, "srcPath": str(source)}],
        "Anno_Blur": [], "Anno_Skipped": []}), encoding="utf-8")
    album_module.invalidate_all()
    return source


def _pend(album_dir: Path, crop):
    pending = album_dir / "editedimages" / ".pending" / "IMG_0001.jpg"
    pending.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(pending), np.full((60, 40, 3), 77, np.uint8))
    culling_app.ai_edit_state.set_pending(
        album_dir, KEY, pending, pending, album_dir / "editedimages" / "IMG_0001.jpg", crop=crop)
    return pending


def test_save_copy_adds_a_new_numbered_photo_to_the_album(album_dir, tmp_path, client):
    import json
    import os

    source = _album_with_info(album_dir, tmp_path)
    os.utime(source, (1_700_000_000, 1_700_000_000))
    pending = _pend(album_dir, (0.5, 0.0, 1.0, 1.0))

    first = client.post("/api/edit/ai-edit/save-copy").json()
    saved = album_dir / "SaveCopy" / "IMG_0001_1.jpg"
    assert first["key"] == "IMG_0001_1.jpg" and Path(first["path"]) == saved
    assert saved.is_file() and not pending.exists()
    assert abs(saved.stat().st_mtime - 1_700_000_000) < 2  # lands in the original's burst

    album = album_for(album_dir)
    copied = album.image("IMG_0001_1.jpg")
    assert copied.original_path == saved and copied.edited_path is None
    assert copied.bodies[0].body_bbox.x1 == pytest.approx(0.1)
    # The photo that was edited keeps its own record and file.
    original = album.image(KEY)
    assert original.original_path == source and original.edited_path is None
    assert original.bodies[0].body_bbox.x1 == pytest.approx(0.55)

    info = json.loads((album_dir / "info.json").read_text(encoding="utf-8"))
    assert [e["src"] for e in info["Anno_Sharp"]] == [KEY, "IMG_0001_1.jpg"]
    assert info["Anno_Sharp"][1]["srcPath"] == str(saved)

    # A second copy takes the next number, and so does one after a gap.
    _pend(album_dir, None)
    assert client.post("/api/edit/ai-edit/save-copy").json()["key"] == "IMG_0001_2.jpg"
    (album_dir / "SaveCopy" / "IMG_0001_1.jpg").unlink()
    _pend(album_dir, None)
    assert client.post("/api/edit/ai-edit/save-copy").json()["key"] == "IMG_0001_3.jpg"


def test_save_copy_needs_a_pending_edit(album_dir, tmp_path, client):
    _album_with_info(album_dir, tmp_path)
    assert client.post("/api/edit/ai-edit/save-copy").status_code == 404
