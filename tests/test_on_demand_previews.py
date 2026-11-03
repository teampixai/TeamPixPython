"""Annotated previews are rendered on demand and cached; face-recognition
names (or "teammate") are drawn on them; confirming faces drops the cache."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import culling_app
from algo import album as album_module
from algo.album import Album, PersonRecord
from algo.config import app_config
from algo.frame import Frame
from algo.stages import annotation
from algo.stages.annotation import TEAMMATE_LABEL, _annotate_frame, _identity_label, invalidate_previews
from conftest import make_body_dict, make_entry, write_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient

KEY = "IMG_0001.jpg"
OTHER = "IMG_0002.jpg"


def _make_source(root: Path, name: str) -> Path:
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = (60, 120, 180)
    fp = root / "src" / name
    fp.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(fp), img)
    return fp


def _preview(album: Path, key: str = KEY) -> Path:
    """Where conftest's (pre-key-naming) entries put the preview."""
    return album / "previews" / f"{Path(key).stem}.jpg"


def _thumb(album: Path, key: str = KEY) -> Path:
    """`AlbumImage.thumbnail_path`: named by the full key."""
    return album / "thumbnails" / f"{key}.jpg"


def _previews(album: Path, key: str = KEY) -> list[Path]:
    return [_preview(album, key), _thumb(album, key)]


@pytest.fixture
def served(album_dir, tmp_path, monkeypatch):
    """An album with one real source photo and no previews on disk, served
    through the app's real routes."""
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path, KEY)))])
    album_module.invalidate_all()
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    return TestClient(culling_app.app), album_dir


@pytest.fixture
def two_photos_no_cache(album_dir, tmp_path, monkeypatch):
    """Two real source photos, nothing cached on disk."""
    write_album(album_dir, [
        make_entry(KEY, file=str(_make_source(tmp_path, KEY))),
        make_entry(OTHER, file=str(_make_source(tmp_path, OTHER))),
    ])
    album_module.invalidate_all()
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    return TestClient(culling_app.app), album_dir


# ---- which label a person gets --------------------------------------------- #

def _body(**overrides):
    return PersonRecord(make_body_dict(**overrides)).to_body()


def test_label_is_the_name_when_there_is_one():
    body = _body(is_teammate=True)
    body.player_name = "Jane Doe"
    assert _identity_label(body) == "Jane Doe"


def test_label_falls_back_to_teammate_when_unnamed():
    assert _identity_label(_body(is_teammate=True)) == TEAMMATE_LABEL


def test_no_label_for_an_unnamed_non_teammate():
    assert _identity_label(_body()) == ""
    body = _body()
    body.player_name = "   "
    assert _identity_label(body) == ""


def test_player_name_survives_the_record_to_body_bridge():
    assert PersonRecord(make_body_dict(player_name="Ana")).to_body().player_name == "Ana"


# ---- the label is actually drawn ------------------------------------------- #

def _render(tmp_path: Path, tag: str, *, name: str = "", teammate: bool = False) -> np.ndarray:
    body = _body(is_teammate=teammate)
    body.player_name = name
    frame = Frame(path=tmp_path / f"{tag}.jpg", bodies=[body],
                  image=np.full((600, 800, 3), 90, np.uint8), output_key=f"{tag}.jpg")
    out = tmp_path / "out" / "previews" / f"{tag}.jpg"
    _annotate_frame(frame, out, app_config)
    return cv2.imread(str(out))


def test_name_and_teammate_labels_are_drawn(tmp_path):
    plain = _render(tmp_path, "plain")
    named = _render(tmp_path, "named", name="Jane Doe")
    teammate = _render(tmp_path, "teammate", teammate=True)
    assert not np.array_equal(plain, named), "the name was not drawn"
    assert not np.array_equal(plain, teammate), "'teammate' was not drawn"
    assert not np.array_equal(named, teammate)


def test_a_name_wins_over_teammate(tmp_path):
    named = _render(tmp_path, "named", name="Jane Doe")
    both = _render(tmp_path, "both", name="Jane Doe", teammate=True)
    assert np.array_equal(named, both)


def test_non_ascii_names_render(tmp_path):
    plain = _render(tmp_path, "plain")
    accented = _render(tmp_path, "accented", name="José Núñez")
    assert not np.array_equal(plain, accented)


def test_rendering_a_preview_does_not_write_a_thumbnail(tmp_path):
    _render(tmp_path, "x", name="Jane")
    assert (tmp_path / "out" / "previews" / "x.jpg").is_file()
    assert not (tmp_path / "out" / "thumbnails").exists()


# ---- rendered on demand, then cached --------------------------------------- #

def test_preview_is_rendered_on_first_request_and_cached(served, monkeypatch):
    client, album = served
    assert not any(p.exists() for p in _previews(album))

    renders = []
    real = culling_app.render_album_preview
    monkeypatch.setattr(culling_app, "render_album_preview",
                        lambda *a, **k: renders.append(1) or real(*a, **k))

    first = client.get(f"/api/anno_img/{KEY}")
    assert first.status_code == 200 and first.content[:2] == b"\xff\xd8"
    assert _preview(album).is_file()
    assert len(renders) == 1

    assert client.get(f"/api/anno_img/{KEY}").status_code == 200
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 200
    assert len(renders) == 1, "a cached preview must not be rendered again"


def test_missing_thumbnail_is_generated_from_the_source_without_a_preview(served, monkeypatch):
    client, album = served

    def no_render(*_a, **_k):
        raise AssertionError("a missing thumbnail must be made from the source, not by rendering a preview")

    monkeypatch.setattr(culling_app, "render_album_preview", no_render)
    res = client.get(f"/api/culling/thumb/{KEY}")
    assert res.status_code == 200
    assert cv2.imread(str(_thumb(album))).shape[:2] == (128, 128)
    assert not _preview(album).exists()


def test_a_deleted_thumbnail_is_generated_again(served):
    client, album = served
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 200
    _thumb(album).unlink()
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 200
    assert _thumb(album).is_file()


def test_import_thumbnail_is_served_without_rendering_a_preview(served, monkeypatch):
    client, album = served
    _thumb(album).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(_thumb(album)), np.full((128, 128, 3), 200, np.uint8))

    def no_render(*_a, **_k):
        raise AssertionError("the nav strip must not render when the thumbnail exists")

    monkeypatch.setattr(culling_app, "render_album_preview", no_render)
    monkeypatch.setattr("algo.stages.image_analysis.render_album_thumbnail", no_render)
    res = client.get(f"/api/culling/thumb/{KEY}")
    assert res.status_code == 200 and res.content == _thumb(album).read_bytes()
    assert not _preview(album).exists()


def test_older_albums_thumbnail_under_the_preview_name_is_reused(served, monkeypatch):
    client, album = served
    legacy = album / "thumbnails" / f"{Path(KEY).stem}.jpg"   # the old <preview stem>.jpg name
    legacy.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(legacy), np.full((128, 128, 3), 77, np.uint8))

    def no_generate(*_a, **_k):
        raise AssertionError("an existing older-album thumbnail must not be regenerated")

    monkeypatch.setattr("algo.stages.image_analysis.render_album_thumbnail", no_generate)
    res = client.get(f"/api/culling/thumb/{KEY}")
    assert res.status_code == 200 and res.content == legacy.read_bytes()
    assert not _thumb(album).exists()


def test_cached_preview_is_the_last_resort_when_the_source_is_gone(served):
    client, album = served
    assert client.get(f"/api/anno_img/{KEY}").status_code == 200   # caches the preview
    Path(album_module.album_for(album).image(KEY).source_file).unlink()
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 200
    assert _thumb(album).is_file(), "derived from the cached preview"


def test_no_thumbnail_and_nothing_to_make_one_from_is_a_404(served):
    client, album = served
    Path(album_module.album_for(album).image(KEY).source_file).unlink()
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 404
    assert client.get("/api/culling/thumb/not-in-album.jpg").status_code == 404


def test_one_unreadable_source_does_not_break_other_thumbnails(two_photos_no_cache):
    client, album = two_photos_no_cache
    Path(album_module.album_for(album).image(KEY).source_file).write_bytes(b"not an image")
    assert client.get(f"/api/culling/thumb/{KEY}").status_code == 404
    assert client.get(f"/api/culling/thumb/{OTHER}").status_code == 200


def test_existing_thumbnail_is_not_replaced_by_a_newer_preview(served):
    _client, album = served
    for fp, value in ((_thumb(album), 10), (_preview(album), 250)):
        fp.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(fp), np.full((300, 300, 3), value, np.uint8))
    before = _thumb(album).read_bytes()
    image = album_module.album_for(album).image(KEY)
    assert image.ensure_thumbnail() == _thumb(album)
    assert _thumb(album).read_bytes() == before


def test_a_deleted_preview_is_rendered_again(served, monkeypatch):
    client, album = served
    renders = []
    real = culling_app.render_album_preview
    monkeypatch.setattr(culling_app, "render_album_preview",
                        lambda *a, **k: renders.append(1) or real(*a, **k))
    client.get(f"/api/anno_img/{KEY}")
    _previews(album)[0].unlink()
    assert client.get(f"/api/anno_img/{KEY}").status_code == 200
    assert len(renders) == 2


def test_missing_source_photo_is_a_404_not_a_crash(served):
    client, album = served
    Path(album_module.album_for(album).image(KEY).source_file).unlink()
    assert client.get(f"/api/anno_img/{KEY}").status_code == 404
    assert client.get("/api/anno_img/not-in-album.jpg").status_code == 404


# ---- names come from the face-recognition result --------------------------- #

def _write_cluster(album: Path, folder: str, name: str, key: str, bbox: dict, crop: str = "a.png") -> Path:
    cluster = album / ".FaceReco" / folder
    (cluster / "Face").mkdir(parents=True, exist_ok=True)
    (cluster / "Face" / crop).write_bytes(b"png")
    (cluster / "face.json").write_text(json.dumps({
        "name": name, "playernum": None, "cluster": folder, "provider": None, "aligned": None,
        "faces": [{"cropFileName": crop, "origFilename": key, "Body": {"body_bbox": bbox}}],
    }), encoding="utf-8")
    return cluster


@pytest.fixture
def name_capture(monkeypatch):
    """Replaces the drawing with a stub that records the name each person
    would be drawn with."""
    seen: list[list[str]] = []

    def fake(image, config=None, player_name_for=None):
        seen.append([player_name_for(r) if player_name_for else "" for r in image.bodies])
        image.preview_path.parent.mkdir(parents=True, exist_ok=True)
        image.preview_path.write_bytes(b"\xff\xd8preview")
        return True

    monkeypatch.setattr(culling_app, "render_album_preview", fake)
    return seen


def test_a_named_cluster_names_the_matching_person(served, name_capture):
    client, album = served
    _write_cluster(album, "Jane Doe", "Jane Doe", KEY, make_body_dict()["body_bbox"])
    assert client.get(f"/api/anno_img/{KEY}").status_code == 200
    assert name_capture == [["Jane Doe"]]


def test_unnamed_clusters_and_other_boxes_give_no_name(served, name_capture):
    client, album = served
    _write_cluster(album, "0001", "", KEY, make_body_dict()["body_bbox"])              # pending cluster
    _write_cluster(album, "Other", "Other", KEY, {"x1": 0.6, "y1": 0.6, "x2": 0.9, "y2": 0.9})  # someone else
    client.get(f"/api/anno_img/{KEY}")
    assert name_capture == [[""]]


def test_name_index_follows_changes_to_the_clusters(served, name_capture):
    client, album = served
    _write_cluster(album, "Jane Doe", "Jane Doe", KEY, make_body_dict()["body_bbox"])
    client.get(f"/api/anno_img/{KEY}")
    _previews(album)[0].unlink()
    (album / ".FaceReco" / "Jane Doe" / "face.json").write_text(
        json.dumps({"name": "Janet Doe", "faces": [{"cropFileName": "a.png", "origFilename": KEY,
                                                     "Body": {"body_bbox": make_body_dict()["body_bbox"]}}]}),
        encoding="utf-8")
    client.get(f"/api/anno_img/{KEY}")
    assert name_capture == [["Jane Doe"], ["Janet Doe"]]


def test_render_falls_back_to_the_name_stored_on_the_person(tmp_path, monkeypatch):
    album = tmp_path / "album"
    entry = make_entry(KEY, file=str(_make_source(tmp_path, KEY)), annotation_data={
        "processing_shape": [1080, 1920], "overall_blurry": False,
        "evaluated": [make_body_dict(player_name="Stored Name")],
    })
    write_album(album, [entry])
    album_module.invalidate_all()
    seen = []
    monkeypatch.setattr(annotation, "_annotate_frame", lambda frame, out, cfg: seen.append(frame.bodies[0].player_name))
    assert annotation.render_album_preview(album_module.album_for(album).image(KEY))
    assert seen == ["Stored Name"]


# ---- confirming faces refreshes the preview -------------------------------- #

@pytest.fixture
def two_photos(album_dir, tmp_path, monkeypatch):
    write_album(album_dir, [
        make_entry(KEY, file=str(_make_source(tmp_path, KEY))),
        make_entry(OTHER, file=str(_make_source(tmp_path, OTHER))),
    ])
    album_module.invalidate_all()
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    for key in (KEY, OTHER):
        for fp in _previews(album_dir, key):
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(b"cached")
    return TestClient(culling_app.app), album_dir


def test_assigning_a_name_drops_only_that_photos_cached_preview(two_photos):
    client, album = two_photos
    _write_cluster(album, "0001", "", KEY, make_body_dict()["body_bbox"])
    res = client.post("/api/faces/commit", json={"operations": [
        {"type": "assign", "sourceCluster": "0001", "crop": "a.png", "name": "Jane Doe"}]})
    assert res.status_code == 200
    assert not _preview(album, KEY).exists(), "assigned photo's preview must be dropped"
    assert _thumb(album, KEY).is_file(), "the plain-photo thumbnail can't go stale, so it is kept"
    assert all(p.is_file() for p in _previews(album, OTHER)), "other photos keep their cache"
    person = album_module.album_for(album).image(KEY).bodies[0]
    assert person.player_name == "Jane Doe"


def test_deleting_a_face_drops_the_preview_and_clears_the_name(two_photos):
    client, album = two_photos
    _write_cluster(album, "Jane Doe", "Jane Doe", KEY, make_body_dict()["body_bbox"])
    image = album_module.album_for(album).image(KEY)
    image.bodies[0].player_name = "Jane Doe"
    album_module.album_for(album).save()

    res = client.post("/api/faces/commit", json={"operations": [
        {"type": "delete", "sourceCluster": "Jane Doe", "crop": "a.png"}]})
    assert res.status_code == 200
    assert not _preview(album, KEY).exists() and _thumb(album, KEY).is_file()
    assert all(p.is_file() for p in _previews(album, OTHER))
    assert album_module.album_for(album).image(KEY).bodies[0].player_name == ""


def test_a_commit_that_changes_nothing_keeps_the_cache(two_photos):
    client, album = two_photos
    _write_cluster(album, "0001", "", KEY, make_body_dict()["body_bbox"])
    res = client.post("/api/faces/commit", json={"operations": [
        {"type": "assign", "sourceCluster": "0001", "crop": "a.png", "name": "  "}]})
    assert res.status_code == 200
    assert all(p.is_file() for key in (KEY, OTHER) for p in _previews(album, key))


# ---- cache invalidation helper / merge ------------------------------------- #

def test_invalidate_previews_by_stem_and_all_leaves_thumbnails(tmp_path):
    for stem in ("a", "b"):
        for sub in ("previews", "thumbnails"):
            (tmp_path / sub).mkdir(exist_ok=True)
            (tmp_path / sub / f"{stem}.jpg").write_bytes(b"x")
    assert invalidate_previews(tmp_path, ["a", "missing"]) == 1
    assert not (tmp_path / "previews" / "a.jpg").exists() and (tmp_path / "previews" / "b.jpg").exists()
    assert invalidate_previews(tmp_path) == 1
    assert sorted(p.name for p in (tmp_path / "thumbnails").iterdir()) == ["a.jpg", "b.jpg"]
    assert invalidate_previews(tmp_path / "nonexistent") == 0


def test_merging_renames_preview_path_and_copies_the_thumbnail_on_key_collision(tmp_path):
    other, target = Album(tmp_path / "other"), Album(tmp_path / "target")
    (other.path / "thumbnails").mkdir(parents=True)
    (other.path / "thumbnails" / "IMG_1.jpg.jpg").write_bytes(b"thumb")

    # The preview was never rendered, but the thumbnail (written at import) is
    # copied anyway, and both take the renamed key.
    entry = {"key": "IMG_1.jpg", "preview_path": "previews/IMG_1.jpg.jpg"}
    target._copy_entry_files(other, entry, "IMG_1.jpg", "IMG_1__2.jpg")
    assert entry["preview_path"] == "previews/IMG_1__2.jpg.jpg"
    assert (target.path / "thumbnails" / "IMG_1__2.jpg.jpg").read_bytes() == b"thumb"

    unchanged = {"key": "IMG_3.jpg", "preview_path": "previews/IMG_3.jpg.jpg"}
    target._copy_entry_files(other, unchanged, "IMG_3.jpg", "IMG_3.jpg")
    assert unchanged["preview_path"] == "previews/IMG_3.jpg.jpg"


# ---- thumbnails are written during import analysis ------------------------- #

def test_raw_and_jpeg_of_the_same_shot_get_separate_preview_and_thumbnail(album_dir):
    album = Album.create(album_dir, team_id="t", paths=[Path("C:/shots/IMG_0001.JPG"), Path("C:/shots/IMG_0001.CR3")])
    jpg, raw = album.image("IMG_0001.JPG"), album.image("IMG_0001.CR3")
    assert jpg.thumbnail_path != raw.thumbnail_path

    from algo.results import build_result_entries
    entries = build_result_entries([Frame(path=Path(p), output_key=k) for p, k in
                                    (("C:/shots/IMG_0001.JPG", jpg.key), ("C:/shots/IMG_0001.CR3", raw.key))])
    assert entries[0]["preview_path"] != entries[1]["preview_path"]


def test_analysis_writes_each_thumbnail_to_its_album_thumbnail_path(tmp_path, album_dir, monkeypatch):
    from algo.stages import image_analysis

    # Two source folders with the same filename -> keys IMG_0001.jpg and IMG_0001__2.jpg.
    first = _make_source(tmp_path / "a", KEY)
    second = _make_source(tmp_path / "b", KEY)
    album = Album.create(album_dir, team_id="t", paths=[first, second])
    keys = [album.images_by_source()[p.resolve()].key for p in (first, second)]
    assert keys == ["IMG_0001.jpg", "IMG_0001__2.jpg"]

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", lambda *a, **k: ([], False))
    stage = image_analysis.MediaPipeImageAnalysisStage(
        None, None, None, None, only_paths=[first, second], album=album,
    )
    frames = stage.process([], app_config)

    assert [f.output_key for f in frames] == keys
    for key in keys:
        thumb = album.image(key).thumbnail_path
        assert thumb == album_dir / "thumbnails" / f"{key}.jpg"
        assert cv2.imread(str(thumb)).shape[:2] == (128, 128)
    assert not (album_dir / "previews").exists(), "analysis must not render previews"


def test_analysis_without_an_album_writes_no_thumbnails(tmp_path, monkeypatch):
    from algo.stages import image_analysis

    src = _make_source(tmp_path, KEY)
    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", lambda *a, **k: ([], False))
    frames = image_analysis.MediaPipeImageAnalysisStage(None, None, None, None, only_paths=[src]).process([], app_config)
    assert frames[0].output_key == ""
    assert not list(tmp_path.rglob("thumbnails"))
