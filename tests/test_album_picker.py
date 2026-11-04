"""Select Album page: each card previews the album's top-ranked photos, served
as cached thumbnails without loading album.json or rendering anything."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import culling_app
from algo import album as album_module
from conftest import make_entry, write_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient


def _source(root: Path, name: str) -> Path:
    fp = root / "src" / name
    fp.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(fp), np.full((60, 80, 3), 120, np.uint8))
    return fp


def _album(album_dir: Path, tmp_path: Path, ratings: dict[str, tuple]) -> None:
    """*ratings*: key -> (stars or None, llm_grade, sharpness_score)."""
    entries = []
    for key, (stars, grade, sharp) in ratings.items():
        extra = {"stars": stars} if stars is not None else {}
        extra.update({"llm_grade": grade} if grade is not None else {})
        entries.append(make_entry(key, file=str(_source(tmp_path, key)), sharpness_score=sharp, **extra))
    write_album(album_dir, entries)
    (album_dir / "info.json").write_text(json.dumps({
        "SrcDir": str(tmp_path / "src"), "Timestamp": "20260101-000000",
        "Anno_Sharp": [{"src": k, "srcPath": str(tmp_path / "src" / k)} for k in ratings],
        "Anno_Blur": [], "Anno_Skipped": [],
    }), encoding="utf-8")
    album_module.invalidate_all()


def test_preview_images_are_the_top_ten_ranked_photos(album_dir, tmp_path):
    ratings = {f"IMG_{i:04d}.jpg": (3, None, 0.5) for i in range(1, 13)}
    ratings["IMG_0003.jpg"] = (5, None, 0.5)             # best: most stars
    ratings["IMG_0007.jpg"] = (4, 0.9, 0.5)              # 4 stars, better LLM grade...
    ratings["IMG_0008.jpg"] = (4, 0.2, 0.9)              # ...than this one
    ratings["IMG_0001.jpg"] = (None, None, 0.99)         # unrated sharp = the default 3 stars
    _album(album_dir, tmp_path, ratings)

    previews = culling_app._build_album_summary(album_dir)["previewImages"]
    assert len(previews) == 10
    assert previews[:3] == ["IMG_0003.jpg", "IMG_0007.jpg", "IMG_0008.jpg"]
    assert previews[3] == "IMG_0001.jpg"                  # 3 stars, highest sharpness
    assert culling_app._build_album_summary(album_dir)["previewImages"] == previews  # deterministic


def test_a_small_album_previews_everything_it_has(album_dir, tmp_path):
    _album(album_dir, tmp_path, {"A.jpg": (3, None, 0.5), "B.jpg": (4, None, 0.5)})
    assert culling_app._build_album_summary(album_dir)["previewImages"] == ["B.jpg", "A.jpg"]


@pytest.fixture
def picker(album_dir, tmp_path, monkeypatch):
    _album(album_dir, tmp_path, {"A.jpg": (3, None, 0.5)})
    monkeypatch.setattr(culling_app, "OUTPUT_DIR", album_dir.parent)
    return TestClient(culling_app.app), album_dir


def test_thumb_is_served_from_disk_without_loading_the_album(picker, monkeypatch):
    client, album = picker
    thumb = album / "thumbnails" / "A.jpg.jpg"
    thumb.parent.mkdir(parents=True)
    assert cv2.imwrite(str(thumb), np.full((128, 128, 3), 9, np.uint8))

    def boom(*_a, **_k):
        raise AssertionError("the album picker must not parse album.json or render a preview per card")

    monkeypatch.setattr(culling_app, "album_for", boom)
    monkeypatch.setattr(culling_app, "render_album_preview", boom)
    response = client.get("/api/albums/thumb", params={"id": album.name, "file": "A.jpg"})
    assert response.status_code == 200
    assert response.content == thumb.read_bytes()
    assert "max-age" in response.headers["cache-control"]


def test_a_missing_thumbnail_is_made_once_from_the_source_not_a_preview(picker, monkeypatch):
    client, album = picker
    monkeypatch.setattr(culling_app, "render_album_preview",
                        lambda *a, **k: pytest.fail("a thumbnail must not need the annotated preview"))
    response = client.get("/api/albums/thumb", params={"id": album.name, "file": "A.jpg"})
    assert response.status_code == 200
    assert cv2.imread(str(album / "thumbnails" / "A.jpg.jpg")).shape[:2] == (128, 128)


def test_thumb_rejects_path_traversal(picker):
    client, album = picker
    assert client.get("/api/albums/thumb", params={"id": album.name, "file": "../album.json"}).status_code == 400
