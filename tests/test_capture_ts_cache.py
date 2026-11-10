"""The photo's capture time is cached in album.json (`capture_ts`) so the
Culling page never re-reads EXIF from every source photo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import culling_app
from algo import album as album_module
from algo.album import Album
from algo.frame import Frame
from algo.results import build_result_entries
from conftest import make_entry, write_album


def _album(album_dir: Path, tmp_path: Path, **entry_extra) -> Path:
    keys = ["A.jpg", "B.jpg"]
    src = tmp_path / "src"
    src.mkdir(parents=True, exist_ok=True)
    entries = []
    for key in keys:
        (src / key).write_bytes(b"not really a jpeg")
        entries.append(make_entry(key, file=str(src / key), **entry_extra))
    album_json = write_album(album_dir, entries)
    (album_dir / "info.json").write_text(json.dumps({
        "SrcDir": str(src), "Timestamp": "20260101-000000",
        "Anno_Sharp": [{"src": k, "srcPath": str(src / k)} for k in keys],
        "Anno_Blur": [], "Anno_Skipped": [],
    }), encoding="utf-8")
    album_module.invalidate_all()
    return album_json


@pytest.fixture
def exif_reads(monkeypatch):
    calls: list[Path] = []

    def fake(path):
        calls.append(Path(path))
        return 1000.5 + len(calls)

    monkeypatch.setattr(culling_app, "image_capture_timestamp", fake)
    return calls


def test_import_records_the_capture_time(tmp_path, monkeypatch):
    import algo.results as results

    monkeypatch.setattr(results, "image_capture_timestamp", lambda p: 1234.5)
    entry = build_result_entries([Frame(path=tmp_path / "IMG.jpg", output_key="IMG.jpg")])[0]
    assert entry["capture_ts"] == 1234.5


def test_capture_ts_roundtrips_and_defaults_to_none(album_dir):
    write_album(album_dir, [make_entry("A.jpg", capture_ts=42.5), make_entry("B.jpg")])
    album = Album(album_dir)
    assert album.image("A.jpg").capture_ts == 42.5
    assert album.image("B.jpg").capture_ts is None


def test_cached_timestamps_are_used_without_reading_sources(album_dir, tmp_path, exif_reads):
    _album(album_dir, tmp_path, capture_ts=500.25)
    images = culling_app._all_culling_images(album_dir)
    assert [im["timestamp"] for im in images] == [500.25, 500.25]
    assert exif_reads == []


def test_whole_second_cached_timestamps_are_invalid_and_refreshed(album_dir, tmp_path, exif_reads):
    album_json = _album(album_dir, tmp_path, capture_ts=500.0)
    assert Album(album_dir).image("A.jpg").capture_ts is None

    images = culling_app._all_culling_images(album_dir)

    assert len(exif_reads) == 2
    assert all(im["timestamp"] > 1000 for im in images)
    saved = json.loads(album_json.read_text(encoding="utf-8"))["results"]
    assert [e["capture_ts"] for e in saved] == [im["timestamp"] for im in images]


def _jpeg_with_exif(path: Path, subsec: str | None, taken: str = "2025:08:02 12:33:58") -> Path:
    from PIL import Image

    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif[306] = taken
    sub = exif.get_ifd(0x8769)
    sub[36867] = taken
    if subsec is not None:
        sub[37521] = subsec
    img.save(path, exif=exif)
    return path


def test_capture_timestamp_adds_the_exif_subsecond(tmp_path):
    from algo.utils import has_millisecond_part, image_capture_timestamp

    whole = image_capture_timestamp(_jpeg_with_exif(tmp_path / "a.jpg", None))
    sub = image_capture_timestamp(_jpeg_with_exif(tmp_path / "b.jpg", "16"))
    assert has_millisecond_part(whole) and has_millisecond_part(sub)
    assert sub - whole == pytest.approx(0.16 - 0.0005, abs=1e-3)


def test_capture_timestamp_falls_back_to_the_file_name_hundredths(tmp_path):
    from algo.utils import image_capture_timestamp

    plain = image_capture_timestamp(_jpeg_with_exif(tmp_path / "IMG.jpg", None))
    named = image_capture_timestamp(_jpeg_with_exif(tmp_path / "20250802-123358-073.jpg", None))
    assert named - plain == pytest.approx(0.73 - 0.0005, abs=1e-3)


def test_capture_timestamp_without_any_subsecond_still_has_a_millisecond_part(tmp_path):
    from algo.utils import has_millisecond_part, image_capture_timestamp

    exact = image_capture_timestamp(_jpeg_with_exif(tmp_path / "IMG.jpg", "00"))
    assert has_millisecond_part(exact)
    assert exact == pytest.approx(int(exact) + 0.0005, abs=1e-6)
    assert has_millisecond_part(image_capture_timestamp(tmp_path / "not-an-image.txt"))


def test_missing_timestamps_are_read_once_and_persisted(album_dir, tmp_path, exif_reads):
    album_json = _album(album_dir, tmp_path)

    first = culling_app._all_culling_images(album_dir)
    assert len(exif_reads) == 2
    saved = json.loads(album_json.read_text(encoding="utf-8"))["results"]
    assert [e["capture_ts"] for e in saved] == [im["timestamp"] for im in first]

    # A fresh process (cold cache) must not read EXIF again.
    album_module.invalidate_all()
    again = culling_app._all_culling_images(album_dir)
    assert len(exif_reads) == 2
    assert [im["timestamp"] for im in again] == [im["timestamp"] for im in first]


def test_summary_never_reads_timestamps_or_saves(album_dir, tmp_path, exif_reads, monkeypatch):
    album_json = _album(album_dir, tmp_path)
    before = album_json.read_bytes()
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)

    summary = culling_app.api_culling_summary()

    assert summary["starCounts"]["3"] == 2
    assert exif_reads == []
    assert album_json.read_bytes() == before


def test_export_selection_never_reads_timestamps(album_dir, tmp_path, exif_reads):
    _album(album_dir, tmp_path)
    assert culling_app._kept_image_basenames(album_dir) == {"A.jpg", "B.jpg"}
    assert exif_reads == []
