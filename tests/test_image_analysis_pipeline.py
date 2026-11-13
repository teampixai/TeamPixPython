"""MediaPipeImageAnalysisStage analyses photos on a bounded pool of worker
threads: results keep file order, each worker gets its own landmarkers, RAM
is bounded by the in-flight cap, and a worker failure surfaces (no hang)."""

from __future__ import annotations

import dataclasses
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from algo import mediapipe_provider
from algo.config import app_config
from algo.image_cache import ImageCache
from algo.models import Body, Box, Face, Point, PredictedKeyPoint
from algo.stages import image_analysis
from algo.stages.image_analysis import MediaPipeImageAnalysisStage


def _sources(root: Path, n: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        img = np.full((120, 160, 3), (i * 7) % 255, np.uint8)
        fp = root / f"IMG_{i:04d}.jpg"
        assert cv2.imwrite(str(fp), img)
        paths.append(fp)
    return paths


def _config(workers: int):
    return dataclasses.replace(app_config, analysis_workers=workers)


def _body() -> Body:
    face = Face(bbox=Box(0.4, 0.1, 0.6, 0.3), confidence=1.0,
                landmarks=[PredictedKeyPoint(Point(0.5, 0.2), 1.0)] * 5)
    return Body(crop=np.empty((0, 0, 3), np.uint8), bbox=Box(0.3, 0.05, 0.7, 0.95),
                faces=[face], keypoints=[])


def test_results_keep_file_order_and_skip_unreadable_files(tmp_path, monkeypatch):
    paths = _sources(tmp_path, 12)
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not a jpeg")
    paths.insert(5, bad)

    def fake_detect(image, *_):
        time.sleep(random.uniform(0, 0.02))
        return [_body()], True

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", fake_detect)
    frames = MediaPipeImageAnalysisStage(None, None, None, None, only_paths=paths).process([], _config(4))

    assert [f.path for f in frames] == [p for p in paths if p != bad]
    assert all(f.image is None and f.normalized_image is None for f in frames)


def test_in_flight_photos_never_exceed_worker_count(tmp_path, monkeypatch):
    paths = _sources(tmp_path, 16)
    lock = threading.Lock()
    active = peak = 0

    def fake_detect(image, *_):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return [], False

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", fake_detect)
    frames = MediaPipeImageAnalysisStage(None, None, None, None, only_paths=paths).process([], _config(3))

    assert len(frames) == 16
    assert 1 < peak <= 3


def test_single_worker_runs_one_photo_at_a_time(tmp_path, monkeypatch):
    paths = _sources(tmp_path, 4)
    threads = set()

    def fake_detect(image, *_):
        threads.add(threading.get_ident())
        return [], False

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", fake_detect)
    frames = MediaPipeImageAnalysisStage(None, None, None, None, only_paths=paths).process([], _config(1))
    assert [f.path for f in frames] == paths
    assert len(threads) == 1


def test_worker_failure_propagates_without_hanging(tmp_path, monkeypatch):
    paths = _sources(tmp_path, 20)
    calls = 0
    lock = threading.Lock()

    def fake_detect(image, *_):
        nonlocal calls
        with lock:
            calls += 1
            nth = calls
        if nth == 7:
            raise RuntimeError("detector blew up")
        time.sleep(0.01)
        return [], False

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", fake_detect)
    stage = MediaPipeImageAnalysisStage(None, None, None, None, only_paths=paths)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="detector blew up"):
        stage.process([], _config(4))
    assert time.monotonic() - started < 10


class _FakeLandmarker:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


def test_each_worker_gets_its_own_landmarkers(tmp_path, monkeypatch):
    paths = _sources(tmp_path, 24)
    created: list[_FakeLandmarker] = []

    def factory(kind):
        def make(**_):
            lm = _FakeLandmarker(f"{kind}{len(created)}")
            created.append(lm)
            return lm
        return make

    monkeypatch.setattr(mediapipe_provider, "create_pose_landmarker", factory("pose"))
    monkeypatch.setattr(mediapipe_provider, "create_face_landmarker", factory("face"))

    used: dict[int, set[tuple[str, str]]] = {}
    lock = threading.Lock()

    def fake_detect(image, detector, pose, face):
        time.sleep(0.01)
        with lock:
            used.setdefault(threading.get_ident(), set()).add((pose.name, face.name))
        return [], False

    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp", fake_detect)
    shared_pose, shared_face = _FakeLandmarker("shared-pose"), _FakeLandmarker("shared-face")
    MediaPipeImageAnalysisStage(None, object(), shared_pose, shared_face,
                                only_paths=paths).process([], _config(4))

    pairs = [next(iter(p)) for p in used.values()]
    assert all(len(p) == 1 for p in used.values()), "a thread switched landmarkers"
    assert len(set(pairs)) == len(pairs), "two threads shared a landmarker pair"
    assert ("shared-pose", "shared-face") in pairs
    assert len(created) == 2 * (len(pairs) - 1)
    assert all(lm.closed for lm in created)
    assert not shared_pose.closed and not shared_face.closed


def test_parallel_workers_store_distinct_cache_crops(tmp_path, monkeypatch):
    paths = _sources(tmp_path / "src", 20)
    monkeypatch.setattr(image_analysis, "detect_qualified_persons_mp",
                        lambda *a, **k: ([_body(), _body()], True))
    cache = ImageCache(tmp_path / "album")
    frames = MediaPipeImageAnalysisStage(None, None, None, None, only_paths=paths, cache=cache,
                                         cache_face_originals=True).process([], _config(4))

    stored = [p for f in frames for b in f.bodies
              for p in (b.cache_normalized_path, *(x for face in b.faces
                        for x in (face.cache_normalized_path, face.cache_original_path)))]
    assert all(stored)
    assert len(set(stored)) == len(stored) == 20 * 2 * 3
    assert all(Path(p).is_file() for p in stored)
