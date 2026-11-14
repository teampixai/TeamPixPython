"""FaceRecoPipeline._predict_samples prepares faces on worker threads and
embeds them in batches on the calling thread: results keep input order,
each photo is decoded once (and decodes never overlap), each worker owns its
own landmarker, and failures surface instead of hanging."""

from __future__ import annotations

import random
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from algo import facereco, mediapipe_provider
from algo.face_crop_embed import PreparedFaceCrop, make_crop_body
from algo.facereco import FaceRecoConfig, FaceRecoPipeline, _PngWriter, _QualifiedBody
from algo.facereco_provider import BodyRecord, Box, FaceRecoProvider, Player


class _Landmarker:
    def __init__(self, name: str):
        self.name = name
        self.closed = False

    def close(self):
        self.closed = True


class _Provider(FaceRecoProvider):
    def __init__(self):
        self.batches: list[int] = []
        self.threads: set[int] = set()
        self.single_calls = 0
        self.fail_batches = False

    def provider_name(self):
        return "fake"

    def predict_player(self, image_bgr, body):
        self.single_calls += 1
        self.threads.add(threading.get_ident())
        return Player(confidence=None, internal={"embedding": np.full(4, image_bgr[0, 0, 0], np.float32)})

    def predict_players(self, images_bgr, bodies):
        if self.fail_batches:
            raise MemoryError("batch too big")
        self.batches.append(len(images_bgr))
        self.threads.add(threading.get_ident())
        return [Player(confidence=None, internal={"embedding": np.full(4, img[0, 0, 0], np.float32)})
                for img in images_bgr]


def _qualified(n_photos: int, bodies_per_photo: int = 3) -> list[_QualifiedBody]:
    out = []
    for p in range(n_photos):
        for b in range(bodies_per_photo):
            body = BodyRecord(
                orig_filename=f"IMG_{p:04d}.jpg", body_index=b,
                body_bbox=Box(0.0, 0.0, 1.0, 1.0),
                face_bbox=Box(0.1 + 0.25 * b, 0.1, 0.3 + 0.25 * b, 0.5), narrow_face_bbox=None,
                cloth_color="White", is_blurry=False, confidence=0.9, raw_body={},
            )
            out.append(_QualifiedBody(image_path=Path(f"C:/shots/IMG_{p:04d}.jpg"), body=body))
    return out


@pytest.fixture
def pipeline(monkeypatch):
    """A pipeline whose decode, landmarker, prepare and annotate steps are
    fast fakes; each fake records what the test needs to assert on."""
    state = {"decodes": {}, "active_decodes": 0, "peak_decodes": 0, "used": {}, "created": [],
             "shared": _Landmarker("shared")}
    lock = threading.Lock()

    def decode(self, path):
        with lock:
            state["decodes"][path] = state["decodes"].get(path, 0) + 1
            state["active_decodes"] += 1
            state["peak_decodes"] = max(state["peak_decodes"], state["active_decodes"])
        time.sleep(0.002)
        with lock:
            state["active_decodes"] -= 1
        if path.name == "IMG_0005.jpg":
            return None
        value = int(path.stem.split("_")[1]) % 250 + 1
        return np.full((400, 400, 3), value, np.uint8)

    def create(num_faces=1):
        lm = _Landmarker(f"own{len(state['created'])}")
        state["created"].append(lm)
        return lm

    def prepare(model, crop, align=True):
        time.sleep(random.uniform(0, 0.003))
        with lock:
            state["used"].setdefault(threading.get_ident(), set()).add(model.name)
        return PreparedFaceCrop(crop, make_crop_body(Box(0, 0, 1, 1), None), [], None, Box(0, 0, 1, 1), None)

    monkeypatch.setattr(FaceRecoPipeline, "_decode_image", decode)
    monkeypatch.setattr(facereco, "load_face_model", lambda **k: state["shared"])
    monkeypatch.setattr(mediapipe_provider, "create_face_landmarker", create)
    monkeypatch.setattr(facereco, "prepare_face_crop", prepare)
    monkeypatch.setattr(facereco, "annotate_face_crop", lambda crop, *a: crop)

    provider = _Provider()
    pipe = FaceRecoPipeline(provider, FaceRecoConfig(workers=4), cpu_only=False)
    return pipe, provider, state


def test_results_keep_order_and_unreadable_photo_is_skipped(pipeline):
    pipe, provider, _ = pipeline
    qualified = _qualified(40)
    predicted = pipe._predict_samples(qualified)

    expected = [q for q in qualified if q.image_path.name != "IMG_0005.jpg"]
    assert [p[0] for p in predicted] == expected
    for item, player, crop, annotated in predicted:
        assert player.confidence == 0.9  # fallback from the body record
        assert float(player.internal["embedding"][0]) == float(crop[0, 0, 0])


def test_each_photo_decoded_once_and_decodes_never_overlap(pipeline):
    pipe, _, state = pipeline
    pipe._predict_samples(_qualified(40))
    assert set(state["decodes"].values()) == {1}
    assert len(state["decodes"]) == 40
    assert state["peak_decodes"] == 1


def test_provider_runs_on_the_calling_thread_in_batches(pipeline):
    pipe, provider, _ = pipeline
    pipe._predict_samples(_qualified(200))  # 600 faces -> more than one round
    assert provider.threads == {threading.get_ident()}
    assert provider.single_calls == 0
    assert len(provider.batches) > 1 and max(provider.batches) > 100


def test_each_worker_owns_a_landmarker_and_extras_are_closed(pipeline):
    pipe, _, state = pipeline
    pipe._predict_samples(_qualified(60))
    names = [next(iter(n)) for n in state["used"].values()]
    assert all(len(n) == 1 for n in state["used"].values()), "a thread switched landmarkers"
    assert len(set(names)) == len(names), "two threads shared a landmarker"
    assert "shared" in names
    assert len(state["created"]) == len(names) - 1
    assert all(lm.closed for lm in state["created"])
    assert not state["shared"].closed


def test_single_worker_and_yolo_engine_prepare_on_one_thread(pipeline):
    pipe, _, state = pipeline
    pipe.config.workers = 1
    pipe._predict_samples(_qualified(20))
    assert len(state["used"]) == 1 and state["created"] == []

    state["used"].clear()
    pipe.config.workers = 4
    pipe.config.engine = "yolo"
    pipe._predict_samples(_qualified(20))
    assert len(state["used"]) == 1


def test_failed_batch_falls_back_to_one_face_at_a_time(pipeline):
    pipe, provider, _ = pipeline
    provider.fail_batches = True
    predicted = pipe._predict_samples(_qualified(10))
    assert len(predicted) == 27  # 10 photos x 3, minus the unreadable one
    assert provider.single_calls == 27


def test_worker_error_propagates_without_hanging(pipeline, monkeypatch):
    pipe, _, _ = pipeline
    calls = 0
    lock = threading.Lock()

    def boom(model, crop, align=True):
        nonlocal calls
        with lock:
            calls += 1
            if calls == 30:
                raise RuntimeError("landmarker blew up")
        return PreparedFaceCrop(crop, make_crop_body(Box(0, 0, 1, 1), None), [], None, Box(0, 0, 1, 1), None)

    monkeypatch.setattr(facereco, "prepare_face_crop", boom)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="landmarker blew up"):
        pipe._predict_samples(_qualified(40))
    assert time.monotonic() - started < 10


def test_png_writer_writes_every_file(tmp_path):
    writer = _PngWriter(4)
    for i in range(30):
        writer.write(tmp_path / f"{i}.png", np.full((20, 20, 3), i, np.uint8))
    writer.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(f"{i}.png" for i in range(30))


def test_facenet_batch_matches_one_at_a_time():
    facenet = pytest.importorskip("algo.facenet_provider")
    if facenet.torch is None:
        pytest.skip("facenet-pytorch not installed")
    from facenet_pytorch.models.inception_resnet_v1 import get_torch_home
    weights = Path(get_torch_home()) / "checkpoints"
    if not any(weights.glob("*vggface2*")):
        pytest.skip("FaceNet weights not downloaded")

    provider = facenet.FaceNetFaceRecoProvider(batch_size=3)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(5)]
    bodies = [make_crop_body(Box(0, 0, 1, 1), None) for _ in images]
    bodies[2] = make_crop_body(Box(0.5, 0.5, 0.5, 0.5), None)  # degenerate box -> no embedding

    batched = provider.predict_players(images, bodies)
    singles = [provider.predict_player(i, b) for i, b in zip(images, bodies)]
    assert batched[2].internal["embedding"] is None and singles[2].internal["embedding"] is None
    for b, s in zip(batched, singles):
        if s.internal["embedding"] is not None:
            np.testing.assert_allclose(b.internal["embedding"], s.internal["embedding"], atol=1e-5)
