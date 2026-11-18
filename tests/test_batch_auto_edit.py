"""Batch Auto Edit (Album Tools): the batched LLM crop requests
(algo/llm/auto_crop.py), autoedit.py's --jobs-file mode, and the server job
that plans the crops and bodies and then runs one autoedit.py process."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import autoedit
import culling_app
from algo import album as album_module
from algo.llm import auto_crop
from conftest import make_body_dict, make_entry, write_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient


# ---- the batched LLM request ------------------------------------------------ #

def _box(x1=0.1, y1=0.2, x2=0.9, y2=0.8) -> dict:
    return {"crop": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}, "removed": "a coach", "reason": "Thirds."}


def _answer(files, **over) -> str:
    return json.dumps({"photos": [{"file": f, "instagram": _box(), "original": _box(0, 0, 1, 1), **over} for f in files]})


def _photo(name: str, width=200, height=100) -> auto_crop.CropPhoto:
    return auto_crop.CropPhoto(file=name, image_b64="QUJD", width=width, height=height)


def test_batch_prompt_names_the_photos_crops_and_answer_shape():
    prompt = auto_crop.build_batch_prompt(6, "White:Bright")
    assert "6 unrelated photos" in prompt and "wears bright white jerseys" in prompt
    assert '"instagram": 4:5 (width:height)' in prompt and "(x2 - x1) * W = 0.800 * (y2 - y1) * H" in prompt
    assert '"original": the photo\'s own proportions, so x2 - x1 = y2 - y1' in prompt
    assert '{"photos": [{"file": "<file name>", "instagram": {"crop"' in prompt
    assert "Rule of thirds" in prompt and "5 to 10%" in prompt  # the shared editorial rules
    assert "LEVEL" not in prompt and "level_degrees" not in prompt
    assert "jerseys" not in auto_crop.build_batch_prompt(2)


def test_parse_batch_matches_photos_by_name_and_drops_the_rest():
    raw = _answer(["a.jpg", "b.jpg", "stranger.jpg"])
    parsed = auto_crop.parse_batch_suggestions(raw, ["b.jpg", "a.jpg", "c.jpg"])
    assert set(parsed) == {"a.jpg", "b.jpg"}
    assert parsed["a.jpg"]["instagram"].box == (0.1, 0.2, 0.9, 0.8)
    assert parsed["a.jpg"]["instagram"].reason == "Thirds." and parsed["a.jpg"]["instagram"].removed == "a coach"
    assert parsed["b.jpg"]["original"].box == (0.0, 0.0, 1.0, 1.0)


def test_parse_batch_keeps_a_photos_usable_crop_when_the_other_is_bad():
    raw = json.dumps({"photos": [
        {"file": "a.jpg", "instagram": _box(0.1, 0.1, 0.12, 0.9), "original": _box()},
        {"file": "b.jpg", "instagram": "nope", "original": None},
        {"file": ["not", "a", "name"], "instagram": _box()},
        "junk",
    ]})
    parsed = auto_crop.parse_batch_suggestions(raw, ["a.jpg", "b.jpg"])
    assert set(parsed) == {"a.jpg"} and set(parsed["a.jpg"]) == {"original"}


@pytest.mark.parametrize("raw", [None, "", "not json", "[]", json.dumps({"photos": "x"}), json.dumps({})])
def test_parse_batch_never_raises_on_an_unusable_answer(raw):
    assert auto_crop.parse_batch_suggestions(raw, ["a.jpg"]) == {}


def test_photos_are_split_into_batches_of_six():
    photos = [_photo(f"{i}.jpg") for i in range(14)]
    assert [len(b) for b in auto_crop.split_batches(photos)] == [6, 6, 2]
    assert auto_crop.AUTO_CROP_BATCH_SIZE == 6
    assert auto_crop.split_batches([]) == []


class FakeBatchProvider:
    model = "fake-model"
    max_concurrency = 3

    def __init__(self, answer=None):
        self.calls: list[tuple[str, list]] = []
        self.answer = answer or (lambda files: _answer(files))

    def ask_json_images(self, instructions, images):
        self.calls.append((instructions, images))
        files = [re.match(r"Image \d+ \((.+?), \d+ x \d+ pixels\)", label).group(1) for label, _ in images]
        return self.answer(files), 0.5

    def get_cost_summary(self):
        return SimpleNamespace(total_cost_usd=0.01)


def test_one_request_carries_every_photo_of_a_batch_with_its_label():
    provider = FakeBatchProvider()
    parsed = auto_crop.suggest_crops_batch(provider, [_photo("a.jpg", 300, 200), _photo("b.jpg")], jersey_color="Navy")
    (instructions, images), = provider.calls
    assert images == [("Image 1 (a.jpg, 300 x 200 pixels):", "QUJD"), ("Image 2 (b.jpg, 200 x 100 pixels):", "QUJD")]
    assert "2 unrelated photos" in instructions and "navy" in instructions
    assert set(parsed) == {"a.jpg", "b.jpg"}


def test_batches_run_concurrently_in_order_and_a_failed_one_is_empty():
    batches = auto_crop.split_batches([_photo(f"{i}.jpg") for i in range(13)])  # 6 + 6 + 1
    provider = FakeBatchProvider()
    original = provider.ask_json_images

    def flaky(instructions, images):
        if len(images) == 1:
            raise RuntimeError("rate limited")
        return original(instructions, images)

    provider.ask_json_images = flaky
    progress = []
    results = auto_crop.suggest_crops_batches(
        provider, batches, on_batch_done=lambda done, total, error: progress.append((done, total, error is not None)),
    )
    assert [len(r) for r in results] == [6, 6, 0]
    assert set(results[0]) == {f"{i}.jpg" for i in range(6)}
    assert sorted(p[0] for p in progress) == [1, 2, 3] and all(p[1] == 3 for p in progress)
    assert sum(p[2] for p in progress) == 1
    assert auto_crop.suggest_crops_batches(provider, []) == []


def test_the_provider_sends_every_image_of_a_batch_in_one_request():
    from algo.llm.culling_provider import OpenAIProvider

    sent = []
    provider = object.__new__(OpenAIProvider)  # no openai SDK needed for this
    provider.model = "gpt-6-luna"
    provider._image_detail = None
    provider._chat_completion = lambda messages, tokens: (sent.append((messages, tokens)) or ('{"photos": []}', 0.5))

    assert provider.ask_json_images("Crop these.", [("Image 1 (a.jpg):", "QUJD"), ("Image 2 (b.jpg):", "REVG")]) == (
        '{"photos": []}', 0.5)
    (messages, tokens), = sent
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Crop these."}
    assert [c.get("text") for c in content[1::2]] == ["Image 1 (a.jpg):", "Image 2 (b.jpg):"]
    assert [c["image_url"]["url"] for c in content[2::2]] == [
        "data:image/jpeg;base64,QUJD", "data:image/jpeg;base64,REVG"]
    assert tokens > 0


@pytest.mark.parametrize("box, ratio, width, height", [
    ((0.2, 0.1, 0.7, 0.9), 0.8, 768, 512),    # reshaped about its centre
    ((0.0, 0.0, 1.0, 1.0), 0.8, 768, 512),    # too wide: shrinks to the photo's height
    ((0.0, 0.0, 1.0, 1.0), 2.5, 600, 600),    # too tall: shrinks to the photo's width
    ((0.8, 0.8, 1.0, 1.0), 1.5, 3000, 2000),  # in a corner: shifted back inside
    ((0.3, 0.3, 0.6, 0.5), 1.0, 640, 480),
])
def test_fit_box_to_ratio_gives_the_exact_shape_inside_the_photo(box, ratio, width, height):
    x1, y1, x2, y2 = auto_crop.fit_box_to_ratio(box, ratio, width, height)
    assert -1e-9 <= x1 < x2 <= 1 + 1e-9 and -1e-9 <= y1 < y2 <= 1 + 1e-9
    assert ((x2 - x1) * width) / ((y2 - y1) * height) == pytest.approx(ratio)


def test_fit_box_to_ratio_keeps_the_centre_and_area_when_it_fits():
    x1, y1, x2, y2 = auto_crop.fit_box_to_ratio((0.2, 0.1, 0.7, 0.9), 0.8, 768, 512)
    assert (x1 + x2) / 2 == pytest.approx(0.45) and (y1 + y2) / 2 == pytest.approx(0.5)
    assert ((x2 - x1) * 768) * ((y2 - y1) * 512) == pytest.approx(0.5 * 768 * 0.8 * 512)


def test_a_chosen_player_is_named_in_each_label_and_the_prompt():
    provider = FakeBatchProvider()
    photos = [
        auto_crop.CropPhoto(file="a.jpg", image_b64="QUJD", width=300, height=200, focus=(0.1, 0.25, 0.3, 0.9)),
        auto_crop.CropPhoto(file="b.jpg", image_b64="QUJD", width=300, height=200, focus=(0.5, 0.2, 0.75, 1.0)),
    ]
    parsed = auto_crop.suggest_crops_batch(provider, photos, jersey_color="Navy")
    (instructions, images), = provider.calls
    assert images[0][0] == (
        "Image 1 (a.jpg, 300 x 200 pixels), primary athlete in box x1=0.100, y1=0.250, x2=0.300, y2=0.900:")
    assert images[1][0].endswith("x1=0.500, y1=0.200, x2=0.750, y2=1.000:")
    assert "inside the box given in its label" in instructions and "jerseys" not in instructions
    assert set(parsed) == {"a.jpg", "b.jpg"}
    assert "inside the box" not in auto_crop.build_batch_prompt(2, "Navy")


def test_photos_without_a_chosen_player_keep_the_plain_label():
    provider = FakeBatchProvider()
    auto_crop.suggest_crops_batch(provider, [_photo("a.jpg", 300, 200)])
    assert provider.calls[0][1][0][0] == "Image 1 (a.jpg, 300 x 200 pixels):"


def test_crop_ratio_is_4_to_5_or_the_photos_own():
    assert auto_crop.crop_ratio("instagram", 6000, 4000) == pytest.approx(0.8)
    assert auto_crop.crop_ratio("original", 6000, 4000) == pytest.approx(1.5)


def _temperature_rejecting_provider(flip_flag_first: bool):
    import threading

    import httpx
    import openai
    from algo.llm.culling_provider import OpenAIProvider

    rejection = openai.BadRequestError(
        "Unsupported value: 'temperature'",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body={"message": "Unsupported value: 'temperature'", "type": "invalid_request_error",
              "param": "temperature", "code": "unsupported_value"},
    )
    provider = object.__new__(OpenAIProvider)
    provider.model = "gpt-6-luna"
    provider._temperature_supported = None
    provider._limits_lock = threading.Lock()
    sent = []

    def create(**kwargs):
        sent.append("temperature" in kwargs)
        if "temperature" in kwargs:
            if flip_flag_first:
                provider._temperature_supported = False  # a concurrent request's 400 got there first
            raise rejection
        return "response"

    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))
    return provider, sent


def test_a_model_that_rejects_temperature_is_retried_without_it_and_remembered():
    provider, sent = _temperature_rejecting_provider(flip_flag_first=False)
    assert provider._post_chat_completion([]) == "response"
    assert provider._post_chat_completion([]) == "response"
    assert sent == [True, False, False]  # later requests do not send it at all


def test_a_concurrent_request_that_lost_the_race_still_retries_without_temperature():
    # Two requests in flight at once, as in a batch: the first 400 flips the
    # shared flag, and the second must retry anyway because it did send one.
    provider, sent = _temperature_rejecting_provider(flip_flag_first=True)
    assert provider._post_chat_completion([]) == "response"
    assert sent == [True, False]


# ---- autoedit.py --jobs-file ------------------------------------------------- #

def _write_jobs(path: Path, jobs) -> Path:
    path.write_text(json.dumps(jobs), encoding="utf-8")
    return path


def test_load_jobs_file_reads_crop_and_bodies_and_skips_bad_entries(tmp_path):
    jobs_file = _write_jobs(tmp_path / "jobs.json", [
        {"source": str(tmp_path / "a.jpg"), "out_file": str(tmp_path / "out" / "a_ai_edited.jpg"),
         "crop": [10, 20, 30.4, 40], "boxes": [
             {"box": [0.1, 0.2, 0.3, 0.4], "keypoints": [[0.1, 0.2, 0.9]]},
             {"box": [0.5, 0.5, 0.9, 0.9], "keypoints": [], "protect": True}]},
        {"source": str(tmp_path / "b.jpg"), "out_file": str(tmp_path / "out" / "b.jpg"), "crop": None},
        {"source": str(tmp_path / "c.jpg")},                                       # no out_file
        {"source": "d.jpg", "out_file": "d_out.jpg", "crop": [1, 2, 0, 5]},        # empty crop
    ])
    jobs = autoedit.load_jobs_file(jobs_file)
    assert [j.source.name for j in jobs] == ["a.jpg", "b.jpg"]
    a, b = jobs
    assert a.crop == (10, 20, 30, 40) and a.sharpens
    assert a.boxes == [(0.1, 0.2, 0.3, 0.4)] and a.keypoints == [[(0.1, 0.2, 0.9)]]
    assert a.protected == [((0.5, 0.5, 0.9, 0.9), [])]
    assert b.crop is None and not b.sharpens and b.boxes == []


def test_a_job_with_only_protected_bodies_does_not_sharpen(tmp_path):
    jobs = autoedit.load_jobs_file(_write_jobs(tmp_path / "jobs.json", [
        {"source": "a.jpg", "out_file": "o.jpg", "boxes": [{"box": [0, 0, 1, 1], "protect": True}]}]))
    assert not jobs[0].sharpens


@pytest.mark.parametrize("content", ["{}", "not json", '"x"'])
def test_load_jobs_file_rejects_a_file_that_is_not_a_job_list(tmp_path, content):
    path = tmp_path / "jobs.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        autoedit.load_jobs_file(path)


def _photo_file(path: Path, width=200, height=100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), np.uint8)
    image[:, : width // 2] = (40, 120, 200)
    assert cv2.imwrite(str(path), image)
    return path


def _args(**over) -> SimpleNamespace:
    values = dict(output=Path("unused"), ext="jpg", compression=95, workers=2, face_enhance=False,
                  sharpen_model=autoedit.DEFAULT_SHARPEN_MODEL)
    values.update(over)
    return SimpleNamespace(**values)


def test_crop_only_jobs_run_without_loading_any_sharpen_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the sharpen models must not be loaded for crop-only jobs")

    monkeypatch.setattr(autoedit.SharpenDispatcher, "start", forbidden)
    jobs = [
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / f"{n}.jpg"), out_file=tmp_path / "out" / f"{n}_ai_edited.jpg",
                         crop=(10, 10, 100, 50) if n == "a" else None)
        for n in ("a", "b")
    ]
    assert autoedit.run_jobs(jobs, _args()) == 0
    assert cv2.imread(str(tmp_path / "out" / "a_ai_edited.jpg")).shape[:2] == (50, 100)
    assert cv2.imread(str(tmp_path / "out" / "b_ai_edited.jpg")).shape[:2] == (100, 200)


class _RecordingDispatcher:
    """Stands in for SharpenDispatcher inside run_jobs: records how it was
    built and every whole-frame submit."""

    created: list = []
    whole_frame: list = []

    def __init__(self, spec, face_enhance, cpu_spillover=True):
        type(self).created.append((face_enhance, cpu_spillover))

    def start(self):
        pass

    def shutdown(self):
        pass

    def submit(self, image, face_enhance=True):
        type(self).whole_frame.append(face_enhance)
        return image


@pytest.fixture
def recording_dispatcher(monkeypatch):
    _RecordingDispatcher.created, _RecordingDispatcher.whole_frame = [], []
    monkeypatch.setattr(autoedit, "SharpenDispatcher", _RecordingDispatcher)
    return _RecordingDispatcher


def _fake_sharpen_bodies(sharpened, ok=True):
    def fake(image, boxes, dispatcher, **kwargs):
        sharpened.append((kwargs["label"], boxes, kwargs["protected_norm"]))
        kwargs["report"].update(sharpened=len(boxes) if ok else 0, total=len(boxes))
        return image

    return fake


def test_one_dispatcher_serves_every_sharpening_job_and_loads_without_cpu_spillover(
    tmp_path, monkeypatch, recording_dispatcher,
):
    sharpened = []
    monkeypatch.setattr(autoedit, "sharpen_bodies", _fake_sharpen_bodies(sharpened))
    protect = ((0.6, 0.1, 0.9, 0.9), [])
    jobs = [
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "crop_only.jpg"), out_file=tmp_path / "out" / "c.jpg"),
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "one.jpg"), out_file=tmp_path / "out" / "1.jpg",
                         boxes=[(0.1, 0.1, 0.4, 0.9)], keypoints=[[]], protected=[protect]),
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "two.jpg"), out_file=tmp_path / "out" / "2.jpg",
                         crop=(0, 0, 100, 100), boxes=[(0.2, 0.2, 0.5, 0.8)], keypoints=[[]]),
    ]
    assert autoedit.run_jobs(jobs, _args(face_enhance=True)) == 0
    assert recording_dispatcher.created == [(True, False)]  # one dispatcher, face enhance on, no CPU copy
    assert sorted(label for label, _, _ in sharpened) == ["one.jpg", "two.jpg"]  # never the crop-only photo
    assert recording_dispatcher.whole_frame == []
    by_label = {label: (boxes, protected) for label, boxes, protected in sharpened}
    assert by_label["one.jpg"] == ([(0.1, 0.1, 0.4, 0.9)], [protect])
    assert all((tmp_path / "out" / name).is_file() for name in ("c.jpg", "1.jpg", "2.jpg"))
    assert cv2.imread(str(tmp_path / "out" / "2.jpg")).shape[:2] == (100, 100)


def test_a_sharpening_job_without_bodies_is_sharpened_whole_and_never_face_enhanced(
    tmp_path, monkeypatch, recording_dispatcher,
):
    sharpened = []
    monkeypatch.setattr(autoedit, "sharpen_bodies", _fake_sharpen_bodies(sharpened))
    jobs = [
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "none.jpg"), out_file=tmp_path / "out" / "n.jpg",
                         sharpen=True),
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "crop_only.jpg"), out_file=tmp_path / "out" / "c.jpg"),
    ]
    assert autoedit.run_jobs(jobs, _args(face_enhance=True)) == 0
    # no job has a body, so the face-restoration model is not even requested
    assert recording_dispatcher.created == [(False, False)]
    assert recording_dispatcher.whole_frame == [False] and sharpened == []


def test_a_job_whose_players_cannot_be_segmented_falls_back_to_the_whole_frame(
    tmp_path, monkeypatch, recording_dispatcher,
):
    sharpened = []
    monkeypatch.setattr(autoedit, "sharpen_bodies", _fake_sharpen_bodies(sharpened, ok=False))
    jobs = [autoedit.EditJob(source=_photo_file(tmp_path / "src" / "p.jpg"), out_file=tmp_path / "out" / "p.jpg",
                             boxes=[(0.1, 0.1, 0.4, 0.9)], keypoints=[[]])]
    assert autoedit.run_jobs(jobs, _args(face_enhance=True)) == 0
    assert [label for label, _, _ in sharpened] == ["p.jpg"]      # tried the player first
    assert recording_dispatcher.whole_frame == [False]            # then the whole frame, without face enhancement
    assert recording_dispatcher.created == [(True, False)]


def test_the_sharpen_flag_defaults_to_having_bodies_and_can_be_set_either_way(tmp_path):
    jobs = autoedit.load_jobs_file(_write_jobs(tmp_path / "jobs.json", [
        {"source": "a.jpg", "out_file": "a.jpg", "boxes": [{"box": [0, 0, 1, 1]}]},
        {"source": "b.jpg", "out_file": "b.jpg"},
        {"source": "c.jpg", "out_file": "c.jpg", "sharpen": True},
        {"source": "d.jpg", "out_file": "d.jpg", "boxes": [{"box": [0, 0, 1, 1]}], "sharpen": False},
    ]))
    assert [j.sharpens for j in jobs] == [True, False, True, False]
    assert autoedit.EditJob(source=Path("x"), out_file=Path("y"), boxes=[(0, 0, 1, 1)]).sharpens
    assert not autoedit.EditJob(source=Path("x"), out_file=Path("y")).sharpens

def test_load_jobs_file_reads_strict_box(tmp_path):
    jobs = autoedit.load_jobs_file(_write_jobs(tmp_path / "jobs.json", [
        {"source": "a.jpg", "out_file": "a_out.jpg", "boxes": [{"box": [0, 0, 1, 1]}], "strict_box": True},
        {"source": "b.jpg", "out_file": "b_out.jpg", "boxes": [{"box": [0, 0, 1, 1]}]},
    ]))
    assert [j.strict_box for j in jobs] == [True, False]


class _Brighten:
    def submit(self, crop):
        return np.clip(crop.astype(np.int16) + 60, 0, 255).astype(np.uint8)


class _EveryoneIsAPerson:
    def segment(self, crop):
        return np.ones(crop.shape[:2], np.float32)


@pytest.mark.parametrize("margin, sharpened_outside", [
    (None, True),   # default: may reach a little past the player's box (a limb poking out)
    (0.0, False),   # strict: never past it
])
def test_sharpen_bodies_can_be_kept_strictly_inside_the_players_box(monkeypatch, margin, sharpened_outside):
    import algo.segmentation as segmentation

    monkeypatch.setattr(segmentation, "get_segmenter", lambda name: _EveryoneIsAPerson())
    image = np.full((800, 1000, 3), 100, np.uint8)
    kwargs = {} if margin is None else {"clip_margin_pct": margin}
    out = autoedit.sharpen_bodies(image, [(0.2, 0.2, 0.8, 0.8)], _Brighten(), label="t.jpg", **kwargs)
    assert out[400, 500].tolist() == [160, 160, 160]        # inside the box
    just_outside = int(out[400, 200 - 20][0])                 # 20 px left of the box (the margin is 30 px)
    assert (just_outside > 150) is sharpened_outside
    assert (just_outside == 100) is (not sharpened_outside)
    assert (out[:, :100] == 100).all() and (out[:100] == 100).all()  # far from the box: untouched


def test_sharpen_bodies_default_margin_is_unchanged():
    import inspect

    default = inspect.signature(autoedit.sharpen_bodies).parameters["clip_margin_pct"].default
    assert default == autoedit.BODY_SHARPEN_CLIP_MARGIN_PCT == 0.05


def test_a_strict_box_job_sharpens_with_no_margin_and_a_normal_job_with_the_default(tmp_path, monkeypatch):
    margins = {}

    class FakeDispatcher:
        def __init__(self, spec, face_enhance, cpu_spillover=True):
            pass

        def start(self):
            pass

        def shutdown(self):
            pass

    def fake_sharpen(image, boxes, dispatcher, **kw):
        margins[kw["label"]] = kw["clip_margin_pct"]
        kw["report"].update(sharpened=1, total=1)
        return image

    monkeypatch.setattr(autoedit, "SharpenDispatcher", FakeDispatcher)
    monkeypatch.setattr(autoedit, "sharpen_bodies", fake_sharpen)
    jobs = [
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "strict.jpg"), out_file=tmp_path / "out" / "s.jpg",
                         boxes=[(0.1, 0.1, 0.4, 0.9)], keypoints=[[]], strict_box=True),
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "normal.jpg"), out_file=tmp_path / "out" / "n.jpg",
                         boxes=[(0.1, 0.1, 0.4, 0.9)], keypoints=[[]]),
    ]
    assert autoedit.run_jobs(jobs, _args()) == 0
    assert margins == {"strict.jpg": 0.0, "normal.jpg": autoedit.BODY_SHARPEN_CLIP_MARGIN_PCT}


def test_run_jobs_counts_a_photo_that_cannot_be_read_as_failed(tmp_path):
    jobs = [
        autoedit.EditJob(source=_photo_file(tmp_path / "src" / "ok.jpg"), out_file=tmp_path / "out" / "ok.jpg"),
        autoedit.EditJob(source=tmp_path / "src" / "missing.jpg", out_file=tmp_path / "out" / "missing.jpg"),
    ]
    assert autoedit.run_jobs(jobs, _args()) == 1


def test_dispatcher_skips_the_cpu_copy_only_when_a_gpu_does_the_work(monkeypatch):
    import sys
    import types

    loaded = []

    class FakeProcessor:
        def __init__(self, device, *args):
            loaded.append(device)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
        is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda i: "GPU"))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(autoedit, "LocalSharpenProcessor", FakeProcessor)
    monkeypatch.setattr(autoedit, "_ensure_sharpen_weights", lambda spec: "w")
    monkeypatch.setattr(autoedit, "_ensure_gfpgan_weights", lambda: "g")
    monkeypatch.setattr(autoedit.SharpenDispatcher, "_spawn", lambda self, *a: None)

    autoedit.SharpenDispatcher(cpu_spillover=False).start()
    assert loaded == ["cuda:0"]
    loaded.clear()
    autoedit.SharpenDispatcher().start()
    assert loaded == ["cuda:0", "cpu"]
    loaded.clear()
    fake_torch.cuda.is_available = lambda: False
    autoedit.SharpenDispatcher(cpu_spillover=False).start()
    assert loaded == ["cpu"]  # with no GPU the CPU is the only worker, spillover or not


def test_jobs_file_conflicts_with_the_single_photo_options(tmp_path, monkeypatch):
    jobs_file = _write_jobs(tmp_path / "jobs.json", [])
    monkeypatch.setattr("sys.argv", ["autoedit.py", str(tmp_path / "a.jpg"), "--jobs-file", str(jobs_file)])
    with pytest.raises(SystemExit) as exit_info:
        autoedit.main()
    assert exit_info.value.code == 2
    monkeypatch.setattr("sys.argv", ["autoedit.py", "--jobs-file", str(jobs_file)])
    with pytest.raises(SystemExit) as exit_info:
        autoedit.main()
    assert exit_info.value.code == 1  # an empty job list is an error


# ---- the server job ---------------------------------------------------------- #

def test_output_names_follow_the_pattern_and_keep_a_raw_jpeg_pair_apart():
    names = culling_app._batch_output_names(["IMG_0001.jpg", "IMG_0002.CR3", "IMG_0002.JPG", "other.png"])
    assert names == {
        "IMG_0001.jpg": "IMG_0001_ai_edited.jpg",
        "IMG_0002.CR3": "IMG_0002_CR3_ai_edited.jpg",
        "IMG_0002.JPG": "IMG_0002_JPG_ai_edited.jpg",
        "other.png": "other_ai_edited.jpg",
    }


def test_output_names_are_unique_even_when_only_the_case_differs():
    names = culling_app._batch_output_names(["a.jpg", "A.jpg"])
    assert len({n.lower() for n in names.values()}) == 2
    assert all(n.endswith("_ai_edited.jpg") or "_ai_edited_" in n for n in names.values())


def _bodies(*teammate_flags: bool) -> list:
    return [
        make_body_dict(is_teammate=flag, body_bbox={"x1": 0.1 + 0.2 * i, "y1": 0.2, "x2": 0.25 + 0.2 * i, "y2": 0.9})
        for i, flag in enumerate(teammate_flags)
    ]


def _entry(key: str, src: Path, stars, *flags: bool) -> dict:
    return make_entry(key, file=str(src / key), stars=stars, annotation_data={
        "processing_shape": [100, 200], "overall_blurry": False, "evaluated": _bodies(*flags)})


@pytest.fixture
def batch_album(album_dir, tmp_path):
    """5 star A (a teammate and an opponent), 5 star B (no teammate), 4 star C
    and 2 star D, all real 200 x 100 photos."""
    src = tmp_path / "src"
    keys = {"A.jpg": (5, (True, False)), "B.jpg": (5, (False,)), "C.jpg": (4, (True,)), "D.jpg": (2, ())}
    entries = []
    for key, (stars, flags) in keys.items():
        _photo_file(src / key)
        entries.append(_entry(key, src, stars, *flags))
    write_album(album_dir, entries)
    (album_dir / "info.json").write_text(json.dumps({
        "SrcDir": str(src), "Timestamp": "20260101-000000",
        "Anno_Sharp": [{"src": k, "srcPath": str(src / k)} for k in keys],
        "Anno_Blur": [], "Anno_Skipped": [],
    }), encoding="utf-8")
    album_module.invalidate_all()
    return album_dir


@pytest.fixture
def client(batch_album, monkeypatch):
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: batch_album)
    monkeypatch.setattr(culling_app, "_album_team_id", lambda album_dir: "team-1")
    team = SimpleNamespace(openaiApiKey="sk-test")
    monkeypatch.setattr(culling_app, "_resolve_team", lambda team_id: team)
    yield TestClient(culling_app.app)
    culling_app.batch_auto_edit_state.mark_stopped()
    culling_app.batch_auto_edit_state.reset()


@pytest.fixture
def autoedit_runs(monkeypatch):
    """Replaces the autoedit.py subprocess: records the jobs file it was given
    and writes every output."""
    runs = []

    def fake(cmd, on_line=None, log_fn=None):
        jobs = json.loads(Path(cmd[cmd.index("--jobs-file") + 1]).read_text(encoding="utf-8"))
        runs.append(SimpleNamespace(cmd=cmd, jobs=jobs))
        for job in jobs:
            Path(job["out_file"]).write_bytes(b"jpeg")
        return 0

    monkeypatch.setattr(culling_app, "_stream_subprocess", fake)
    return runs


@pytest.fixture
def provider(monkeypatch):
    fake = FakeBatchProvider()
    monkeypatch.setattr(auto_crop, "make_provider", lambda api_key: fake)
    return fake


def _wait(client, timeout=15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/album-tools/batch-auto-edit/status").json()
        if not status["running"]:
            return status
        time.sleep(0.02)
    raise AssertionError("the batch auto edit job did not finish")


def _start(client, tmp_path, **over):
    body = {"destination": str(tmp_path / "out"), "resize": "instagram", "sharpen": True, "minStars": 5}
    body.update(over)
    return client.post("/api/album-tools/batch-auto-edit/start", json=body)


def _tag_players(album: Path, tags: dict[str, list[tuple[str, int]]]) -> None:
    """Names people the way the album's face recognition does: one named
    cluster per player, holding the (photo, body index) pairs they're in."""
    for name, faces in tags.items():
        cluster = album / ".FaceReco" / name
        cluster.mkdir(parents=True, exist_ok=True)
        (cluster / "face.json").write_text(json.dumps({"name": name, "faces": [
            {"origFilename": key, "cropFileName": "a.png", "Body": {"body_bbox": {
                "x1": 0.1 + 0.2 * index, "y1": 0.2, "x2": 0.25 + 0.2 * index, "y2": 0.9}}}
            for key, index in faces
        ]}), encoding="utf-8")


@pytest.fixture
def tagged(batch_album):
    """In A.jpg Ann Lee is the teammate (body 0) and Jane Doe the opponent
    (body 1); Ann Lee is also the teammate in C.jpg."""
    _tag_players(batch_album, {"Ann Lee": [("A.jpg", 0), ("C.jpg", 0)], "Jane Doe": [("A.jpg", 1)]})
    return batch_album


def test_the_picker_lists_the_tagged_players(client, tagged):
    assert client.get("/api/album-tools/batch-auto-edit/players").json() == {"players": ["Ann Lee", "Jane Doe"]}


def test_the_picker_is_empty_when_nobody_is_tagged(client):
    assert client.get("/api/album-tools/batch-auto-edit/players").json() == {"players": []}


def test_a_player_limits_the_photos_to_the_ones_they_are_in(tagged):
    assert [im["file"] for im in culling_app._batch_candidates(tagged, 1, "Ann Lee")] == ["A.jpg", "C.jpg"]
    assert [im["file"] for im in culling_app._batch_candidates(tagged, 5, "Ann Lee")] == ["A.jpg"]
    assert [im["file"] for im in culling_app._batch_candidates(tagged, 1, "Jane Doe")] == ["A.jpg"]
    assert [im["file"] for im in culling_app._batch_candidates(tagged, 5)] == ["A.jpg", "B.jpg"]  # all players


def test_a_name_only_matches_the_body_it_was_given_to(tagged):
    album = album_module.album_for(tagged)
    assert culling_app._batch_player_body_ids(tagged, album.image("A.jpg"), "Jane Doe") == [1]
    assert culling_app._batch_player_body_ids(tagged, album.image("A.jpg"), "Ann Lee") == [0]
    assert culling_app._batch_player_body_ids(tagged, album.image("B.jpg"), "Ann Lee") == []


def test_the_focus_box_surrounds_the_chosen_players_bodies(tagged):
    bodies = album_module.album_for(tagged).image("A.jpg").bodies
    assert culling_app._batch_focus_box(bodies, [1]) == pytest.approx((0.3, 0.2, 0.45, 0.9))
    assert culling_app._batch_focus_box(bodies, [0, 1]) == pytest.approx((0.1, 0.2, 0.45, 0.9))
    assert culling_app._batch_focus_box(bodies, []) is None


def test_the_chosen_player_is_sharpened_whether_or_not_they_are_a_teammate(tagged):
    bodies = album_module.album_for(tagged).image("A.jpg").bodies
    entries = culling_app._batch_sharpen_entries(bodies, [1])  # the opponent
    # the chosen player is sharpened and the team's own player is protected
    assert [bool(e.get("protect")) for e in entries] == [False, True]
    assert entries[0]["box"] == pytest.approx([0.3, 0.2, 0.45, 0.9])
    assert entries[1]["box"] == pytest.approx([0.1, 0.2, 0.25, 0.9])
    both = culling_app._batch_sharpen_entries(bodies, [0, 1])
    assert [e["box"] for e in both] == [pytest.approx([0.1, 0.2, 0.25, 0.9]), pytest.approx([0.3, 0.2, 0.45, 0.9])]
    assert culling_app._batch_sharpen_entries(bodies, []) == []


@pytest.mark.parametrize("player, min_stars, count", [
    ("", 5, 2), ("Ann Lee", 5, 1), ("Ann Lee", 4, 2), ("Jane Doe", 5, 1), ("Nobody", 1, 0),
])
def test_summary_counts_the_photos_of_the_chosen_player(client, tagged, player, min_stars, count):
    response = client.get("/api/album-tools/batch-auto-edit/summary", params={"minStars": min_stars, "player": player})
    assert response.json() == {"count": count}


def test_start_rejects_a_player_who_is_not_tagged_in_the_album(client, tagged, tmp_path):
    response = _start(client, tmp_path, player="Nobody")
    assert response.status_code == 400 and "not a tagged player" in response.json()["detail"]
    assert not culling_app.batch_auto_edit_state.running


def test_start_needs_photos_of_the_player_at_the_minimum(client, tagged, tmp_path):
    _tag_players(tagged, {"Bench Player": [("C.jpg", 0)]})  # C.jpg is only 4 stars
    response = _start(client, tmp_path, player="Bench Player", minStars=5)
    assert response.status_code == 400 and "No photos of Bench Player have 5+ stars" in response.json()["detail"]
    assert not culling_app.batch_auto_edit_state.running


def test_a_player_batch_crops_around_and_sharpens_only_that_player(client, tagged, tmp_path, provider, autoedit_runs):
    assert _start(client, tmp_path, player="Jane Doe").status_code == 200
    status = _wait(client)

    assert status["error"] is None
    (instructions, images), = provider.calls
    assert [label for label, _ in images] == [
        "Image 1 (A.jpg, 200 x 100 pixels), primary athlete in box x1=0.300, y1=0.200, x2=0.450, y2=0.900:"]
    assert "inside the box given in its label" in instructions

    run, = autoedit_runs
    job, = run.jobs  # only A.jpg has Jane Doe
    assert Path(job["source"]).name == "A.jpg"
    # the chosen player (an opponent by jersey) is sharpened inside their box; the team's own player is protected
    assert job["strict_box"] is True
    assert [bool(e.get("protect")) for e in job["boxes"]] == [False, True]
    assert job["boxes"][0]["box"] == pytest.approx([0.3, 0.2, 0.45, 0.9])
    assert job["boxes"][1]["box"] == pytest.approx([0.1, 0.2, 0.25, 0.9])
    assert any("1 photo(s) at 5+ stars with Jane Doe" in line for line in status["lines"])
    assert any("centred on the chosen player" in line for line in status["lines"])
    assert any("inside the player's box" in line for line in status["lines"])


def test_a_player_batch_without_resize_never_calls_the_llm(client, tagged, tmp_path, provider, autoedit_runs):
    assert _start(client, tmp_path, player="Ann Lee", resize=None).status_code == 200
    status = _wait(client)
    assert provider.calls == []
    job, = autoedit_runs[0].jobs
    assert Path(job["source"]).name == "A.jpg" and job["crop"] is None and job["strict_box"] is True
    assert [bool(e.get("protect")) for e in job["boxes"]] == [False, True]
    assert job["boxes"][0]["box"] == pytest.approx([0.1, 0.2, 0.25, 0.9])
    assert status["result"]["processed"] == 1


def test_all_players_still_means_no_player_filter(client, tagged, tmp_path, provider, autoedit_runs):
    assert _start(client, tmp_path, player="").status_code == 200
    _wait(client)
    assert {Path(j["source"]).name for j in autoedit_runs[0].jobs} == {"A.jpg", "B.jpg"}
    assert "inside the box" not in provider.calls[0][0]
    # the teammate logic is unchanged: teammates sharpened with the usual margin, everyone else protected
    assert all(job["strict_box"] is False for job in autoedit_runs[0].jobs)


def test_sharpen_entries_pick_the_teammates_and_protect_everyone_else(batch_album):
    bodies = album_module.album_for(batch_album).image("A.jpg").bodies
    entries = culling_app._batch_sharpen_entries(bodies)
    assert [bool(e.get("protect")) for e in entries] == [False, True]
    assert entries[0]["box"] == pytest.approx([0.1, 0.2, 0.25, 0.9])
    assert entries[1]["box"] == pytest.approx([0.3, 0.2, 0.45, 0.9])
    assert culling_app._batch_sharpen_entries(album_module.album_for(batch_album).image("B.jpg").bodies) == []


def test_candidates_are_the_photos_at_or_above_the_minimum_stars(batch_album):
    assert [im["file"] for im in culling_app._batch_candidates(batch_album, 5)] == ["A.jpg", "B.jpg"]
    assert [im["file"] for im in culling_app._batch_candidates(batch_album, 4)] == ["A.jpg", "B.jpg", "C.jpg"]
    assert len(culling_app._batch_candidates(batch_album, 1)) == 4


@pytest.mark.parametrize("min_stars, count", [(5, 2), (4, 3), (3, 3), (2, 4), (1, 4), (9, 2), (0, 4)])
def test_summary_counts_the_photos_that_will_be_processed(client, min_stars, count):
    assert client.get(f"/api/album-tools/batch-auto-edit/summary?minStars={min_stars}").json() == {"count": count}


@pytest.mark.parametrize("body, message", [
    ({"destination": " "}, "Destination folder is required"),
    ({"resize": "square"}, "Resize must be"),
    ({"resize": None, "sharpen": False}, "Turn on Resize"),
])
def test_start_rejects_an_incomplete_request(client, tmp_path, body, message):
    response = _start(client, tmp_path, **body)
    assert response.status_code == 400 and message in response.json()["detail"]
    assert not culling_app.batch_auto_edit_state.running


def test_start_rejects_a_file_as_the_destination(client, tmp_path):
    target = tmp_path / "a_file"
    target.write_text("x")
    response = _start(client, tmp_path, destination=str(target))
    assert response.status_code == 400 and "not a folder" in response.json()["detail"]


def test_start_needs_at_least_one_photo_at_the_minimum(client, tmp_path, monkeypatch):
    monkeypatch.setattr(culling_app, "_batch_candidates", lambda album_path, min_stars, player=None: [])
    response = _start(client, tmp_path)
    assert response.status_code == 400 and "No photos have 5+ stars" in response.json()["detail"]
    assert not culling_app.batch_auto_edit_state.running


def test_resize_needs_an_openai_key_but_sharpening_alone_does_not(client, tmp_path, monkeypatch, autoedit_runs):
    culling_app._resolve_team(None).openaiApiKey = ""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = _start(client, tmp_path)
    assert response.status_code == 400 and "OpenAI API key" in response.json()["detail"]
    assert _start(client, tmp_path, resize=None).status_code == 200
    assert _wait(client)["error"] is None


def test_start_is_refused_while_something_else_runs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(culling_app.processing_state, "running", True)
    assert _start(client, tmp_path).status_code == 409
    monkeypatch.setattr(culling_app.processing_state, "running", False)
    monkeypatch.setattr(culling_app.export_state, "running", True)
    assert _start(client, tmp_path).status_code == 409
    monkeypatch.setattr(culling_app.export_state, "running", False)
    assert culling_app.batch_auto_edit_state.claim()
    try:
        response = _start(client, tmp_path)
        assert response.status_code == 409 and "already running" in response.json()["detail"]
    finally:
        culling_app.batch_auto_edit_state.mark_stopped()


def test_a_batch_asks_the_llm_once_and_hands_autoedit_one_jobs_file(client, tmp_path, provider, autoedit_runs):
    out = tmp_path / "out"
    assert _start(client, tmp_path).status_code == 200
    status = _wait(client)

    assert status["error"] is None and status["done"] is True
    assert len(provider.calls) == 1  # both 5 star photos in a single request
    instructions, images = provider.calls[0]
    assert [label.split(",")[0] for label, _ in images] == ["Image 1 (A.jpg", "Image 2 (B.jpg"]
    assert "royal blue jerseys" in instructions  # the album's team colour

    run, = autoedit_runs  # one process for the whole batch, so the models load once
    assert run.cmd[1].endswith("autoedit.py") and "--face-enhance" in run.cmd
    by_name = {Path(job["source"]).name: job for job in run.jobs}
    assert set(by_name) == {"A.jpg", "B.jpg"}
    a, b = by_name["A.jpg"], by_name["B.jpg"]
    assert Path(a["out_file"]) == out / "A_ai_edited.jpg" and Path(b["out_file"]) == out / "B_ai_edited.jpg"
    # 4:5 of a 200 x 100 photo is the full height, 80 x 100 pixels, kept inside the photo
    assert a["crop"][2:] == [80, 100] and a["crop"][0] + a["crop"][2] <= 200
    assert b["crop"][2:] == [80, 100]
    assert [bool(e.get("protect")) for e in a["boxes"]] == [False, True]  # teammate sharpened, opponent protected
    assert b["boxes"] == []  # no teammate in B ...
    assert a["sharpen"] is True and b["sharpen"] is True  # ... so autoedit sharpens the whole frame instead

    assert status["result"] == {
        "processed": 2, "failed": 0, "skipped": 0, "cropped": 2, "sharpened": 2, "outputDir": str(out)}
    assert any("1 request(s) of up to 6" in line for line in status["lines"])


def test_the_original_ratio_crop_matches_the_photo(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: _answer(files, original=_box(0.1, 0.1, 0.8, 0.9))
    assert _start(client, tmp_path, resize="original", sharpen=False).status_code == 200
    assert _wait(client)["error"] is None
    run, = autoedit_runs
    assert "--face-enhance" not in run.cmd  # nothing to sharpen
    for job in run.jobs:
        _x, _y, w, h = job["crop"]
        assert w / h == pytest.approx(2.0, abs=0.03)  # the photo is 200 x 100
        assert job["boxes"] == [] and job["sharpen"] is False  # crop only


def test_a_lower_minimum_processes_more_photos(client, tmp_path, provider, autoedit_runs):
    assert _start(client, tmp_path, minStars=4, resize="instagram", sharpen=False).status_code == 200
    assert _wait(client)["result"]["processed"] == 3
    assert [len(images) for _, images in provider.calls] == [3]


def test_sharpening_alone_never_calls_the_llm_and_sharpens_every_photo(
    client, tmp_path, provider, autoedit_runs,
):
    assert _start(client, tmp_path, resize=None, sharpen=True).status_code == 200
    status = _wait(client)
    assert provider.calls == []
    run, = autoedit_runs
    by_name = {Path(job["source"]).name: job for job in run.jobs}
    assert set(by_name) == {"A.jpg", "B.jpg"}  # B has no teammate but is sharpened (whole frame) all the same
    assert by_name["A.jpg"]["boxes"] and by_name["B.jpg"]["boxes"] == []
    assert all(job["sharpen"] is True and job["crop"] is None for job in run.jobs)
    assert status["result"]["skipped"] == 0 and status["result"]["processed"] == 2
    assert status["result"]["sharpened"] == 2
    assert any("1 of 2 photo(s) have teammates to sharpen" in line and "sharpened whole" in line
               for line in status["lines"])


def test_without_sharpen_no_job_sharpens(client, tmp_path, provider, autoedit_runs):
    assert _start(client, tmp_path, resize="instagram", sharpen=False).status_code == 200
    _wait(client)
    assert all(job["sharpen"] is False and job["boxes"] == [] for job in autoedit_runs[0].jobs)


def test_a_photo_the_model_left_out_is_asked_about_again(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: _answer([f for f in files if not (len(files) > 1 and f == "B.jpg")])
    assert _start(client, tmp_path, sharpen=False).status_code == 200
    status = _wait(client)
    assert [len(images) for _, images in provider.calls] == [2, 1]  # the batch, then B alone
    assert {Path(j["source"]).name for j in autoedit_runs[0].jobs} == {"A.jpg", "B.jpg"}
    assert all(j["crop"] for j in autoedit_runs[0].jobs) and status["error"] is None


def test_a_photo_the_model_never_answers_for_is_still_sharpened(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: _answer([f for f in files if f != "B.jpg"])
    assert _start(client, tmp_path).status_code == 200
    status = _wait(client)
    by_name = {Path(j["source"]).name: j for j in autoedit_runs[0].jobs}
    assert by_name["A.jpg"]["crop"] and by_name["B.jpg"]["crop"] is None  # B: no crop, but sharpened whole
    assert by_name["B.jpg"]["sharpen"] is True
    assert status["result"]["skipped"] == 0 and status["error"] is None


def test_a_photo_the_model_never_answers_for_is_skipped_when_not_sharpening(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: _answer([f for f in files if f != "B.jpg"])
    assert _start(client, tmp_path, sharpen=False).status_code == 200
    status = _wait(client)
    assert [Path(j["source"]).name for j in autoedit_runs[0].jobs] == ["A.jpg"]  # B: no crop, nothing else to do
    assert status["result"]["skipped"] == 1 and status["error"] is None


def test_a_whole_frame_answer_still_writes_the_photo(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: _answer(files)  # original = (0, 0, 1, 1)
    assert _start(client, tmp_path, resize="original", sharpen=False).status_code == 200
    status = _wait(client)
    assert status["result"]["processed"] == 2 and status["result"]["cropped"] == 0
    assert all(job["crop"] is None for job in autoedit_runs[0].jobs)


def test_a_failed_llm_request_is_logged_and_the_run_carries_on(client, tmp_path, provider, autoedit_runs):
    def boom(instructions, images):
        raise RuntimeError("rate limited")

    provider.ask_json_images = boom
    assert _start(client, tmp_path).status_code == 200
    status = _wait(client)
    # no crops, but every photo is still sharpened (A its teammate, B the whole frame)
    assert {Path(j["source"]).name for j in autoedit_runs[0].jobs} == {"A.jpg", "B.jpg"}
    assert all(j["crop"] is None and j["sharpen"] is True for j in autoedit_runs[0].jobs)
    assert any("failed: rate limited" in line for line in status["lines"])
    assert status["result"]["processed"] == 2


def test_photos_that_autoedit_did_not_write_are_reported_failed(client, tmp_path, provider, monkeypatch):
    def fake(cmd, on_line=None, log_fn=None):
        jobs = json.loads(Path(cmd[cmd.index("--jobs-file") + 1]).read_text(encoding="utf-8"))
        Path(jobs[0]["out_file"]).write_bytes(b"jpeg")
        return 1

    monkeypatch.setattr(culling_app, "_stream_subprocess", fake)
    assert _start(client, tmp_path).status_code == 200
    status = _wait(client)
    assert status["result"]["processed"] == 1 and status["result"]["failed"] == 1
    assert "1 photo(s) could not be processed" in status["error"]


def test_nothing_to_process_is_an_error(client, tmp_path, provider, autoedit_runs):
    provider.answer = lambda files: json.dumps({"photos": []})
    assert _start(client, tmp_path, minStars=3, sharpen=False).status_code == 200
    status = _wait(client)
    assert "nothing to process" in status["error"] and autoedit_runs == []


def test_open_folder_needs_a_finished_run(client, tmp_path, provider, autoedit_runs, monkeypatch):
    assert client.post("/api/album-tools/batch-auto-edit/open-folder").status_code == 400
    opened = []
    monkeypatch.setattr(culling_app, "_open_in_file_explorer", lambda path: opened.append(path))
    assert _start(client, tmp_path).status_code == 200
    _wait(client)
    assert client.post("/api/album-tools/batch-auto-edit/open-folder").status_code == 200
    assert opened == [tmp_path / "out"]
