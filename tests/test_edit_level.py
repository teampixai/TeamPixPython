"""Edit window: Level (autoedit.py --rotate, the turned canvas the crop is in,
how a turned edit is recorded on the album entry) and Auto crop (the LLM
suggestion and its background job)."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import autoedit
import culling_app
from algo import album as album_module
from algo.album import album_for
from algo.llm import auto_crop
from conftest import make_entry, write_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient

KEY = "IMG_0001.jpg"


# ---- autoedit.py ------------------------------------------------------------ #

@pytest.mark.parametrize("degrees", [0, 5, -33.3, 90, 180, -90, -180, 179.5])
def test_rotate_image_fills_the_canvas_rotated_size_promises(degrees):
    img = np.full((60, 80, 3), 255, np.uint8)
    out = autoedit.rotate_image(img, degrees)
    assert (out.shape[1], out.shape[0]) == autoedit.rotated_size(80, 60, degrees)


def test_rotated_size_of_a_quarter_turn_swaps_the_sides():
    assert autoedit.rotated_size(200, 100, 90) == (100, 200)
    assert autoedit.rotated_size(200, 100, -90) == (100, 200)
    assert autoedit.rotated_size(200, 100, 180) == (200, 100)
    assert autoedit.rotated_size(200, 100, 0) == (200, 100)


def test_positive_degrees_turn_the_photo_counter_clockwise():
    img = np.zeros((60, 80, 3), np.uint8)
    img[:10, :10] = 255  # top-left marker
    ccw = autoedit.rotate_image(img, 90)
    assert ccw[-1, 0].tolist() == [255, 255, 255]  # ends bottom-left
    cw = autoedit.rotate_image(img, -90)
    assert cw[0, -1].tolist() == [255, 255, 255]   # ends top-right
    upside_down = autoedit.rotate_image(img, 180)
    assert upside_down[-1, -1].tolist() == [255, 255, 255]


def test_a_small_turn_leaves_black_corners_and_keeps_the_middle():
    img = np.full((100, 200, 3), 255, np.uint8)
    out = autoedit.rotate_image(img, 10)
    assert out.shape[0] > 100 and out.shape[1] > 200
    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[out.shape[0] // 2, out.shape[1] // 2].tolist() == [255, 255, 255]


def test_a_small_turn_moves_the_image_the_right_way():
    img = np.zeros((101, 201, 3), np.uint8)
    img[:, 190:] = 255  # a bright band along the right edge
    out = autoedit.rotate_image(img, 20)  # counter-clockwise: the right edge swings up
    cy, cx = out.shape[0] // 2, out.shape[1] // 2
    ys, xs = np.nonzero(out[..., 0] > 128)
    assert ys.mean() < cy and xs.mean() > cx


def test_no_turn_returns_the_image_itself():
    img = np.zeros((10, 20, 3), np.uint8)
    assert autoedit.rotate_image(img, 0) is img
    assert autoedit.rotate_image(img, float("nan")) is img


def test_process_one_turns_then_crops_in_the_turned_canvas(tmp_path):
    src = tmp_path / "in.jpg"
    assert cv2.imwrite(str(src), np.full((100, 200, 3), 128, np.uint8))
    out = tmp_path / "out.jpg"
    result = autoedit.process_one(
        src, [autoedit.TonePipeline()], tmp_path, "jpg", 95, None, None,
        out_file=out, tone=autoedit.ToneParams(), crop=(0, 0, 50, 150), rotate=90,
    )
    assert result == out
    # 200x100 turned by 90 degrees is 100x200; the crop is 50 wide, 150 tall.
    assert cv2.imread(str(out)).shape[:2] == (150, 50)


def test_cli_accepts_rotate():
    assert autoedit.ROTATE_LIMIT_DEGREES == 180.0
    assert autoedit.clamp_rotation(400) == 180.0
    assert autoedit.clamp_rotation(-400) == -180.0


# ---- server helpers --------------------------------------------------------- #

def test_clean_level_limits_and_rounds():
    assert culling_app._clean_level(0) == 0
    assert culling_app._clean_level(1.234) == 1.23
    assert culling_app._clean_level(500) == 180
    assert culling_app._clean_level(-500) == -180
    assert str(culling_app._clean_level(-0.001)) == "0.0"


def test_prepare_before_image_turns_and_crops_to_match(tmp_path):
    src = tmp_path / "a.jpg"
    assert cv2.imwrite(str(src), np.full((60, 80, 3), 200, np.uint8))
    plain = culling_app._prepare_before_image(src, tmp_path)
    assert plain == src  # served straight from disk

    turned = culling_app._prepare_before_image(src, tmp_path, None, 90)
    assert turned != src and cv2.imread(str(turned)).shape[:2] == (80, 60)

    cropped = culling_app._prepare_before_image(src, tmp_path, (0, 0, 30, 40), 90)
    assert cv2.imread(str(cropped)).shape[:2] == (40, 30)


# ---- album bookkeeping ------------------------------------------------------ #

def _make_source(root: Path, name: str = KEY, shape=(60, 80)) -> Path:
    fp = root / "src" / name
    fp.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((*shape, 3), np.uint8)
    img[:] = (60, 120, 180)
    assert cv2.imwrite(str(fp), img)
    return fp


def _with_rotated_edit(album_dir: Path, tmp_path: Path, rotated=True) -> Path:
    edited = album_dir / "editedimages" / "IMG_0001.jpg"
    entry = make_entry(KEY, file=str(_make_source(tmp_path)), edited_image="editedimages/IMG_0001.jpg")
    if rotated:
        entry["edited_rotated"] = True
    write_album(album_dir, [entry])
    edited.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(edited), np.zeros((30, 40, 3), np.uint8))
    album_module.invalidate_all()
    return edited


def test_edited_rotated_is_only_reported_while_the_edit_exists(album_dir, tmp_path):
    edited = _with_rotated_edit(album_dir, tmp_path)
    image = album_for(album_dir).image(KEY)
    assert image.edited_rotated is True
    edited.unlink()
    assert image.edited_rotated is False


def test_clear_edited_and_copies_forget_the_turn(album_dir, tmp_path):
    _with_rotated_edit(album_dir, tmp_path)
    album = album_for(album_dir)
    image = album.image(KEY)
    copy = image.copy_entry(file=Path("x.jpg"), key="x.jpg")
    assert "edited_rotated" not in copy

    image.clear_edited()
    album.save()
    assert album_for(album_dir).image(KEY).edited_rotated is False
    assert "edited_rotated" not in album_for(album_dir).image(KEY)._d


# ---- endpoints -------------------------------------------------------------- #

@pytest.fixture
def client(album_dir, monkeypatch):
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    yield TestClient(culling_app.app)
    culling_app.edit_session_state.clear()
    culling_app.ai_edit_state.clear()


def _jpeg_shape(content: bytes) -> tuple[int, int]:
    return cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR).shape[:2]


def test_preview_and_before_are_turned_by_the_level(album_dir, tmp_path, client):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    album_module.invalidate_all()
    session = client.post("/api/edit/ai-edit/session", json={"file": KEY}).json()
    assert (session["width"], session["height"]) == (80, 60)
    assert session["bodiesUnavailable"] is False

    preview = client.get("/api/edit/ai-edit/session/preview", params={"file": KEY})
    assert _jpeg_shape(preview.content) == (60, 80)
    turned = client.get("/api/edit/ai-edit/session/preview", params={"file": KEY, "level": 90})
    assert _jpeg_shape(turned.content) == (80, 60)

    assert _jpeg_shape(client.get("/api/edit/ai-edit/session/before", params={"file": KEY}).content) == (60, 80)
    before = client.get("/api/edit/ai-edit/session/before", params={"file": KEY, "level": -90})
    assert _jpeg_shape(before.content) == (80, 60)
    assert client.get("/api/edit/ai-edit/session/preview", params={"file": KEY, "level": "x"}).status_code == 422


def test_session_has_no_bodies_on_a_turned_edit(album_dir, tmp_path, client):
    _with_rotated_edit(album_dir, tmp_path)
    data = client.post("/api/edit/ai-edit/session", json={"file": KEY}).json()
    assert data["bodiesUnavailable"] is True and data["bodies"] == []


def test_apply_rejects_a_level_out_of_range(album_dir, tmp_path, client):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    album_module.invalidate_all()
    assert client.post("/api/edit/ai-edit/apply", json={"file": KEY, "level": 181}).status_code == 422


def test_accept_records_a_turned_edit_without_a_crop(album_dir, tmp_path, client):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    album_module.invalidate_all()
    pending = album_dir / "editedimages" / ".pending" / "IMG_0001.jpg"
    pending.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(pending), np.zeros((10, 10, 3), np.uint8))
    culling_app.ai_edit_state.set_pending(
        album_dir, KEY, pending, pending, album_dir / "editedimages" / "IMG_0001.jpg", rotated=True,
    )
    assert client.post("/api/edit/ai-edit/accept").status_code == 200
    image = album_for(album_dir).image(KEY)
    assert image.edited_rotated is True and image.edited_crop is None


def _run_edit(album_dir, tmp_path, monkeypatch, **request):
    """Runs _run_ai_edit with autoedit.py stubbed out; returns the command."""
    album_module.invalidate_all()
    source = tmp_path / "src" / KEY
    pending = album_dir / "editedimages" / ".pending" / "IMG_0001.jpg"
    commands = []

    def fake_run(cmd):
        commands.append(cmd)
        pending.write_bytes(b"jpg")
        return 0

    monkeypatch.setattr(culling_app.processing_state, "run_subprocess", fake_run)
    req = culling_app.AiEditApplyRequest(file=KEY, **request)
    culling_app._run_ai_edit(
        album_dir, KEY, source, pending, album_dir / "editedimages" / "IMG_0001.jpg", req, (80, 60))
    return commands[0]


def test_apply_passes_the_level_and_sizes_the_crop_to_the_turned_canvas(album_dir, tmp_path, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    cmd = _run_edit(album_dir, tmp_path, monkeypatch, level=90,
                    crop=culling_app.CropRect(x=0, y=10, w=999, h=40))
    assert cmd[cmd.index("--rotate") + 1] == "90"
    # The 80x60 photo turned 90 degrees is a 60x80 canvas: the crop is clamped to it.
    assert cmd[cmd.index("--crop") + 1] == "0,10,60,40"
    pending = culling_app.ai_edit_state.snapshot()
    assert pending["rotated"] is True and pending["crop"] is None and pending["crop_px"] is None


def test_apply_without_level_is_unchanged(album_dir, tmp_path, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    cmd = _run_edit(album_dir, tmp_path, monkeypatch, crop=culling_app.CropRect(x=8, y=6, w=40, h=30))
    assert "--rotate" not in cmd
    assert cmd[cmd.index("--crop") + 1] == "8,6,40,30"
    pending = culling_app.ai_edit_state.snapshot()
    assert pending["rotated"] is False
    assert pending["crop"] == pytest.approx((0.1, 0.1, 0.6, 0.6))


def test_a_crop_over_an_already_turned_edit_is_not_recorded_as_a_crop(album_dir, tmp_path, monkeypatch):
    _with_rotated_edit(album_dir, tmp_path)
    cmd = _run_edit(album_dir, tmp_path, monkeypatch, crop=culling_app.CropRect(x=0, y=0, w=20, h=15))
    assert "--rotate" not in cmd
    pending = culling_app.ai_edit_state.snapshot()
    assert pending["rotated"] is True and pending["crop"] is None


def test_sharpening_is_skipped_on_top_of_a_turned_edit(album_dir, tmp_path, monkeypatch):
    _with_rotated_edit(album_dir, tmp_path)
    cmd = _run_edit(album_dir, tmp_path, monkeypatch, sharpen=True, bodyIds=[0])
    assert cmd[cmd.index("-pipelines") + 1] == "tone"
    assert "--boxes-file" not in cmd


def test_sharpening_runs_on_the_unturned_photo_before_the_turn(album_dir, tmp_path, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    cmd = _run_edit(album_dir, tmp_path, monkeypatch, sharpen=True, bodyIds=[0], level=3.5)
    assert cmd[cmd.index("-pipelines") + 1] == "tone aisharpen"
    assert cmd[cmd.index("--rotate") + 1] == "3.5"
    boxes = json.loads((album_dir / "editedimages" / ".pending" / "boxes.json").read_text())
    assert boxes[KEY][0]["box"] == pytest.approx([0.1, 0.2, 0.3, 0.4])  # still the original frame


# ---- Auto crop: the suggestion ---------------------------------------------- #

def _answer(**over) -> str:
    data = {"level_degrees": -2.5, "crop": {"x1": 0.1, "y1": 0.2, "x2": 0.9, "y2": 0.8}, "reason": " Tilted. "}
    data.update(over)
    return json.dumps(data)


def test_parse_suggestion_reads_level_box_and_reason():
    s = auto_crop.parse_suggestion(_answer())
    assert s.level == -2.5 and s.box == (0.1, 0.2, 0.9, 0.8) and s.reason == "Tilted."


def test_parse_suggestion_clamps_and_orders_the_numbers():
    s = auto_crop.parse_suggestion(_answer(level_degrees=400, crop={"x1": 1.2, "y1": 0.9, "x2": -0.1, "y2": 0.1}))
    assert s.level == 180 and s.box == (0.0, 0.1, 1.0, 0.9)


@pytest.mark.parametrize("raw", [
    None, "", "not json", "[]", json.dumps({"level_degrees": 1}),
    _answer(crop={"x1": 0.1, "y1": 0.1, "x2": 0.12, "y2": 0.9}),
    _answer(level_degrees="tilted"),
    _answer(crop={"x1": 0, "y1": 0, "x2": 1}),
])
def test_parse_suggestion_rejects_unusable_answers(raw):
    with pytest.raises(ValueError):
        auto_crop.parse_suggestion(raw)


def test_parse_suggestion_keeps_the_level_check_and_what_was_removed():
    s = auto_crop.parse_suggestion(_answer(level_check=" Touchline rises 2 degrees. ", removed=" a bystander "))
    assert (s.level_check, s.removed) == ("Touchline rises 2 degrees.", "a bystander")
    assert auto_crop.parse_suggestion(_answer()).level_check == ""


def test_prompt_asks_for_a_level_evaluation_before_a_crop_that_removes_distractions():
    prompt = auto_crop.build_prompt()
    assert prompt.index("LEVEL") < prompt.index("CROP")
    assert "level_check" in prompt and "Do not default to 0" in prompt
    assert "remove what does not belong" in prompt and "bystanders" in prompt


def test_prompt_mentions_the_ratio_only_when_one_is_selected():
    assert "ratio" not in auto_crop.build_prompt()
    prompt = auto_crop.build_prompt(2034, 1398)
    assert "2034:1398" in prompt and "0.687" in prompt


def test_suggest_crop_asks_the_provider_and_returns_the_suggestion():
    calls = []

    class FakeProvider:
        model = "fake-model"

        def ask_json(self, instructions, image_b64):
            calls.append((instructions, image_b64))
            return _answer(), 1.25

        def get_cost_summary(self):
            return SimpleNamespace(total_cost_usd=0.0042)

    s = auto_crop.suggest_crop(FakeProvider(), b"\xff\xd8jpeg", ratio_h=5, ratio_w=4)
    assert (s.level, s.elapsed, s.cost_usd) == (-2.5, 1.25, 0.0042)
    (instructions, image_b64), = calls
    assert "5:4" in instructions
    assert base64.b64decode(image_b64) == b"\xff\xd8jpeg"


def test_the_provider_ask_json_uses_the_shared_request_path():
    from algo.llm.culling_provider import OpenAIProvider

    sent = []
    provider = object.__new__(OpenAIProvider)  # no openai SDK needed for this
    provider.model = "gpt-6-luna"
    provider._image_detail = None
    provider._chat_completion = lambda messages, tokens: (sent.append((messages, tokens)) or ('{"a": 1}', 0.5))

    assert provider.ask_json("Where?", "QUJD") == ('{"a": 1}', 0.5)
    (messages, tokens), = sent
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Where?"}
    assert content[1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD" and tokens > 0


# ---- Auto crop: the background job ------------------------------------------ #

def _wait_for_job(client, timeout=10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get("/api/edit/ai-edit/auto-crop/status").json()
        if not status["running"]:
            return status
        time.sleep(0.02)
    raise AssertionError("the auto crop job did not finish")


@pytest.fixture
def team(monkeypatch):
    holder = SimpleNamespace(openaiApiKey="sk-test")
    monkeypatch.setattr(culling_app, "_album_team_id", lambda album_dir: "team-1")
    monkeypatch.setattr(culling_app, "_resolve_team", lambda team_id: holder)
    return holder


def test_auto_crop_needs_an_openai_key(album_dir, tmp_path, client, team, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    team.openaiApiKey = ""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/api/edit/ai-edit/auto-crop", json={"file": KEY})
    assert response.status_code == 400 and "OpenAI API key" in response.json()["detail"]
    assert not culling_app.auto_crop_state.running


def test_auto_crop_job_returns_the_suggestion(album_dir, tmp_path, client, team, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path, shape=(600, 1600))))])
    album_module.invalidate_all()
    assert client.post("/api/edit/ai-edit/session", json={"file": KEY}).status_code == 200

    seen = {}

    def fake_provider(api_key):
        seen["api_key"] = api_key
        return SimpleNamespace(model="fake-model")

    def fake_suggest(provider, jpeg, **kwargs):
        seen.update(model=provider.model, shape=_jpeg_shape(jpeg), **kwargs)
        return auto_crop.AutoCropSuggestion(
            level=1.5, box=(0.1, 0.2, 0.9, 0.8), reason="Straighten.", elapsed=2.0, cost_usd=0.001)

    monkeypatch.setattr(auto_crop, "make_provider", fake_provider)
    monkeypatch.setattr(auto_crop, "suggest_crop", fake_suggest)
    response = client.post("/api/edit/ai-edit/auto-crop", json={"file": KEY, "ratioH": 5, "ratioW": 4})
    assert response.status_code == 200
    status = _wait_for_job(client)
    assert status["error"] is None and status["done"] is True
    assert status["result"] == {"level": 1.5, "box": [0.1, 0.2, 0.9, 0.8], "reason": "Straighten."}
    assert any("Straighten." in line for line in status["lines"])
    assert any("fake-model" in line for line in status["lines"])  # the log is written by the server's own code
    assert seen["api_key"] == "sk-test" and (seen["ratio_h"], seen["ratio_w"]) == (5, 4)
    # A small copy goes to the model, not the 1600-pixel preview.
    assert max(seen["shape"]) == auto_crop.AUTO_CROP_IMAGE_LONG_EDGE


def test_auto_crop_job_reports_a_failure(album_dir, tmp_path, client, team, monkeypatch):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    album_module.invalidate_all()
    assert client.post("/api/edit/ai-edit/session", json={"file": KEY}).status_code == 200

    def boom(*args, **kwargs):
        raise ValueError("The model's answer could not be read: nope")

    monkeypatch.setattr(auto_crop, "make_provider", lambda api_key: SimpleNamespace(model="fake-model"))
    monkeypatch.setattr(auto_crop, "suggest_crop", boom)
    assert client.post("/api/edit/ai-edit/auto-crop", json={"file": KEY}).status_code == 200
    status = _wait_for_job(client)
    assert "could not be read" in status["error"] and status["result"] is None
    assert client.post("/api/edit/ai-edit/auto-crop", json={"file": KEY}).status_code == 200  # can retry
    _wait_for_job(client)


def test_a_second_auto_crop_while_one_runs_is_refused(album_dir, tmp_path, client, team):
    write_album(album_dir, [make_entry(KEY, file=str(_make_source(tmp_path)))])
    assert culling_app.auto_crop_state.claim()
    try:
        assert client.post("/api/edit/ai-edit/auto-crop", json={"file": KEY}).status_code == 409
    finally:
        culling_app.auto_crop_state.mark_stopped()
