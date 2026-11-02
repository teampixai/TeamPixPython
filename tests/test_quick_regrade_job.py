"""Quick Regrade / Apply Jersey Colour run as background jobs whose log streams
to the page (/api/processing-output), instead of one blocking request that
only returns a summary at the end."""

from __future__ import annotations

import time

import pytest

import culling_app
from algo.album import Album
from test_extended_threshold import _write_regrade_album

TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def client(album_dir, monkeypatch):
    _write_regrade_album(album_dir)
    monkeypatch.setattr(culling_app, "_current_album_path", lambda: album_dir)
    # The jersey-colour endpoint looks the team up; the regrade itself does not.
    monkeypatch.setattr(culling_app, "_resolve_team", lambda team_id: None)
    monkeypatch.setattr(culling_app, "_album_team_id", lambda path: "team-1")
    state = culling_app.processing_state
    with state.lock:
        state.running, state.lines, state.return_code, state.result = False, [], None, None
    return TestClient(culling_app.app)


def _wait_for_job(client, timeout=30.0):
    """Polls /api/processing-output the way the page does, until the job stops."""
    lines, since, deadline = [], 0, time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/processing-output?since={since}").json()
        lines += data["lines"]
        since = data["next"]
        if not data["running"]:
            return lines, data
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_quick_regrade_streams_its_log_and_returns_the_summary(client, album_dir):
    started = client.post("/api/album-tools/regrade-sensitivity", json={
        "sensitivityMode": "custom", "sensitivityCustomValue": 0.5, "mode": "shallow",
    })
    assert started.status_code == 200 and started.json()["started"] is True

    lines, final = _wait_for_job(client)
    text = "\n".join(lines)
    assert "[regrade] sharpness threshold 0.50, extended threshold 0.20" in text
    assert "applying verdicts and teammate flags" in text
    assert "is_teammate False -> True" in text
    assert "teammates: 3 total (3 newly flagged, 0 no longer teammates" in text
    assert "setting changed - extended_threshold_offset: (not set) -> 0.3" in text
    assert "Regrade finished" in text
    assert final["returnCode"] == 0
    assert final["result"]["imagesConsidered"] == 1
    assert final["result"]["teamColor"] is not None
    assert [b.is_teammate for b in Album(album_dir).images["IMG_0001.jpg"].bodies] == [
        True, True, True, False, False, False,
    ]


def test_quick_regrade_blocks_a_second_job_while_running(client):
    assert culling_app.processing_state.claim()
    try:
        response = client.post("/api/album-tools/regrade-sensitivity", json={
            "sensitivityMode": "medium", "sensitivityCustomValue": 0.5, "mode": "shallow",
        })
        assert response.status_code == 409
    finally:
        culling_app.processing_state.finish(0)


def test_quick_regrade_failure_is_reported_not_lost(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(culling_app, "regrade_sensitivity", boom)
    client.post("/api/album-tools/regrade-sensitivity", json={
        "sensitivityMode": "medium", "sensitivityCustomValue": 0.5, "mode": "shallow",
    })
    lines, final = _wait_for_job(client)
    assert any("RuntimeError: disk on fire" in line for line in lines)
    assert final["returnCode"] == 1
    assert final["result"] is None
    assert culling_app.processing_state.running is False  # the slot is freed


def test_log_streaming_restores_the_loggers(client):
    import logging

    names = ("TeamPixPython", "CullingApp")
    before = [(logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names]
    client.post("/api/album-tools/regrade-sensitivity", json={
        "sensitivityMode": "medium", "sensitivityCustomValue": 0.5, "mode": "shallow",
    })
    _wait_for_job(client)
    after = [(logging.getLogger(n).level, list(logging.getLogger(n).handlers)) for n in names]
    assert after == before


def test_jersey_colour_apply_runs_as_a_streamed_job(client, album_dir):
    started = client.post("/api/album-tools/jersey-color", json={"teamColor": "", "rerunLlmCulling": False})
    assert started.status_code == 200 and started.json()["started"] is True
    lines, final = _wait_for_job(client)
    assert any("[regrade]" in line for line in lines)
    assert final["returnCode"] == 0
    assert final["result"]["pinned"] is False
    assert "llmRerun" not in final["result"]
