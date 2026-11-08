"""Players page: team face-DB curation (list, move, delete, profile, rebuild flush)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import culling_app as app

TEAM = "33333333-3333-4333-8333-333333333333"
OTHER = "44444444-4444-4444-8444-444444444444"


def _entry(crop: str) -> dict:
    return {"cropFileName": crop, "dtype": "float32", "shape": [2], "encoding": "base64", "value": "AAAAAAAAAAA="}


def _person(team_dir: Path, folder: str, name: str, pos=(), neg=(), **extra) -> Path:
    d = team_dir / folder
    (d / "Face").mkdir(parents=True)
    (d / "Negative").mkdir()
    for c in pos:
        (d / "Face" / c).write_bytes(b"img")
    for c in neg:
        (d / "Negative" / c).write_bytes(b"img")
    payload = {"name": name, "playernum": None, "provider": "facenet",
               "faces": [_entry(c) for c in pos], "negative_faces": [_entry(c) for c in neg], **extra}
    (d / "face.json").write_text(json.dumps(payload), encoding="utf-8")
    return d


def _face_json(d: Path) -> dict:
    return json.loads((d / "face.json").read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    team_json = repo / "team.json"
    team_json.write_text(json.dumps({"Teams": [
        {"id": TEAM, "name": "Sharks", "players": [{"name": "Ansel Chang", "number": "13"}, {"name": "Rookie", "number": ""}]},
        {"id": OTHER, "name": "Other", "players": []},
    ]}), encoding="utf-8")
    monkeypatch.setattr(app, "REPO_ROOT", repo)
    monkeypatch.setattr(app, "TEAM_JSON_PATH", team_json)
    monkeypatch.setattr(app, "_has_active_album", lambda: False)
    team_dir = repo / ".FaceReco" / TEAM
    team_dir.mkdir(parents=True)
    return team_dir


@pytest.fixture
def client():
    return TestClient(app.app)


def _player(client, name, team=TEAM):
    data = client.get(f"/api/players/data?teamId={team}").json()
    return next(p for p in data["players"] if p["name"] == name)


def _commit(client, name, *ops, team=TEAM):
    return client.post("/api/players/commit", json={"teamId": team, "name": name, "ops": list(ops)})


def test_data_merges_roster_aliases_and_db_only_people(env, client):
    _person(env, "Ansel Chang", "Ansel Chang", pos=["b.png", "a.png"])
    _person(env, "Ansel", "Ansel Chang", pos=["c.png"], neg=["n.png"])
    _person(env, "Coach Kim", "Coach Kim", pos=["k.png"])
    (env / "Ansel Chang" / "Face" / "Retired").mkdir()
    (env / "Ansel Chang" / "Face" / "Retired" / "old.png").write_bytes(b"x")
    (env / "Annotated.260101-000000").mkdir()

    data = client.get("/api/players/data").json()
    assert data["teamId"] == TEAM  # no active album -> first team
    by_name = {p["name"]: p for p in data["players"]}
    assert list(by_name) == ["Ansel Chang", "Coach Kim", "Rookie"]

    ansel = by_name["Ansel Chang"]
    assert ansel["number"] == "13" and ansel["inRoster"]
    # Primary folder first, face.json order kept, Retired/ excluded.
    assert [(i["folder"], i["crop"]) for i in ansel["positives"]] == [
        ("Ansel Chang", "b.png"), ("Ansel Chang", "a.png"), ("Ansel", "c.png")]
    assert ansel["negatives"] == [{"folder": "Ansel", "crop": "n.png"}]
    assert ansel["profile"] == {"folder": "Ansel Chang", "crop": "b.png"}

    assert not by_name["Coach Kim"]["inRoster"]
    assert by_name["Rookie"]["positives"] == [] and by_name["Rookie"]["profile"] is None


def test_move_carries_embedding_between_sets(env, client):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png", "b.png"], neg=["n.png"])
    r = _commit(client, "Ansel Chang",
                {"op": "move", "folder": "Ansel Chang", "crop": "a.png", "to": "Negative"},
                {"op": "move", "folder": "Ansel Chang", "crop": "n.png", "to": "Face"})
    assert r.status_code == 200, r.text
    assert (d / "Negative" / "a.png").is_file() and not (d / "Face" / "a.png").exists()
    assert (d / "Face" / "n.png").is_file()
    fj = _face_json(d)
    assert [f["cropFileName"] for f in fj["faces"]] == ["b.png", "n.png"]
    assert [f["cropFileName"] for f in fj["negative_faces"]] == ["a.png"]
    player = r.json()["player"]
    assert player["profile"]["crop"] == "b.png"  # a.png left the positives


def test_delete_moves_to_trash_and_tombstones(env, client):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png", "b.png"], neg=["n.png"])
    r = _commit(client, "Ansel Chang",
                {"op": "delete", "folder": "Ansel Chang", "crop": "a.png"},
                {"op": "delete", "folder": "Ansel Chang", "crop": "n.png"})
    assert r.status_code == 200, r.text
    assert (d / "Deleted" / "a.png").is_file() and (d / "Deleted" / "n.png").is_file()
    fj = _face_json(d)
    assert [f["cropFileName"] for f in fj["faces"]] == ["b.png"]
    assert fj["negative_faces"] == []
    assert sorted(fj[app.DELETED_CROPS_KEY]) == ["a.png", "n.png"]


def test_profile_set_and_fallbacks(env, client):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png", "b.png"])
    assert _commit(client, "Ansel Chang", {"op": "profile", "folder": "Ansel Chang", "crop": "b.png"}).status_code == 200
    assert _face_json(d)[app.PROFILE_CROP_KEY] == "b.png"
    assert _player(client, "Ansel Chang")["profile"]["crop"] == "b.png"

    # Deleting the profile picture falls back to the first positive.
    r = _commit(client, "Ansel Chang", {"op": "delete", "folder": "Ansel Chang", "crop": "b.png"})
    assert r.json()["player"]["profile"]["crop"] == "a.png"
    assert _face_json(d)[app.PROFILE_CROP_KEY] == "a.png"

    # No positives left -> no profile (secret-player icon in the UI).
    r = _commit(client, "Ansel Chang", {"op": "move", "folder": "Ansel Chang", "crop": "a.png", "to": "Negative"})
    assert r.json()["player"]["profile"] is None
    assert app.PROFILE_CROP_KEY not in _face_json(d)

    # A negative can't be the profile picture.
    r = _commit(client, "Ansel Chang", {"op": "profile", "folder": "Ansel Chang", "crop": "a.png"})
    assert r.status_code == 400


def test_commit_rejects_foreign_folders_and_traversal(env, client):
    _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png"])
    _person(env, "Coach Kim", "Coach Kim", pos=["k.png"])
    assert _commit(client, "Ansel Chang", {"op": "delete", "folder": "Coach Kim", "crop": "k.png"}).status_code == 400
    assert _commit(client, "Ansel Chang", {"op": "delete", "folder": "Ansel Chang", "crop": "..\\x"}).status_code == 400
    assert _commit(client, "Ansel Chang", {"op": "delete", "folder": "Ansel Chang", "crop": "zzz.png"}).status_code == 404
    assert _commit(client, "Nobody", {"op": "delete", "folder": "x", "crop": "y"}).status_code == 404
    assert (env / "Coach Kim" / "Face" / "k.png").is_file()

    assert client.get(f"/api/players/img/{TEAM}/Ansel Chang/Face/a.png").status_code == 200
    assert client.get(f"/api/players/img/{TEAM}/Ansel Chang/Retired/a.png").status_code == 404
    assert client.get(f"/api/players/img/{TEAM}/..%5C{TEAM}/Face/a.png").status_code == 400


def test_failed_op_still_persists_earlier_ops(env, client):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png", "b.png"])
    r = _commit(client, "Ansel Chang",
                {"op": "move", "folder": "Ansel Chang", "crop": "a.png", "to": "Negative"},
                {"op": "delete", "folder": "Ansel Chang", "crop": "missing.png"})
    assert r.status_code == 404
    # File and embedding stay consistent with each other.
    assert (d / "Negative" / "a.png").is_file()
    assert [f["cropFileName"] for f in _face_json(d)["negative_faces"]] == ["a.png"]


def test_commit_blocked_while_export_or_rebuild_runs(env, client):
    _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png"])
    op = {"op": "delete", "folder": "Ansel Chang", "crop": "a.png"}
    for state in (app.export_state, app.face_db_rebuild_state):
        assert state.claim()
        try:
            assert _commit(client, "Ansel Chang", op).status_code == 409
        finally:
            state.mark_stopped()


def test_export_sync_skips_curated_out_crops(env, tmp_path):
    album = tmp_path / "album"
    album.mkdir()
    (album / "album.json").write_text(json.dumps({"team_id": TEAM, "import_status": "complete", "results": []}), encoding="utf-8")
    cluster = album / ".FaceReco" / "Ansel Chang"
    (cluster / "Face").mkdir(parents=True)
    for c in ("del.png", "neg.png", "new.png"):
        (cluster / "Face" / c).write_bytes(b"img")
    (cluster / "face.json").write_text(json.dumps({"name": "Ansel Chang", "faces": [
        {"cropFileName": c, "origFilename": "p.jpg"} for c in ("del.png", "neg.png", "new.png")]}), encoding="utf-8")
    dest = _person(env, "Ansel Chang", "Ansel Chang", neg=["neg.png"], **{app.DELETED_CROPS_KEY: ["del.png"]})

    app._sync_named_clusters_to_system_db(album, {"p.jpg"})
    assert sorted(p.name for p in (dest / "Face").iterdir()) == ["new.png"]


def test_flush_empties_trash_and_annotated(env):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png"])
    (d / "Deleted").mkdir()
    (d / "Deleted" / "x.png").write_bytes(b"x")
    (env / "Annotated.260101-000000" / "Ansel Chang").mkdir(parents=True)
    lines: list[str] = []
    result = app._flush_team_face_db(env, lines.append)
    assert result == {"flushedFiles": 1, "flushedAnnotated": 1}
    assert not (d / "Deleted").exists() and not (env / "Annotated.260101-000000").exists()
    assert (d / "Face" / "a.png").is_file()


def test_rebuild_endpoint_runs_flush_then_script(env, client, monkeypatch):
    d = _person(env, "Ansel Chang", "Ansel Chang", pos=["a.png"])
    (d / "Deleted").mkdir()
    (d / "Deleted" / "x.png").write_bytes(b"x")
    calls: list[list[str]] = []

    def fake_stream(cmd, on_line=None, log_fn=None):
        calls.append(cmd)
        log_fn("rebuilt")
        return 0

    monkeypatch.setattr(app, "_stream_subprocess", fake_stream)
    monkeypatch.setattr(app.threading, "Thread", _InlineThread)
    r = client.post("/api/players/rebuild", json={"teamId": TEAM})
    assert r.status_code == 200, r.text
    status = client.get("/api/players/rebuild/status").json()
    assert status["done"] and not status["error"] and not status["running"]
    assert "rebuilt" in status["lines"]
    assert not (d / "Deleted").exists()
    assert calls and calls[0][-1] == str(env) and "--skip-calibration" not in calls[0]


def test_rebuild_rejected_during_export(env, client):
    assert app.export_state.claim()
    try:
        assert client.post("/api/players/rebuild", json={"teamId": TEAM}).status_code == 409
        assert not app.face_db_rebuild_state.running
    finally:
        app.export_state.mark_stopped()


def test_unknown_team_is_404(env):
    with pytest.raises(HTTPException) as exc:
        app._team_or_404("nope")
    assert exc.value.status_code == 404


class _InlineThread:
    def __init__(self, target, args=(), daemon=None):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)
