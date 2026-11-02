"""First-run flow: no team is created implicitly, and pages that need a team
send the user to the Team page until they have created one themselves."""

from __future__ import annotations

import json

import pytest

import culling_app

TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def team_json(tmp_path, monkeypatch):
    path = tmp_path / "team.json"
    monkeypatch.setattr(culling_app, "TEAM_JSON_PATH", path)
    return path


@pytest.fixture
def client(team_json):
    return TestClient(culling_app.app, follow_redirects=False)


def test_missing_team_json_means_no_teams_and_is_not_created(team_json):
    assert culling_app._load_teams_file().Teams == []
    assert not team_json.exists()


@pytest.mark.parametrize("content", ['{"Teams": [], "Global": {}}', "not json", ""])
def test_empty_or_unreadable_team_json_means_no_teams(team_json, content):
    team_json.write_text(content, encoding="utf-8")
    assert culling_app._load_teams_file().Teams == []
    assert team_json.read_text(encoding="utf-8") == content


@pytest.mark.parametrize("url", ["/", "/select-album", "/add-new-album"])
def test_pages_redirect_to_team_page_when_no_team(client, url):
    resp = client.get(url)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/team"


def test_team_page_is_reachable_without_a_team(client):
    resp = client.get("/team")
    assert resp.status_code == 200
    # Select Album is shown but disabled until a team exists.
    assert 'data-step="select-album"' in resp.text
    assert "Create a team first" in resp.text


def test_pages_render_once_a_team_exists(client):
    saved = client.post("/api/team", json={"name": "U10 Sharks"})
    assert saved.status_code == 200
    assert client.get("/select-album").status_code == 200
    assert client.get("/add-new-album").status_code == 200
    assert client.get("/").headers["location"] == "/select-album"
    assert 'href="/select-album"' in client.get("/team").text


def test_first_team_is_named_by_the_user_not_team_1(client, team_json):
    client.post("/api/team", json={"name": "U10 Sharks"})
    teams = json.loads(team_json.read_text(encoding="utf-8"))["Teams"]
    assert [t["name"] for t in teams] == ["U10 Sharks"]


@pytest.mark.parametrize("name", ["", "   "])
def test_saving_a_team_requires_a_name(client, team_json, name):
    resp = client.post("/api/team", json={"name": name})
    assert resp.status_code == 400
    assert not team_json.exists()


def test_start_processing_requires_a_team(client, tmp_path):
    resp = client.post("/api/start-processing", json={"path": str(tmp_path)})
    assert resp.status_code == 409
    assert "team" in resp.json()["detail"].lower()
    assert not culling_app.processing_state.running


def test_resolve_team_without_teams_is_a_clean_conflict(team_json):
    with pytest.raises(culling_app.HTTPException) as exc:
        culling_app._resolve_team("")
    assert exc.value.status_code == 409
