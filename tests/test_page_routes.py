"""The old "Album Tools" page is split into "Export" and "Tools & More"."""

from __future__ import annotations

import re

import pytest

import culling_app

TestClient = pytest.importorskip("fastapi.testclient").TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(culling_app, "TEAM_JSON_PATH", tmp_path / "team.json")
    return TestClient(culling_app.app, follow_redirects=False)


def _titles(html: str) -> list[str]:
    """Panel and section headings, in page order."""
    raw = re.findall(r'<h3[^>]*>(.*?)</h3>', html, flags=re.S)
    return [re.sub(r"<[^>]+>", "", t).replace("&rarr;", "→").strip() for t in raw]


def test_stepper_lists_export_and_tools_instead_of_album_tools(client):
    html = client.get("/team").text
    stepper = html.split("</nav>", 1)[0]
    labels = re.findall(r'<span class="step-label">(.*?)</span>', stepper)
    assert labels == ["Select Album", "Culling", "Faces", "Export", "Tools &amp; More"]
    assert 'data-step="album-tools"' not in stepper


def test_old_album_tools_url_redirects_to_export(client):
    resp = client.get("/album-tools")
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/export"


def test_export_page_panels_in_order(client):
    resp = client.get("/export")
    assert resp.status_code == 200
    titles = _titles(resp.text)
    assert titles[1:] == ["Export to Folder", "Import More Images"]
    assert 'id="starBreakdown"' in resp.text
    assert "Export All Face Tagging (CSV)" in resp.text
    assert "js/export.js" in resp.text
    assert 'id="batchAutoEditPanel"' not in resp.text


def test_tools_page_sections_and_panels_in_order(client):
    resp = client.get("/tools")
    assert resp.status_code == 200
    assert _titles(resp.text) == [
        "Photo Utilities",
        "Batch Auto Edit (experimental)",
        "Per-Player Highlight Videos (experimental)",
        "Burst → MP4 (experimental)",
        "Album Maintenance",
        "Adjust Blur Sensitivity",
        "Change Jersey Color",
        "Re-run Face Detection",
    ]
    assert "js/tools.js" in resp.text
    assert 'id="exportPanel"' not in resp.text


@pytest.mark.parametrize("page,script", [("export", "export.js"), ("tools", "tools.js")])
def test_page_script_exists(page, script):
    assert (culling_app.REPO_ROOT / "webui" / "culling" / "static" / "js" / script).is_file()
