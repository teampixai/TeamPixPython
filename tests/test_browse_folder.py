"""/api/browse-folder: the Tk dialog must never leave Tk garbage for another thread to free."""
from __future__ import annotations

import gc
import threading
from pathlib import Path

import pytest

import culling_app

TestClient = pytest.importorskip("fastapi.testclient").TestClient
tk = pytest.importorskip("tkinter")
filedialog = pytest.importorskip("tkinter.filedialog")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(culling_app, "_load_state", lambda: {})
    return TestClient(culling_app.app)


def test_endpoint_passes_title_and_default_folder(client, monkeypatch):
    seen = []
    monkeypatch.setattr(culling_app, "_pick_folder_dialog", lambda t, d: seen.append((t, d)) or "C:\\Fotos\\Müller")
    resp = client.post("/api/browse-folder", json={"title": "Pick me"})
    assert resp.status_code == 200
    assert resp.json() == {"path": "C:\\Fotos\\Müller"}
    assert seen == [("Pick me", str(Path.home()))]


def test_endpoint_cancel_returns_null(client, monkeypatch):
    monkeypatch.setattr(culling_app, "_pick_folder_dialog", lambda t, d: None)
    assert client.post("/api/browse-folder", json={}).json() == {"path": None}


def _live_tk_roots() -> int:
    return sum(1 for o in gc.get_objects() if isinstance(o, tk.Tk))


def _run_in_worker(fn):
    """Run fn on a worker thread like FastAPI does; returns (result, error)."""
    out = {}

    def target():
        try:
            out["result"] = fn()
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    return out.get("result"), out.get("error")


@pytest.fixture
def tk_available():
    try:
        tk.Tk().destroy()
    except tk.TclError:
        pytest.skip("no display available for Tk")


def test_returns_chosen_path_and_leaves_no_tk_objects(tk_available, monkeypatch):
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kw: "C:/picked")
    result, err = _run_in_worker(lambda: culling_app._pick_folder_dialog("t", "C:/"))
    assert err is None
    assert result == "C:/picked"
    assert _live_tk_roots() == 0


def test_cancel_returns_none(tk_available, monkeypatch):
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kw: "")
    assert _run_in_worker(lambda: culling_app._pick_folder_dialog("t", "C:/")) == (None, None)


def test_dialog_failure_is_swallowed_and_cleaned_up(tk_available, monkeypatch):
    def boom(**kw):
        raise RuntimeError("dialog exploded")

    monkeypatch.setattr(filedialog, "askdirectory", boom)
    assert _run_in_worker(lambda: culling_app._pick_folder_dialog("t", "C:/")) == (None, None)
    assert _live_tk_roots() == 0


def test_tk_reference_cycles_are_freed_on_the_dialog_thread(tk_available, monkeypatch):
    """A cycle holding the Tk root used to be freed later by whichever thread ran the GC (aborting the
    process). The cycle must be gone before the dialog thread returns."""
    def leaky(**kw):
        cycle = [tk.StringVar(kw["parent"])]
        cycle.append(cycle)
        return "C:/picked"

    monkeypatch.setattr(filedialog, "askdirectory", leaky)
    gc.collect()
    result, err = _run_in_worker(lambda: culling_app._pick_folder_dialog("t", "C:/"))
    assert (result, err) == ("C:/picked", None)
    assert _live_tk_roots() == 0
    gc.collect()  # would abort the interpreter here if Tk garbage had been left behind


@pytest.mark.parametrize("enabled", [True, False])
def test_gc_state_is_restored(tk_available, monkeypatch, enabled):
    monkeypatch.setattr(filedialog, "askdirectory", lambda **kw: "C:/picked")
    was = gc.isenabled()
    try:
        (gc.enable if enabled else gc.disable)()
        _run_in_worker(lambda: culling_app._pick_folder_dialog("t", "C:/"))
        assert gc.isenabled() is enabled
    finally:
        (gc.enable if was else gc.disable)()
