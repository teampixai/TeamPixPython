"""End-to-end photo culling & organization web UI.

Launches a local FastAPI server backing a jQuery/jQuery UI multi-page app —
each step of the workflow is its own template + own JS file (sharing one
Jinja2 base layout for the topbar/stepper look-and-feel), giving each page
natural code isolation instead of one giant single-page app:

  1. Team setup       — jersey colours, player roster, OpenAI key.
  2. Select album     — resume a previously processed album, or create a new
                         one (source folder, blur sensitivity, face
                         recognition) and run the detection pipeline.
  3. Culling          — sort blur / sharp / skipped.
  4. Faces            — group and label face clusters.
  5. Album Tools      — move files / write face-reco info, with progress.

This module currently implements the server-side pieces for pages 1 and 2
(team persistence + import-folder selection with "remember last" support).
Later iterations will add the processing/culling/faces/album-tools endpoints.

Usage::

    python culling_app.py [--host 127.0.0.1] [--port 8010]
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from algo.facereco import record_manual_override
from algo.regrade import regrade_sensitivity
from algo.stages.annotation import invalidate_previews, render_album_preview
from algo.album import ALBUM_JSON_NAME, EDITEDIMAGES_SUBDIR, AlbumImage, PersonRecord, album_for, box_in_crop
from algo.album_info import CATEGORIES as CULLING_CATEGORIES
from algo.album_info import INFO_JSON_NAME, AlbumInfo, category_for_status
from algo.utils import THUMBNAILS_SUBDIR, image_capture_timestamp
from algo.burst_video import DEFAULT_FPS, DEFAULT_MIN_FRAMES, render_album_bursts, render_player_videos, summarize_bursts

log = logging.getLogger("CullingApp")

REPO_ROOT = Path(__file__).resolve().parent
WEBUI_DIR = REPO_ROOT / "webui" / "culling"

TEAM_JSON_PATH = REPO_ROOT / "team.json"
STATE_JSON_PATH = REPO_ROOT / ".culling_state.json"
OUTPUT_DIR = REPO_ROOT / "albums"

# "high" recalibrated 2026-07-27 (0.70 -> 0.68) alongside the production
# sharpness-evaluator swap to WeightedGeometricMeanEvaluator (algo/sharpness.py)
# so "high" sensitivity keeps the same recall (~0.63) the old evaluator
# achieved at its old 0.70 threshold — see _setup_tmp/sharpness_eval/calibrate_high_threshold.py.
# Recalibrated again 2026-08-31 (low 0.35 -> 0.40, high 0.68 -> 0.62) alongside
# the swap to ContrastNormalizedEvaluator, on the same recall-preserving basis.
# Must stay in sync with 1_prep_review.py's SENSITIVITY_THRESHOLDS.
SENSITIVITY_PRESETS: dict[str, float] = {"low": 0.40, "medium": 0.50, "high": 0.62}

# Folder-name candidates for a face-DB dir, mirroring 1_prep_review.py.
FACE_DB_DIR_CANDIDATES: tuple[str, ...] = (".FaceReco", ".facereco", ".Facereco")


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
class JerseyColor(BaseModel):
    color: str
    forced: bool = False


class Player(BaseModel):
    name: str = ""
    number: str = ""


def _is_uuid(value: object) -> bool:
    """True for a canonical hyphenated UUID string."""
    if not isinstance(value, str):
        return False
    try:
        return str(uuid.UUID(value)) == value.lower()
    except ValueError:
        return False


class TeamData(BaseModel):
    # Stable identity (full UUID). `name` is display-only and may change; the
    # system face DB folder and every album's team_id are keyed by this id.
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    # Pre-UUID ids this team used to have; albums created back then still
    # store them in team_id, so lookups match these too.
    legacyIds: list[str] = Field(default_factory=list)
    jerseyColors: list[JerseyColor] = Field(default_factory=list)
    players: list[Player] = Field(default_factory=list)
    openaiApiKey: str = ""
    # ISO-8601 timestamp of the last time this team was used to start an
    # import, or None if it's never been used. Drives the "most recently
    # used first" ordering of the team picker on the Add Album page.
    lastUsedAt: Optional[str] = None


class GlobalSettings(BaseModel):
    """App-wide settings that aren't tied to any one team. Empty for now —
    reserved for future cross-team settings; `extra="allow"` so hand-added
    fields round-trip instead of being dropped on save."""
    model_config = ConfigDict(extra="allow")


class TeamsFile(BaseModel):
    """On-disk shape of team.json: a list of teams plus shared Global
    settings."""
    Teams: list[TeamData] = Field(default_factory=list)
    Global: GlobalSettings = Field(default_factory=GlobalSettings)


class ImportPathRequest(BaseModel):
    path: str


class StartProcessingRequest(BaseModel):
    path: str
    albumName: str = ""                   # optional custom name for the output album folder
    sensitivityMode: str = "medium"       # "low" | "medium" | "high" | "custom"
    sensitivityCustomValue: float = 0.50  # used only when mode == "custom", 0-0.99
    recognizeFaces: bool = True
    noTeam: bool = False                  # "Ignore jersey color" — bypasses team-colour filtering entirely (--noteam)
    teamColor: str = ""                   # pin a specific registered jersey color, or "" for auto-detect (--team-color)
    teamId: str = ""                      # which team's jersey colors/roster/API key to use


class BrowseFolderRequest(BaseModel):
    title: str = "Select photo folder to import"
    # "import" seeds the dialog from the *parent* of the last-used import
    # source folder; "export" seeds it from the *parent* of the last export
    # destination — either way, picking a sibling folder for the next
    # album is one click.
    context: str = "import"


class ExportRequest(BaseModel):
    destination: str
    exportFaceTagging: bool = True
    # Minimum star rating (1-5) an image must have to be exported — replaces
    # the old keep/drop flag as the export criterion. Defaults to 3 (the
    # baseline "keep" tier).
    minStars: int = 3


class BurstVideoRenderRequest(BaseModel):
    destination: str
    minFrames: int = DEFAULT_MIN_FRAMES
    mode: str = "fps"      # "fps" | "timestamp"
    fps: float = DEFAULT_FPS


class FaceTaggingCsvRequest(BaseModel):
    destination: str


class PlayerVideoRenderRequest(BaseModel):
    destination: str
    fps: float = DEFAULT_FPS


class RegradeSensitivityRequest(BaseModel):
    sensitivityMode: str = "medium"       # "low" | "medium" | "high" | "custom"
    sensitivityCustomValue: float = 0.50  # used only when mode == "custom", 0-0.99
    # "shallow" = re-threshold the stored sharpness scores (instant);
    # "deep" = re-run the full detection/scoring pipeline (background job).
    mode: str = "shallow"


class JerseyColorRequest(BaseModel):
    # "Hue:Shade" label pinning the team colour, or "" to restore
    # auto-detection from the photos.
    teamColor: str = ""
    # Burst membership depends on which photos end up "sharp", so a colour
    # change can invalidate the LLM's rankings/grades. Opt-in because
    # re-running spends OpenAI credits.
    rerunLlmCulling: bool = False


class AlbumSelectRequest(BaseModel):
    id: str  # album directory name, e.g. "20260719-030120-TestImages" — not a path


class AlbumDeleteRequest(BaseModel):
    id: str  # album directory name, e.g. "20260719-030120-TestImages" — not a path


class CullingApplyRequest(BaseModel):
    # {original filename: stars} for every image the user star-rated (hotkeys
    # 1-5, or the space-bar quick keep/drop toggle) this session, across every
    # star-filter view visited. Anything not present here keeps its current
    # effective (explicit-or-default) star rating. "keep" is always derived
    # server-side from the final star rating (3+ = keep) — there is no
    # separate keep/drop flag to track.
    starOverrides: dict[str, int] = Field(default_factory=dict)


class ImportMoreRequest(BaseModel):
    # Source folder of additional photos to merge into the current album.
    # Sensitivity/jersey-colour/engine/team are intentionally NOT accepted
    # here — 1_prep_review.py's merge-mode auto-locks those from the
    # album's own stored run_settings so a second import can never drift
    # from the settings the first import used.
    path: str


class AiEditRequest(BaseModel):
    # Album bookkeeping "key" (falls back to plain basename for older
    # albums) identifying which photo to edit — same identifier already used
    # by /api/original, /api/anno_img, etc.
    file: str


class CropRect(BaseModel):
    """A crop rectangle in pixels of the photo being edited (top-left x/y,
    width/height) — see AiEditApplyRequest.crop."""

    x: int
    y: int
    w: int
    h: int


class AiEditApplyRequest(BaseModel):
    """Everything the edit-mode panel collected; see api_ai_edit_apply()."""

    file: str
    # autoedit.ToneParams fields: EV per zone (-2..2) and contrast (-100..100).
    highlights: float = 0.0
    midtones: float = 0.0
    shadows: float = 0.0
    contrast: float = 0.0
    sharpen: bool = False
    # Indexes into the photo's bodies, as returned by /api/edit/ai-edit/session.
    bodyIds: list[int] = Field(default_factory=list)
    # The Crop tab's rectangle, in pixels of the editable source (the
    # session's width/height). None, or one covering the whole photo, means
    # no crop.
    crop: Optional[CropRect] = None

# --------------------------------------------------------------------------- #
# Processing (1_prep_review.py) — run in a background thread, buffer combined
# stdout/stderr lines so the browser can poll for them.
# --------------------------------------------------------------------------- #
class ProcessingState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.lines: list[str] = []
        self.running = False
        self.return_code: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        # Structured outcome of in-process jobs (e.g. regrade summaries), served
        # next to the log lines so the page can show more than "exit code 0".
        self.result: Optional[dict] = None

    def claim(self) -> bool:
        """Atomically claim the single shared processing slot (only one of
        start-processing/import-more/rerun-facereco/deep-regrade/jersey-
        colour LLM re-run/AI edit can run at a time); returns False if
        something is already running (callers turn that into an HTTP 409)."""
        with self.lock:
            if self.running:
                return False
            self.running = True
            self.return_code = None
            self.result = None
            self.lines = []
            return True

    def append(self, line: str) -> None:
        """Adds one line to the output the browser polls."""
        with self.lock:
            self.lines.append(line)

    def finish(self, return_code: int, result: Optional[dict] = None) -> None:
        """Ends an in-process job: records its outcome, THEN frees the slot,
        so the browser never sees "not running" without the result."""
        with self.lock:
            self.return_code = return_code
            self.result = result
            self.running = False

    def run_subprocess(self, cmd: list[str], cwd: Optional[Path] = None) -> int:
        """Runs *cmd*, streaming its combined stdout/stderr into self.lines
        (polled by /api/processing-output) and tracking self.process —
        shared by every _run_* function below that reuses this job slot.
        Returns the exit code; leaves self.running untouched so callers that
        need to act on the result before the browser sees "done" (e.g. AI
        edit publishing its pending-edit state) can do so first."""
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd or REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        with self.lock:
            self.process = proc
        assert proc.stdout is not None
        for line in proc.stdout:
            with self.lock:
                self.lines.append(line.rstrip("\n"))
        proc.wait()
        with self.lock:
            self.return_code = proc.returncode
            self.process = None
        return proc.returncode

    def mark_stopped(self) -> None:
        with self.lock:
            self.running = False


processing_state = ProcessingState()


# --------------------------------------------------------------------------- #
# AI edit (autoedit.py) — the freshly generated image is parked in
# editedimages/.pending/ until the user accepts or rejects it in the
# before/after compare view. Only one edit can be pending at a time (the
# job lock in processing_state already serialises the generation step).
# --------------------------------------------------------------------------- #
class AiEditState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.album_dir: Optional[Path] = None
        self.key: Optional[str] = None
        self.before_path: Optional[Path] = None
        self.pending_path: Optional[Path] = None
        self.final_path: Optional[Path] = None
        # The photo's edited_crop to record if this edit is accepted (see
        # AlbumImage.edited_crop) — any earlier crop composed with this run's.
        self.crop: Optional[tuple[float, float, float, float]] = None
        # The same crop in pixels of the original photo ({x, y, h, w}), which
        # is what gets written into the album entry as well.
        self.crop_px: Optional[dict] = None

    def set_pending(
        self,
        album_dir: Path,
        key: str,
        before_path: Path,
        pending_path: Path,
        final_path: Path,
        crop: Optional[tuple[float, float, float, float]] = None,
        crop_px: Optional[dict] = None,
    ) -> None:
        with self.lock:
            self.album_dir = album_dir
            self.key = key
            self.before_path = before_path
            self.pending_path = pending_path
            self.final_path = final_path
            self.crop = crop
            self.crop_px = crop_px

    def snapshot(self) -> Optional[dict]:
        with self.lock:
            if self.key is None or self.pending_path is None or not self.pending_path.is_file():
                return None
            return {
                "album_dir": self.album_dir,
                "key": self.key,
                "before_path": self.before_path,
                "pending_path": self.pending_path,
                "final_path": self.final_path,
                "crop": self.crop,
                "crop_px": self.crop_px,
            }

    def clear(self, delete_files: bool = False) -> None:
        """Forget the pending edit. With *delete_files*, also removes the
        generated image (and any rasterized "before" cache) from disk — used
        both by an explicit reject and to sweep up an orphan left behind by a
        server restart before starting a new edit."""
        with self.lock:
            stale = [self.pending_path, self.before_path]
            self.album_dir = None
            self.key = None
            self.before_path = None
            self.pending_path = None
            self.final_path = None
            self.crop = None
            self.crop_px = None
        if not delete_files:
            return
        for path in stale:
            if path is None:
                continue
            # Never touch the caller's real source photo: only files that
            # live inside the pending folder are ours to delete.
            if path.parent.name != _AI_EDIT_PENDING_SUBDIR:
                continue
            path.unlink(missing_ok=True)


ai_edit_state = AiEditState()


# --------------------------------------------------------------------------- #
# Edit mode session — the photo currently open in the edit-mode panel (before
# the user hits Apply). Holds a downscaled copy of the exact pixels
# autoedit.py will read, so live previews run autoedit.apply_tone() — the same
# code the final render uses — without re-decoding the file every time.
# --------------------------------------------------------------------------- #
EDIT_PREVIEW_LONG_EDGE = 1600
_EDIT_PREVIEW_JPEG_QUALITY = 88


class EditSessionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.album_dir: Optional[Path] = None
        self.key: Optional[str] = None
        self.base = None  # BGR uint8 numpy array, long edge <= EDIT_PREVIEW_LONG_EDGE
        self.before_jpeg: Optional[bytes] = None
        # Full-resolution (width, height) of the source — the pixel space the
        # Crop tab's rectangle is expressed in.
        self.size: Optional[tuple[int, int]] = None

    def open(self, album_dir: Path, key: str, base, before_jpeg: bytes,
             size: Optional[tuple[int, int]] = None) -> None:
        with self.lock:
            self.album_dir = album_dir
            self.key = key
            self.base = base
            self.before_jpeg = before_jpeg
            self.size = size

    def snapshot(self) -> Optional[dict]:
        with self.lock:
            if self.key is None:
                return None
            return {"album_dir": self.album_dir, "key": self.key, "base": self.base,
                    "before_jpeg": self.before_jpeg, "size": self.size}

    def clear(self) -> None:
        with self.lock:
            self.album_dir = None
            self.key = None
            self.base = None
            self.before_jpeg = None
            self.size = None


edit_session_state = EditSessionState()

# --------------------------------------------------------------------------- #
# BackgroundJobState — shared bookkeeping for a background job that writes
# log lines into a buffer the browser polls via a `/status?since=N` endpoint
# (export, burst-video, face-tagging CSV, per-player videos below). Each of
# these used to hand-roll the same lock/running/done/error/lines/result
# fields and the same claim-or-409/status-dict boilerplate; subclasses now
# only add whatever extra fields their feature needs.
# --------------------------------------------------------------------------- #
class BackgroundJobState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.running = False
        self.done = False
        self.error: Optional[str] = None
        self.lines: list[str] = []
        self.result: Optional[dict] = None

    def claim(self) -> bool:
        """Atomically start a fresh run, or return False if one is already
        in flight (callers turn that into an HTTP 409)."""
        with self.lock:
            if self.running:
                return False
            self.reset()
            self.running = True
            return True

    def log(self, msg: str) -> None:
        with self.lock:
            self.lines.append(msg)

    def mark_stopped(self) -> None:
        with self.lock:
            self.running = False

    def status(self, since: int) -> dict:
        with self.lock:
            new_lines = self.lines[since:]
            return {
                "lines": new_lines,
                "next": since + len(new_lines),
                "running": self.running,
                "done": self.done,
                "error": self.error,
                "result": self.result,
            }


# --------------------------------------------------------------------------- #
# Export (page 5 — Album Tools) — copy "keep" images and merge tagged-player faces
# into a .FaceReco database at a destination folder, run in a background
# thread with the same buffered-lines-for-polling pattern as ProcessingState.
# --------------------------------------------------------------------------- #
class ExportState(BackgroundJobState):
    def reset(self) -> None:
        super().reset()
        self.total_images = 0
        self.copied_images = 0
        self.total_players = 0
        self.processed_players = 0
        self.dest_dir: Optional[str] = None

    def status(self, since: int) -> dict:
        payload = super().status(since)
        with self.lock:
            payload.update({
                "totalImages": self.total_images,
                "copiedImages": self.copied_images,
                "totalPlayers": self.total_players,
                "processedPlayers": self.processed_players,
                "destDir": self.dest_dir,
            })
        return payload


export_state = ExportState()


# --------------------------------------------------------------------------- #
# Burst → MP4 (experimental, page 5 — Album Tools) — see algo/burst_video.py.
# Same buffered-lines-for-polling background-job pattern as ExportState; like
# PlayerVideoState/FaceTaggingCsvState, writes to a user-chosen destination
# folder (tracked here so "Open folder" knows where to point after a run).
# --------------------------------------------------------------------------- #
class BurstVideoState(BackgroundJobState):
    def reset(self) -> None:
        super().reset()
        self.output_dir: Optional[str] = None


burst_video_state = BurstVideoState()


# --------------------------------------------------------------------------- #
# Export all face tagging to CSV / per-player highlight MP4s (page 5 — Album
# Tools). Same buffered-lines-for-polling background-job pattern as
# BurstVideoState — both write to a user-chosen destination folder.
# --------------------------------------------------------------------------- #
class FaceTaggingCsvState(BackgroundJobState):
    def reset(self) -> None:
        super().reset()
        self.output_dir: Optional[str] = None


face_tagging_csv_state = FaceTaggingCsvState()


class PlayerVideoState(BackgroundJobState):
    def reset(self) -> None:
        super().reset()
        self.output_dir: Optional[str] = None


player_video_state = PlayerVideoState()


# --------------------------------------------------------------------------- #
# Face-DB rebuild (Players page) — flush deleted crops, then re-run
# RebuildFaceDB.py on one team's face DB. Same background-job pattern; never
# runs alongside an export (export also rebuilds the DB).
# --------------------------------------------------------------------------- #
class FaceDbRebuildState(BackgroundJobState):
    def reset(self) -> None:
        super().reset()
        self.team_id: Optional[str] = None


face_db_rebuild_state = FaceDbRebuildState()


# --------------------------------------------------------------------------- #
# Heartbeat watchdog — the browser pings /api/heartbeat periodically from
# every page (common.js); if none arrives within the timeout, the watchdog
# prints a reminder (with the relaunch link) instead of shutting the server
# down, since a closed/idle tab shouldn't kill an otherwise-healthy local
# server out from under a still-running job.
# --------------------------------------------------------------------------- #
# Seconds without a browser heartbeat (pages ping every 10s) before the
# watchdog prints the relaunch link. Override with --heartbeat-timeout;
# 0 disables the watchdog.
HEARTBEAT_TIMEOUT_SECONDS = 60.0


class _HeartbeatAccessLogFilter(logging.Filter):
    """Drop uvicorn access-log lines for the periodic /api/heartbeat pings."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "/api/heartbeat" not in record.getMessage()


class HeartbeatState:
    def __init__(self, timeout: float = HEARTBEAT_TIMEOUT_SECONDS) -> None:
        self.lock = threading.Lock()
        self.timeout = timeout
        self.last_heartbeat = time.time()
        self.url = ""
        self.notified = False


heartbeat_state = HeartbeatState()


def _heartbeat_watchdog() -> None:
    """Print a reminder (once) if no browser heartbeat arrives within the
    timeout — the server keeps running either way.

    Runs as a daemon thread. The timer starts from server launch, so the
    browser has one full timeout window to load the page and send its first
    heartbeat before the watchdog can fire. The reminder re-arms if a
    heartbeat resumes and then stops again later (e.g. the tab is reopened
    and closed a second time).
    """
    poll_interval = max(1.0, min(5.0, heartbeat_state.timeout / 3))
    last_poll = time.time()
    while True:
        time.sleep(poll_interval)
        now = time.time()
        poll_gap = now - last_poll
        last_poll = now
        if poll_gap > poll_interval * 3:
            # This watchdog thread itself missed multiple polls — wall-clock
            # time jumped far more than a sleeping thread should ever drift.
            # That only happens if the whole machine (this server process
            # included) was suspended, not just the browser tab/window being
            # closed — a closed tab doesn't affect our own thread's timing at
            # all. Since the browser suspends/resumes together with us on the
            # same machine, forgive this cycle instead of treating it as a
            # real timeout: it will send a fresh heartbeat shortly after
            # waking anyway.
            log.info(
                "Watchdog poll delayed %.0fs (expected ~%.0fs) — system likely "
                "resumed from sleep; resetting heartbeat timer.",
                poll_gap, poll_interval,
            )
            with heartbeat_state.lock:
                heartbeat_state.last_heartbeat = now
                heartbeat_state.notified = False
            continue
        with heartbeat_state.lock:
            elapsed = now - heartbeat_state.last_heartbeat
            already_notified = heartbeat_state.notified
            if elapsed > heartbeat_state.timeout and not already_notified:
                heartbeat_state.notified = True
            elif elapsed <= heartbeat_state.timeout:
                heartbeat_state.notified = False
        if elapsed > heartbeat_state.timeout and not already_notified:
            log.info("=" * 70)
            log.info(
                "Heartbeat stopped: no heartbeat received for %.0fs "
                "(threshold %.0fs); the browser tab appears to be closed.",
                elapsed, heartbeat_state.timeout,
            )
            log.info("The server is still running. To relaunch the app, click the URL:")
            log.info("  %s", heartbeat_state.url)
            log.info("(Press Ctrl+C in this window to exit.)")
            log.info("=" * 70)


def _sensitivity_arg(mode: str, custom_value: float) -> str:
    if mode == "custom":
        return str(custom_value)
    return mode


def _sensitivity_threshold(mode: str, custom_value: float) -> float:
    return SENSITIVITY_PRESETS.get(mode, custom_value)


def _jerseycolor_arg(team: TeamData) -> str:
    parts = []
    for jc in team.jerseyColors:
        prefix = "+" if jc.forced else ""
        parts.append(prefix + jc.color)
    return ";".join(parts)


def _run_processing(path: str, album_name: str, sensitivity_mode: str, sensitivity_custom_value: float, recognize_faces: bool, no_team: bool, team_color: str, team_id: str) -> None:
    team = _resolve_team(team_id)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_stem = _sanitize_player_dirname(album_name.strip()) if album_name.strip() else Path(path).stem
    output_dir = OUTPUT_DIR / f"{ts}-{folder_stem}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "1_prep_review.py"),
        path,
        "--sensitivity", _sensitivity_arg(sensitivity_mode, sensitivity_custom_value),
        "--output", str(output_dir),
    ]
    if not recognize_faces:
        cmd += ["--skip-facereco"]
    if no_team:
        cmd += ["--noteam"]
    else:
        jerseycolor = _jerseycolor_arg(team)
        if jerseycolor:
            cmd += ["--jerseycolor", jerseycolor]
        if team_color:
            cmd += ["--team-color", team_color]
    if team.id:
        cmd += ["--team-id", team.id]
    if team.openaiApiKey:
        cmd += ["--openaikey", team.openaiApiKey]

    log.info("Starting processing: %s", " ".join("***" if a == team.openaiApiKey else a for a in cmd))
    returncode = processing_state.run_subprocess(cmd)
    processing_state.mark_stopped()
    log.info("Processing finished (exit code %s)", returncode)

    if returncode == 0 and _is_album_complete(output_dir):
        state = _load_state()
        state["currentAlbum"] = str(output_dir.resolve())
        _save_state(state)
        log.info("Current album set from processing run: %s", output_dir.resolve())


def _album_team_id(album_dir: Path) -> str:
    """Best-effort read of the Album's team_id, so "import more" can look up
    the same team's OpenAI key without asking the user to re-select a team
    (sensitivity/jerseycolor/engine/noteam/team_id itself are already locked
    server-side by 1_prep_review.py's merge-mode — this is only needed
    because the API key itself is intentionally never persisted into the
    Album's run_settings)."""
    return album_for(album_dir).team_id


def _run_import_more(path: str, album_dir: Path) -> None:
    """Import additional photos into an EXISTING, already-processed album.

    Runs 1_prep_review.py with --output pointed at the existing album
    directory (not a fresh timestamped one) — its merge-mode (see main())
    detects the existing Album, skips already-imported source files,
    and auto-locks sensitivity/jerseycolor/engine/noteam/team-id/autoadjust
    to whatever the album's first import used, so none of those need to be
    passed here.
    """
    team = _resolve_team(_album_team_id(album_dir))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "1_prep_review.py"),
        path,
        "--output", str(album_dir),
    ]
    if not _album_has_facereco(album_dir):
        cmd += ["--skip-facereco"]
    if team.openaiApiKey:
        cmd += ["--openaikey", team.openaiApiKey]

    log.info("Starting import-more: %s", " ".join("***" if a == team.openaiApiKey else a for a in cmd))
    returncode = processing_state.run_subprocess(cmd)
    processing_state.mark_stopped()
    log.info("Import-more finished (exit code %s)", returncode)


def _run_rerun_facereco(album_dir: Path) -> None:
    """Re-run ONLY the face-recognition/clustering stage against an
    already-processed album, via 1_prep_review.py --rerun-facereco-only —
    reuses that album's own already-written Album (no re-detection of
    blur/bodies, no re-importing photos) and replays manual_overrides.json
    on top, so existing tagged players/deletions survive the recluster.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "1_prep_review.py"),
        "--output", str(album_dir),
        "--rerun-facereco-only",
    ]

    log.info("Starting face-detection re-run: %s", " ".join(cmd))
    returncode = processing_state.run_subprocess(cmd)
    processing_state.mark_stopped()
    log.info("Face-detection re-run finished (exit code %s)", returncode)


def _resolve_ai_edit_paths(album_dir: Path, key: str) -> tuple[Path, Path, Path]:
    """Resolve the (source, pending, final) paths for an AI-edit run on *key*.

    Chains onto the existing edited copy if one is already recorded and still
    on disk (that copy becomes the source — i.e. the "before" image the user
    compares against); otherwise starts from the original source file.
    autoedit.py always writes to the *pending* path, never over the source or
    the final destination, so nothing the user can already see is touched
    until they explicitly accept the result (see api_ai_edit_accept()).
    Raises HTTPException if neither the entry nor a readable source is found.
    """
    album = album_for(album_dir)
    image = album.image(key)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in album")

    stem = Path(key).stem
    edited_dir = album_dir / _EDITEDIMAGES_SUBDIR
    final = edited_dir / f"{stem}.jpg"
    pending = edited_dir / _AI_EDIT_PENDING_SUBDIR / f"{stem}.jpg"

    edited = image.edited_path
    if edited is not None:
        return edited, pending, final

    src = image.original_path
    if src is None or not src.is_file():
        raise HTTPException(status_code=404, detail="Source image not found on disk")
    return src, pending, final


def _prepare_before_image(
    source_path: Path, pending_dir: Path, crop_px: Optional[tuple[int, int, int, int]] = None
) -> Path:
    """Return a browser-displayable copy of *source_path* for the "before"
    pane of the compare view. JPEG/PNG/WebP sources are served straight from
    disk; anything else (RAW originals, TIFF) is rasterized to a JPEG cached
    next to the pending edit. With *crop_px* (x, y, w, h) the before image is
    cut to the same rectangle as the edit, so the two line up."""
    if crop_px is None and source_path.suffix.lower() in _WEB_DISPLAYABLE_EXTENSIONS:
        return source_path

    import cv2  # lazy: pulls in the heavy CV stack

    if crop_px is not None:
        import autoedit  # decodes exactly like the edit itself, so the crop lines up

        image = autoedit.read_image(source_path)
        if image is not None:
            image = autoedit.crop_image(image, crop_px)
    else:
        from algo.stages.image_analysis import _read_image  # lazy: pulls in the heavy CV stack

        image = _read_image(source_path)
    if image is None:
        raise RuntimeError(f"Could not decode {source_path}")
    out = pending_dir / f"{source_path.stem}__before.jpg"

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError(f"Could not encode a preview for {source_path}")
    out.write_bytes(buf.tobytes())
    return out


def _album_sharpness_threshold(run_settings: dict) -> float:
    """The album's face-sharpness threshold from its stored ``sensitivity``,
    which is either a preset name ("medium") or a number written by a
    regrade ("0.55"). Falls back to the medium preset if it is unreadable."""
    raw = str(run_settings.get("sensitivity", "medium"))
    threshold = SENSITIVITY_PRESETS.get(raw)
    if threshold is None:
        try:
            threshold = float(raw)
        except ValueError:
            threshold = SENSITIVITY_PRESETS["medium"]
    return threshold


NormBox = tuple[float, float, float, float]


def _compose_crop(outer: Optional[NormBox], inner: Optional[NormBox]) -> Optional[NormBox]:
    """*inner* (normalised to the frame *outer* cuts out of the original)
    expressed in the original's frame — i.e. what a photo cropped to *outer*
    and then to *inner* shows of the original. Either may be None (no crop)."""
    if inner is None:
        return outer
    if outer is None:
        return inner
    ox1, oy1, ox2, oy2 = outer
    ow, oh = ox2 - ox1, oy2 - oy1
    x1, y1, x2, y2 = inner
    return (ox1 + x1 * ow, oy1 + y1 * oh, ox1 + x2 * ow, oy1 + y2 * oh)


def _box_in_crop(box: NormBox, crop: Optional[NormBox]) -> Optional[NormBox]:
    """*box* (normalised to the original) re-expressed in the frame of a
    photo cropped to *crop*, clipped to it; None if nothing of it is left."""
    return box_in_crop(box, crop)


def _keypoints_in_crop(body, crop: Optional[NormBox]) -> list[list[float]]:
    """*body*'s keypoints as [x, y, confidence], in the frame of a photo
    cropped to *crop*; keypoints the crop cut off are dropped."""
    points = [[kp.point.x, kp.point.y, kp.confidence] for kp in body.body_keypoints]
    if crop is None:
        return points
    cx1, cy1, cx2, cy2 = crop
    cw, ch = cx2 - cx1, cy2 - cy1
    out = []
    for x, y, conf in points:
        nx, ny = (x - cx1) / cw, (y - cy1) / ch
        if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
            out.append([nx, ny, conf])
    return out


def _body_box(body, crop: Optional[NormBox]) -> Optional[list[float]]:
    """*body*'s box as [x1, y1, x2, y2] in the frame of a photo cropped to
    *crop* (see _box_in_crop), or None if it has no box or is cropped out."""
    box = body.body_bbox
    if box is None:
        return None
    mapped = _box_in_crop((box.x1, box.y1, box.x2, box.y2), crop)
    return list(mapped) if mapped is not None else None


def _edit_session_bodies(bodies, crop: Optional[NormBox] = None) -> list[dict]:
    """Bodies the edit-mode panel draws as selectable boxes: every body with
    a stored box, ``id`` being its index in *bodies* (what the panel sends
    back as bodyIds) and ``isTeammate`` whether the album marked it as
    wearing the team's jersey of the day (see algo/thresholds.py) — those
    start selected. Albums that predate the flag count as not teammates.

    Stored boxes are relative to the original photo; *crop* (the accepted
    edit's AlbumImage.edited_crop) maps them onto an already cropped edit,
    leaving out bodies the crop removed."""
    out = []
    for index, body in enumerate(bodies):
        box = _body_box(body, crop)
        if box is None:
            continue
        out.append({
            "id": index,
            "box": box,
            "isTeammate": bool(body.is_teammate),
        })
    return out


def _selected_body_entries(bodies, body_ids, crop: Optional[NormBox] = None) -> list[dict]:
    """Per-body payload for autoedit.py's --boxes-file: the normalised
    [x1, y1, x2, y2] box of each body (by index into *bodies*, see
    _edit_session_bodies) the user selected for sharpening; everyone else
    stays exactly as in the original. Unknown ids and bodies without a box
    (or cropped out by *crop*, see _edit_session_bodies) are skipped.

    Also includes each body's 17 COCO keypoints (normalised x, y,
    confidence; empty list if the album never stored them) — autoedit.py's
    sharpen_bodies() uses them to tell this player's own segmentation blob
    apart from a neighbour's when both land in the same padded crop (see its
    _select_body_component() and BODY_SHARPEN_* constants)."""
    entries = []
    for index in sorted(set(body_ids)):
        if not 0 <= index < len(bodies):
            continue
        body = bodies[index]
        box = _body_box(body, crop)
        if box is None:
            continue
        entries.append({
            "box": box,
            "keypoints": _keypoints_in_crop(body, crop),
        })
    return entries


def _protected_body_entries(bodies, body_ids, crop: Optional[NormBox] = None) -> list[dict]:
    """--boxes-file entries ("protect": true) for every boxed body the user
    did NOT select, so autoedit.py's sharpen_bodies() keeps them untouched
    even where they overlap a selected player (see its protected_norm)."""
    selected = set(body_ids)
    entries = []
    for index, body in enumerate(bodies):
        if index in selected:
            continue
        box = _body_box(body, crop)
        if box is None:
            continue
        entries.append({"box": box, "keypoints": _keypoints_in_crop(body, crop), "protect": True})
    return entries


def _crop_to_px(
    crop: NormBox, source_size: tuple[int, int], prior_crop: Optional[NormBox]
) -> dict:
    """*crop* (normalised to the original photo) as pixels of the original:
    ``{"x", "y", "h", "w"}``. *source_size* is the size of the file that was
    edited — an earlier cropped edit when *prior_crop* is set, so the
    original's own size is that divided by the fraction it kept."""
    width, height = source_size
    if prior_crop is not None:
        width = width / (prior_crop[2] - prior_crop[0])
        height = height / (prior_crop[3] - prior_crop[1])
    x1, y1 = round(crop[0] * width), round(crop[1] * height)
    x2, y2 = round(crop[2] * width), round(crop[3] * height)
    return {"x": x1, "y": y1, "h": y2 - y1, "w": x2 - x1}


def _clamp_crop_px(crop: Optional[CropRect], width: int, height: int) -> Optional[tuple[int, int, int, int]]:
    """The panel's crop rectangle clamped to a *width* x *height* source as
    (x, y, w, h), or None when there is nothing to crop (no rectangle, or
    one covering the whole photo)."""
    if crop is None or width <= 0 or height <= 0:
        return None
    x = min(max(0, crop.x), width - 1)
    y = min(max(0, crop.y), height - 1)
    w = min(max(1, crop.w), width - x)
    h = min(max(1, crop.h), height - y)
    if (x, y, w, h) == (0, 0, width, height):
        return None
    return (x, y, w, h)


def _run_ai_edit(
    album_dir: Path,
    key: str,
    source_path: Path,
    pending_path: Path,
    final_path: Path,
    req: AiEditApplyRequest,
    source_size: Optional[tuple[int, int]] = None,
) -> None:
    """Run autoedit.py with the edit-mode panel's settings (*req*) against a
    single photo, writing to *pending_path* (editedimages/.pending/<key
    stem>.jpg — RAW originals are rasterized to JPEG in the process): the
    "tone" pipeline with the exact brightness/contrast values the user
    previewed, then — if Sharpen players is on and at least one player is
    selected — body-only "aisharpen" (with GFPGAN face enhance) on just
    those players, and finally the Crop tab's rectangle (if any).
    *source_size* is the source's full-resolution (width, height) when the
    caller already knows it (the open edit session); otherwise it is read
    from the file when a crop needs it.

    Nothing is recorded on the Album entry here: the result stays
    parked in the pending folder until the user accepts it in the
    before/after compare view (api_ai_edit_accept) or discards it
    (api_ai_edit_reject).
    """
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop any previously pending edit (including an orphan left by a server
    # restart) so the folder only ever holds the run we are about to start.
    ai_edit_state.clear(delete_files=True)
    for stale in pending_path.parent.iterdir():
        if stale.is_file():
            stale.unlink(missing_ok=True)

    crop_px = None
    if req.crop is not None:
        if source_size is None:
            import autoedit  # lazy: pulls in the CV stack

            decoded = autoedit.read_image(source_path)
            if decoded is not None:
                source_size = (decoded.shape[1], decoded.shape[0])
            del decoded
        if source_size is not None:
            crop_px = _clamp_crop_px(req.crop, *source_size)
        else:
            log.warning("AI edit: could not read %s to size the crop; not cropping", source_path.name)

    try:
        before_path = _prepare_before_image(source_path, pending_path.parent, crop_px)
    except (RuntimeError, OSError, ImportError) as exc:
        log.warning("AI edit: could not prepare the 'before' image for %s: %s", key, exc)
        before_path = source_path

    image = album_for(album_dir).image(key)
    bodies = image.bodies if image is not None else []
    # Stored detections are relative to the original; when the source is an
    # already cropped edit they are mapped onto it.
    prior_crop = image.edited_crop if image is not None else None
    entries = _selected_body_entries(bodies, req.bodyIds, prior_crop) if req.sharpen else []

    new_crop = None
    if crop_px is not None and source_size is not None:
        x, y, w, h = crop_px
        width, height = source_size
        new_crop = (x / width, y / height, (x + w) / width, (y + h) / height)
    composed_crop = _compose_crop(prior_crop, new_crop)
    # An uncropped run on top of a cropped edit keeps that edit's recorded crop.
    composed_crop_px = None
    if new_crop is not None:
        composed_crop_px = _crop_to_px(composed_crop, source_size, prior_crop)
    elif image is not None:
        composed_crop_px = image.edited_crop_px

    pipelines = ["tone"] + (["aisharpen"] if entries else [])
    cmd = [
        sys.executable,
        str(REPO_ROOT / "autoedit.py"),
        str(source_path),
        "-pipelines", " ".join(pipelines),
        "--out-file", str(pending_path),
        "--highlights", f"{req.highlights:g}",
        "--midtones", f"{req.midtones:g}",
        "--shadows", f"{req.shadows:g}",
        "--contrast", f"{req.contrast:g}",
    ]
    if crop_px is not None:
        cmd += ["--crop", ",".join(str(v) for v in crop_px)]

    # Feeds autoedit.py's aisharpen pipeline the boxes (+ keypoints) of the
    # selected players (see _selected_body_entries) so it sharpens+blends
    # just those players instead of the whole frame, plus every unselected
    # player as "protect" so none of them is sharpened where they overlap a
    # selected one. Keyed by source_path.name (the exact file autoedit.py
    # reads) so it can look itself up without any other album context.
    if entries:
        boxes_file = pending_path.parent / "boxes.json"
        payload = entries + _protected_body_entries(bodies, req.bodyIds, prior_crop)
        boxes_file.write_text(json.dumps({source_path.name: payload}), encoding="utf-8")
        cmd += ["--face-enhance", "--boxes-file", str(boxes_file)]
        # Debug feature: "save_silhouette": true in .culling_state.json also
        # writes the original, silhouette mask, sharpened layer and final
        # image to <album>/debug/esrgan_silhouette/.
        if _load_state().get("save_silhouette") is True:
            cmd += ["--save-silhouette", str(album_dir / "debug" / "esrgan_silhouette")]

    log.info("Starting AI edit for %s: %s", key, " ".join(cmd))
    returncode = processing_state.run_subprocess(cmd)
    if returncode == 0 and pending_path.is_file():
        ai_edit_state.set_pending(
            album_dir=album_dir,
            key=key,
            before_path=before_path,
            pending_path=pending_path,
            final_path=final_path,
            crop=composed_crop,
            crop_px=composed_crop_px,
        )
    processing_state.mark_stopped()
    log.info("AI edit finished for %s (exit code %s)", key, returncode)


def _run_deep_regrade(album_dir: Path, sensitivity: str) -> None:
    """Deep regrade: re-run the full analysis pipeline (person/pose + face
    detection, sharpness scoring, jersey re-poll, preview regeneration) over
    the album's already-imported photos at a new sensitivity, via
    1_prep_review.py --regrade-only. Star ratings/keep flags/LLM culling
    results are preserved and FaceReco clusters are untouched.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "1_prep_review.py"),
        "--output", str(album_dir),
        "--regrade-only",
        "--sensitivity", sensitivity,
    ]

    log.info("Starting deep regrade: %s", " ".join(cmd))
    returncode = processing_state.run_subprocess(cmd)
    processing_state.mark_stopped()
    log.info("Deep regrade finished (exit code %s)", returncode)


def _llm_recull_command(album_dir: Path) -> tuple[list[str], Optional[str]]:
    """(command, openai key) for RerunLLMCulling.py against *album_dir*."""
    team = _resolve_team(_album_team_id(album_dir))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "RerunLLMCulling.py"),
        str(album_dir),
    ]
    if team.openaiApiKey:
        cmd += ["--openaikey", team.openaiApiKey]
    return cmd, team.openaiApiKey


def _run_rerun_llm_culling(album_dir: Path) -> None:
    """Re-rank the album's bursts with the LLM (RerunLLMCulling.py) after a
    change that moved photos between sharp and blurry -- burst membership,
    keep flags, grades and star tiers are all derived from the sharp set."""
    cmd, api_key = _llm_recull_command(album_dir)
    log.info("Starting LLM re-cull: %s", " ".join("***" if a == api_key else a for a in cmd))
    returncode = processing_state.run_subprocess(cmd)
    processing_state.mark_stopped()
    log.info("LLM re-cull finished (exit code %s)", returncode)


class _JobLogHandler(logging.Handler):
    """Copies log records emitted by ONE thread into the job output the
    browser polls, so an in-process job (the quick regrade) streams the same
    lines its logger already writes to the server console."""

    def __init__(self, thread_id: int) -> None:
        super().__init__(level=logging.INFO)
        self._thread_id = thread_id
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self._thread_id:
            processing_state.append(self.format(record))


@contextlib.contextmanager
def _stream_logs_to_job():
    """While active, INFO+ lines logged by the current thread under the app's
    loggers also appear in the polled job output. The loggers are lowered to
    INFO for the duration (uvicorn leaves them at WARNING), then restored."""
    handler = _JobLogHandler(threading.get_ident())
    loggers = [logging.getLogger(name) for name in ("TeamPixPython", "CullingApp")]
    saved = [lg.level for lg in loggers]
    for lg in loggers:
        lg.addHandler(handler)
        if lg.getEffectiveLevel() > logging.INFO:
            lg.setLevel(logging.INFO)
    try:
        yield
    finally:
        for lg, level in zip(loggers, saved):
            lg.removeHandler(handler)
            lg.setLevel(level)


def _regrade_result(summary) -> dict:
    """The fields the Album Tools page shows after a quick regrade."""
    return {
        "threshold": summary.threshold,
        "teamColor": summary.team_color,
        "pinned": summary.team_color_pinned,
        "imagesConsidered": summary.images_considered,
        "recovered": summary.recovered,
        "demoted": summary.demoted,
        "starsRebaselined": summary.stars_rebaselined,
        "jerseyRechecked": summary.jersey_rechecked,
        "jerseyRecheckUnreadable": summary.jersey_recheck_unreadable,
        "previewsInvalidated": summary.previews_invalidated,
        "teammates": summary.teammates,
        "teammatesAdded": summary.teammates_added,
        "teammatesRemoved": summary.teammates_removed,
    }


def _run_quick_regrade(
    album_dir: Path, threshold: float, team_color_override: Optional[str] = None, rerun_llm: bool = False,
) -> None:
    """Quick regrade (algo/regrade.py) as a background job on the shared slot,
    streaming its log into the polled output. Used by both "Quick Regrade"
    and "Apply Jersey Colour" (which passes *team_color_override* and may ask
    for an LLM burst re-rank afterwards, in the same job)."""
    return_code, result = 1, None
    try:
        with _stream_logs_to_job():
            summary = regrade_sensitivity(album_dir, threshold, team_color_override=team_color_override)
            log.info(
                "Regrade finished: %s (threshold=%.2f) — recovered=%d demoted=%d stars_rebaselined=%d team_colour=%s",
                album_dir.resolve(), threshold, summary.recovered, summary.demoted,
                summary.stars_rebaselined, summary.team_color,
            )
        result = _regrade_result(summary)
        return_code = 0
        if rerun_llm:
            processing_state.append("Re-running LLM burst ranking…")
            cmd, api_key = _llm_recull_command(album_dir)
            log.info("Starting LLM re-cull: %s", " ".join("***" if a == api_key else a for a in cmd))
            return_code = processing_state.run_subprocess(cmd)
            result["llmRerun"] = True
    except Exception as exc:  # noqa: BLE001 — surfaced to the page, never lost in a thread
        log.exception("Quick regrade failed for %s", album_dir)
        processing_state.append(f"Error: {type(exc).__name__}: {exc}")
    finally:
        processing_state.finish(return_code, result)


# --------------------------------------------------------------------------- #
# team.json persistence — {"Teams": [...], "Global": {...}}. Multiple teams
# are supported; each is identified by a stable `id` assigned the first time
# it's loaded/saved without one. QueryTeams (below) exposes every team's
# settings; the Team Setup page can create/update any team by id.u
# --------------------------------------------------------------------------- #
def _default_teams_file() -> TeamsFile:
    # No team is ever created implicitly: a first-time user must click "Add
    # new team" on the Team page, so every install goes through the same flow.
    return TeamsFile()


def _load_teams_file() -> TeamsFile:
    """Load team.json, transparently migrating the legacy flat single-team
    format (no "Teams" key — the whole file *was* one team's data) into
    {"Teams": [...], "Global": {...}}, and backfilling a stable `id`/`name`
    for any team that predates those fields (re-saved immediately so the
    generated id doesn't change on the next load). A missing, unreadable or
    empty file yields no teams (and is never written by a plain load)."""
    if not TEAM_JSON_PATH.is_file():
        return _default_teams_file()
    try:
        with open(TEAM_JSON_PATH, encoding="utf-8") as fh:
            payload = json.load(fh)
        if "Teams" not in payload:
            payload = {"Teams": [payload], "Global": {}}
        needs_resave = False
        # Defensive: coerce null player fields to "" so one malformed entry
        # (e.g. hand-edited JSON, or a future bulk-parse edge case) doesn't
        # discard the whole file back to defaults. Also backfill id/name.
        for i, team in enumerate(payload.get("Teams", [])):
            if not isinstance(team, dict):
                continue
            if not _is_uuid(team.get("id")):
                legacy = team.get("id")
                legacy_ids = team.get("legacyIds")
                if not isinstance(legacy_ids, list):
                    legacy_ids = []
                if legacy and legacy not in legacy_ids:
                    legacy_ids.append(legacy)
                team["legacyIds"] = legacy_ids
                team["id"] = str(uuid.uuid4())
                needs_resave = True
            if not team.get("name"):
                team["name"] = f"Team {i + 1}"
                needs_resave = True
            for player in team.get("players", []):
                if isinstance(player, dict):
                    player["name"] = player.get("name") or ""
                    player["number"] = player.get("number") or ""
        teams_file = TeamsFile.model_validate(payload)
        if needs_resave:
            _save_teams_file(teams_file)
        return teams_file
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        log.warning("Failed to load team.json (%s) — treating as no teams", exc)
        return _default_teams_file()


def _save_teams_file(teams_file: TeamsFile) -> None:
    with open(TEAM_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(teams_file.model_dump(), fh, indent=2)


def _find_team(teams_file: TeamsFile, team_id: str) -> Optional[TeamData]:
    for team in teams_file.Teams:
        if team.id == team_id or team_id in team.legacyIds:
            return team
    return None


def _canonical_team_id(team_id: str) -> str:
    """Map a possibly pre-UUID team id (as stored in an older album or
    state.json) to the team's current id; unknown ids pass through."""
    if not team_id:
        return ""
    team = _find_team(_load_teams_file(), team_id)
    return team.id if team is not None else team_id


def _has_teams() -> bool:
    return bool(_load_teams_file().Teams)


def _resolve_team(team_id: str) -> TeamData:
    """Resolve a team by id for use by the processing pipeline, falling back
    to the first team on disk if no id was given or it no longer exists."""
    teams_file = _load_teams_file()
    if not teams_file.Teams:
        raise HTTPException(status_code=409, detail="No team exists yet — create one on the Team page first.")
    if team_id:
        team = _find_team(teams_file, team_id)
        if team is not None:
            return team
        log.warning("Team id %r not found — falling back to the first team.", team_id)
    return teams_file.Teams[0]


def _touch_team_last_used(team_id: str) -> None:
    """Stamp `team_id`'s lastUsedAt with the current time, so it sorts to
    the front of the team picker next time. No-op if the team is unknown."""
    if not team_id:
        return
    teams_file = _load_teams_file()
    team = _find_team(teams_file, team_id)
    if team is None:
        return
    team.lastUsedAt = datetime.now(timezone.utc).isoformat()
    _save_teams_file(teams_file)


def _save_team(team: TeamData) -> TeamData:
    """Create (if `team.id` is blank or unrecognized) or update (if it
    matches an existing team) a single team, preserving every other team /
    Global settings already on disk. Returns the saved team (with its id
    filled in if it was newly created)."""
    teams_file = _load_teams_file()
    if not team.id:
        team.id = str(uuid.uuid4())
    for i, existing in enumerate(teams_file.Teams):
        if existing.id == team.id or team.id in existing.legacyIds:
            # Server-owned: never let a client drop or replace the real id.
            team.id = existing.id
            team.legacyIds = existing.legacyIds
            teams_file.Teams[i] = team
            break
    else:
        teams_file.Teams.append(team)
    _save_teams_file(teams_file)
    log.info("team.json saved (team %r: %d player(s), %d jersey colour(s))",
              team.name, len(team.players), len(team.jerseyColors))
    return team


# --------------------------------------------------------------------------- #
# App state persistence (last import path, etc.) — kept separate from
# team.json since it isn't "team" data.
# --------------------------------------------------------------------------- #
def _load_state() -> dict:
    if not STATE_JSON_PATH.is_file():
        return {"lastImportPath": "", "currentAlbum": "", "selectedTeamId": ""}
    try:
        with open(STATE_JSON_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {"lastImportPath": "", "currentAlbum": "", "selectedTeamId": ""}


def _save_state(state: dict) -> None:
    with open(STATE_JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


# --------------------------------------------------------------------------- #
# Album discovery — list/select previously processed albums/ folders so the
# user can resume review/tagging without re-running 1_prep_review.py.
# --------------------------------------------------------------------------- #
def _album_has_facereco(path: Path) -> bool:
    return any((path / name).is_dir() for name in FACE_DB_DIR_CANDIDATES)


def _is_album_complete(path: Path) -> bool:
    return _album_data(path)[0]


# How many of an album's highest-ranked sharp photos the polaroid preview /
# hover slideshow cycles through (see _build_album_summary).
PREVIEW_IMAGE_COUNT = 10


# Matches the "<yyyymmdd-hhmmss>-" prefix _run_processing() stamps onto every
# output folder name (see `output_dir = OUTPUT_DIR / f"{ts}-{folder_stem}"`).
_ALBUM_DIR_TIMESTAMP_PREFIX_RE = re.compile(r"^\d{8}-\d{6}-")


def _album_display_name(dir_name: str) -> str:
    """Human-friendly album name for UI display: the directory name with its
    leading run-timestamp prefix stripped (that timestamp is already shown
    separately as `createdDisplay`). Falls back to the full directory name
    if it doesn't match the expected `1_prep_review.py` naming scheme."""
    return _ALBUM_DIR_TIMESTAMP_PREFIX_RE.sub("", dir_name) or dir_name


def _format_created(timestamp: str) -> str:
    """"20260719-030759" -> "Jul 19, 2026 03:07 AM"; falls back to the raw
    string (or "") if it doesn't match the expected 1_prep_review.py format."""
    if not timestamp:
        return ""
    try:
        return datetime.strptime(timestamp, "%Y%m%d-%H%M%S").strftime("%b %d, %Y %I:%M %p")
    except ValueError:
        return timestamp


def _build_album_summary(path: Path) -> dict:
    """Computes the /select-album summary from scratch — parses album.json
    (via `album_for`) and info.json. Expensive on a large album (album.json
    routinely reaches tens of MB, see algo/album.py's docstring); callers
    should go through `_read_album_summary`'s persisted cache instead of
    calling this directly."""
    summary: dict = {
        "name": path.name,  # also serves as the album's id — never the absolute path
        "displayName": _album_display_name(path.name),  # human-friendly, for UI display only
        "srcDir": "",
        "timestamp": "",
        "createdDisplay": "",
        "ourJerseyColor": None,
        "sharpCount": 0,
        "blurCount": 0,
        "skippedCount": 0,
        "hasFaceReco": _album_has_facereco(path),
        "previewImages": [],
    }
    info = AlbumInfo(path)
    if not info.exists:
        return summary
    summary["srcDir"] = info.src_dir
    summary["timestamp"] = info.timestamp
    summary["createdDisplay"] = _format_created(info.timestamp)
    summary["ourJerseyColor"] = info.our_jersey_color
    summary["sharpCount"] = info.count("sharp")
    summary["blurCount"] = info.count("blur")
    summary["skippedCount"] = info.count("skipped")
    album_images = album_for(path).images
    # The album's best photos: highest star rating first (the explicit rating,
    # else the sharp default the Culling page uses), then the LLM grade, then
    # sharpness. Deterministic, so the card shows the same strong shots every
    # visit, and cheap to serve: they are plain cached thumbnails.
    ranked = []
    for entry in info.entries("sharp"):
        image = album_images.get(entry.key)
        if image and image.preview_path:
            stars = image.stars if image.stars is not None else _CULLING_DEFAULT_STARS["sharp"]
            ranked.append((stars, image.llm_grade or 0.0, image.sharpness_score or 0.0, image.key))
    ranked.sort(reverse=True)
    summary["previewImages"] = [key for *_rest, key in ranked[:PREVIEW_IMAGE_COUNT]]
    return summary


# Persisted per-album cache for `_read_album_summary`/`_is_album_complete` —
# select-album lists every album directory on every visit, and computing a
# summary means parsing the WHOLE album.json (tens of MB, see
# algo/album.py's docstring) just to read a handful of counts/previews. The
# in-process `album_for()` LRU (capped at 4 albums) doesn't help once there
# are more albums than that, so this cache is a small JSON file written next
# to album.json/info.json, invalidated by THEIR mtime+size (same stat-based
# freshness idiom as `Album.refresh()`) rather than by any TTL — any writer
# (a 1_prep_review.py subprocess, regrade, LLM culling, ...) already changes
# album.json's mtime/size on save, so this needs no cache-busting of its own.
# A CODE change to _build_album_summary's output shape is a separate concern
# (album.json/info.json wouldn't change) — bump _ALBUM_SUMMARY_CACHE_VERSION
# whenever that shape changes so stale on-disk caches are rejected instead of
# served with missing/renamed fields.
_ALBUM_SUMMARY_CACHE_NAME = ".summary_cache.json"
_ALBUM_SUMMARY_CACHE_VERSION = 2  # 2: previewImages = the top-ranked photos, no longer a random sample


def _album_summary_cache_path(path: Path) -> Path:
    return path / _ALBUM_SUMMARY_CACHE_NAME


def _album_file_stamp(path: Path, name: str) -> Optional[list]:
    try:
        st = (path / name).stat()
    except OSError:
        return None
    return [st.st_mtime, st.st_size]


def _read_album_summary_cache(path: Path) -> Optional[dict]:
    """The cached ``{"complete":, "summary":}`` dict, or None on a miss —
    missing/corrupt cache file, or album.json/info.json changed since it was
    written."""
    try:
        with open(_album_summary_cache_path(path), encoding="utf-8") as fh:
            cached = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if cached.get("version") != _ALBUM_SUMMARY_CACHE_VERSION:
        return None
    if cached.get("albumStamp") != _album_file_stamp(path, ALBUM_JSON_NAME):
        return None
    if cached.get("infoStamp") != _album_file_stamp(path, INFO_JSON_NAME):
        return None
    return cached


def _write_album_summary_cache(path: Path, complete: bool, summary: dict) -> None:
    cache = {
        "version": _ALBUM_SUMMARY_CACHE_VERSION,
        "albumStamp": _album_file_stamp(path, ALBUM_JSON_NAME),
        "infoStamp": _album_file_stamp(path, INFO_JSON_NAME),
        "complete": complete,
        "summary": summary,
    }
    try:
        with open(_album_summary_cache_path(path), "w", encoding="utf-8") as fh:
            json.dump(cache, fh)
    except OSError:
        pass  # best-effort; a failed write just means the next call recomputes


def _album_data(path: Path) -> tuple:
    """``(is_complete, summary)`` for *path*, served from the persisted
    cache above when it's still fresh."""
    cached = _read_album_summary_cache(path)
    if cached is not None:
        return cached["complete"], cached["summary"]
    complete = album_for(path).is_complete
    summary = _build_album_summary(path)
    if _album_file_stamp(path, ALBUM_JSON_NAME) is not None:
        _write_album_summary_cache(path, complete, summary)  # skip stray dirs with no album.json
    return complete, summary


def _read_album_summary(path: Path) -> dict:
    return _album_data(path)[1]


def _list_albums() -> list[dict]:
    if not OUTPUT_DIR.is_dir():
        return []
    albums = [
        summary
        for child in OUTPUT_DIR.iterdir()
        if child.is_dir()
        for complete, summary in [_album_data(child)]
        if complete
    ]
    albums.sort(key=lambda a: a["name"], reverse=True)
    return albums


# --------------------------------------------------------------------------- #
# Face clustering — assign/delete review of the current album's .FaceReco
# clusters, native to this workflow (own template + own JS).
# --------------------------------------------------------------------------- #
FACE_SUBDIR = "Face"
FACE_ANNOTATED_SUBDIR = "Face.annotated"
NEGATIVE_SUBDIR = "Negative"
# Players page: per-person recycle bin (emptied by "Rebuild face DB"), the
# face.json key naming the profile-picture crop, and the face.json key that
# remembers deleted crop names so export never re-syncs them.
DELETED_SUBDIR = "Deleted"
PROFILE_CROP_KEY = "profileCrop"
DELETED_CROPS_KEY = "deletedCrops"

# Matching tolerance for body_bbox floats when writing back to the Album.
BBOX_EPS = 1e-6


def _album_dir_by_id(album_id: str) -> Path:
    """Resolve an album's directory name (id) to its path under OUTPUT_DIR —
    _safe_component rejects separators/".." so this can only ever land on a
    direct child of OUTPUT_DIR, never an arbitrary path on disk."""
    path = OUTPUT_DIR / _safe_component(album_id)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail="Not a valid album directory.")
    return path


def _current_album_path() -> Path:
    state = _load_state()
    album_path_str = state.get("currentAlbum", "")
    if not album_path_str:
        raise HTTPException(status_code=400, detail="No album selected.")
    path = Path(album_path_str)
    if not path.is_dir() or not _is_album_complete(path):
        raise HTTPException(status_code=400, detail="Album not found or not fully processed.")
    return path


def _has_active_album() -> bool:
    """True when the current session has a fully-processed album selected --
    used by base.html (via the ``has_active_album()`` Jinja global below) to
    decide whether the Culling/Faces/Summary nav steps should be enabled."""
    state = _load_state()
    album_path_str = state.get("currentAlbum", "")
    if not album_path_str:
        return False
    path = Path(album_path_str)
    return path.is_dir() and _is_album_complete(path)


# --------------------------------------------------------------------------- #
# Culling (page 3) — filter images by effective star rating (0 = skipped,
# 1-5 = graded; independently selectable, not a threshold), grouped into
# time-based "bursts". Decisions are staged client-side and only committed
# to the Album (as an explicit "keep" boolean per entry) when the user hits
# Apply — see api_culling_apply().
# --------------------------------------------------------------------------- #
# Categories and the info.json buckets behind them now live in
# algo/album_info.py (imported above as CULLING_CATEGORIES).
# Where autoedit.py's output for a photo is kept — see api_ai_edit_apply()/
# _run_ai_edit() below. Sibling of previews/ under the album directory.
_EDITEDIMAGES_SUBDIR = EDITEDIMAGES_SUBDIR
# Freshly generated edits land here first and are only promoted into
# _EDITEDIMAGES_SUBDIR once the user accepts them in the compare view.
_AI_EDIT_PENDING_SUBDIR = ".pending"
# Formats a browser can render directly; anything else (RAW, TIFF) has to be
# rasterized before it can be shown in the before/after compare view.
_WEB_DISPLAYABLE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
# Effective star rating when the Album has no explicit "stars" value yet
# (older albums, or the LLM culling stage never ran): sharp images default to
# a baseline "keep" of 3, blur images to 1 (the worst tier a genuinely-graded
# photo can land on). Skipped images (no person detected — never graded at
# all) default to a distinct 0, so they're never conflated with a photo that
# WAS scored and simply came out blurry.
_CULLING_DEFAULT_STARS = {"blur": 1, "sharp": 3, "skipped": 0}

# Consecutive photos within this many seconds of each other are treated as
# the same "burst" for the nav-pane grouping.
BURST_GAP_SECONDS = 1.0


def _culling_images(
    album_path: Path, category: str, with_timestamp: bool = True,
    backfilled: Optional[list] = None,
) -> list[dict]:
    """Per-image metadata for one review category: album key, source path,
    effective star rating, effective keep state (derived from the star
    rating — 3+ = keep — so it can never drift out of sync with the
    star-driven color scheme), and capture timestamp. Preview
    and thumbnail images are fetched by key (see /api/anno_img and
    /api/culling/thumb), so no filenames are sent to the client.

    The timestamp (only burst grouping/sorting needs it) comes from the
    album's cached ``capture_ts``; a photo imported before that was recorded
    gets it read from the source once, set on the album, and appended to
    *backfilled* so the caller can persist it. ``with_timestamp=False``
    skips all of that (``timestamp`` is None)."""
    info = AlbumInfo(album_path)
    src_dir = Path(info.src_dir)
    album_images = album_for(album_path).images
    default_stars = _CULLING_DEFAULT_STARS[category]

    images = []
    for entry in info.entries(category):
        src_name = entry.key
        if not src_name:
            continue
        image = album_images.get(src_name)
        preview = image.preview_path if image else None
        if preview is None:
            continue
        stars = image.stars
        stars = stars if stars is not None else default_stars
        keep = stars >= 3
        burst_ranking = image.burst_ranking
        llm_grade = image.llm_grade
        # Prefer the per-entry absolute source path (multi-source-directory
        # imports) over joining the album's single legacy SrcDir.
        src_path = entry.source_path or src_dir / src_name
        timestamp = None
        if with_timestamp:
            timestamp = image.capture_ts
            if timestamp is None:
                timestamp = image.capture_ts = image_capture_timestamp(src_path)
                if backfilled is not None:
                    backfilled.append(image)
        images.append({
            "file": src_name,
            "path": str(src_path),
            "keep": keep,
            "stars": stars,
            "burstRanking": burst_ranking,
            "llmGrade": llm_grade,
            "timestamp": timestamp,
        })
    return images


def _kept_image_basenames(album_path: Path, min_stars: int = 3) -> set[str]:
    """Basenames of every image at or above *min_stars* — the same effective
    (explicit-or-default) star rating the Culling page shows. Used by the
    Album Tools/export step to decide which original photos (and which face crops)
    to copy to the destination."""
    kept: set[str] = set()
    for category in CULLING_CATEGORIES:
        for image in _culling_images(album_path, category, with_timestamp=False):
            if image["stars"] >= min_stars:
                kept.add(image["file"])
    return kept


def _all_culling_images(album_path: Path, with_timestamp: bool = True) -> list[dict]:
    """Every reviewable image (blur + sharp + skipped combined) with its
    effective star rating — the single pool the Culling page's star filter
    is applied against. Capture times computed for photos that had none
    cached are persisted to album.json (once, synchronously) before
    returning."""
    images = []
    backfilled: list = []
    for category in CULLING_CATEGORIES:
        images.extend(_culling_images(album_path, category, with_timestamp, backfilled))
    if backfilled:
        album_for(album_path).save()
        log.info("Cached capture time for %d photo(s) in %s", len(backfilled), album_path.name)
    return images


def _parse_star_filter(raw: str) -> set[int]:
    """Parses the Culling page's star-filter query param: a comma-separated
    list of star levels (0-5, independently selectable — not a >=/<=
    threshold). Raises 400 on anything empty or out of range."""
    try:
        stars = {int(s) for s in raw.split(",") if s.strip() != ""}
    except ValueError:
        stars = set()
    if not stars or not stars.issubset(set(range(6))):
        raise HTTPException(status_code=400, detail="Invalid stars filter")
    return stars


def _group_bursts(images: list[dict]) -> list[list[dict]]:
    """Group images into "bursts": a new group starts whenever the gap since
    the chronologically-previous image exceeds BURST_GAP_SECONDS."""
    ordered = sorted(images, key=lambda im: (im["timestamp"], im["file"]))
    groups: list[list[dict]] = []
    for im in ordered:
        if groups and im["timestamp"] - groups[-1][-1]["timestamp"] <= BURST_GAP_SECONDS:
            groups[-1].append(im)
        else:
            groups.append([im])
    return groups


def _sort_groups(groups: list[list[dict]], sort_mode: str) -> list[list[dict]]:
    """Order in which burst groups are listed in the nav pane. Images within
    a group are always chronological, regardless of this setting."""
    if sort_mode == "old":
        return sorted(groups, key=lambda g: g[0]["timestamp"])
    if sort_mode == "new":
        return sorted(groups, key=lambda g: g[0]["timestamp"], reverse=True)
    return sorted(groups, key=lambda g: (-len(g), g[0]["timestamp"]))  # "size" (default)


_RATING_GROUP_SIZE = 6


def _rating_sort_key(im: dict) -> tuple:
    """Stars desc, then llmGrade desc (missing sorts lowest), then timestamp
    asc."""
    llm_grade = im["llmGrade"]
    grade_key = (0, -llm_grade) if llm_grade is not None else (1, 0.0)
    return (-im["stars"], grade_key, im["timestamp"])


def _rating_groups(images: list[dict]) -> list[list[dict]]:
    """"Rating" sort mode: ignore burst/time-gap grouping entirely — sort the
    full flat image list by rating, then chunk sequentially into fixed
    groups of _RATING_GROUP_SIZE (last group may be smaller)."""
    ordered = sorted(images, key=_rating_sort_key)
    return [ordered[i:i + _RATING_GROUP_SIZE] for i in range(0, len(ordered), _RATING_GROUP_SIZE)]


def _facereco_dir(album_path: Path) -> Optional[Path]:
    for name in FACE_DB_DIR_CANDIDATES:
        candidate = album_path / name
        if candidate.is_dir():
            return candidate
    return None


SYSTEM_FACE_DB_NAME = ".FaceReco"


def _system_face_db_root() -> Path:
    """Root of the cross-album face DB (repo-root .FaceReco); holds one
    sub-folder per team id, each of which is a self-contained face DB."""
    for name in FACE_DB_DIR_CANDIDATES:
        candidate = REPO_ROOT / name
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / SYSTEM_FACE_DB_NAME


def _team_face_db_dir(team_id: str) -> Path:
    """Face DB folder for one team: <repo>/.FaceReco/<team uuid>/. The same
    player on two teams is two independent entries, one per team folder."""
    return _system_face_db_root() / _resolve_team(team_id).id


def _safe_component(name: str) -> str:
    """Reject a path component containing separators or traversal."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid path component")
    return name


_INVALID_FS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_player_dirname(name: str) -> str:
    """Make a free-text player name safe to use as a destination folder
    name (Windows/POSIX invalid characters replaced, trailing dots/spaces
    trimmed — Windows rejects both)."""
    cleaned = _INVALID_FS_CHARS.sub("_", name).strip(" .")
    return cleaned or "Unnamed"


def _is_pending_cluster(folder_name: str) -> bool:
    return folder_name.isdigit()


def _load_face_json(cluster_dir: Path) -> dict:
    fp = cluster_dir / "face.json"
    if not fp.is_file():
        return {"name": "", "playernum": None, "cluster": cluster_dir.name,
                "provider": None, "aligned": None, "faces": []}
    with open(fp, encoding="utf-8") as fh:
        return json.load(fh)


def _write_face_json(cluster_dir: Path, payload: dict) -> None:
    with open(cluster_dir / "face.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _build_clusters(album_path: Path) -> list[dict]:
    fr = _facereco_dir(album_path)
    if fr is None:
        return []

    source_index = album_for(album_path).source_index
    pending: list[dict] = []
    matched: list[dict] = []
    for child in sorted(fr.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        payload = _load_face_json(child)
        faces = []
        for face in payload.get("faces", []):
            crop = face.get("cropFileName")
            if not crop:
                continue
            orig_filename = face.get("origFilename", "")
            faces.append({
                "crop": crop,
                "origFilename": orig_filename,
                "originalPath": source_index.get(orig_filename, ""),
            })
        cluster = {
            "id": child.name,
            "pending": _is_pending_cluster(child.name),
            "name": payload.get("name") or "",
            "playernum": payload.get("playernum"),
            "faces": faces,
        }
        (pending if cluster["pending"] else matched).append(cluster)

    # Pending (numeric) clusters first, then matched (named) clusters.
    return pending + matched


def _collect_names(album_path: Path) -> list[str]:
    """Collect cluster-page autocomplete names, limited to the album's own
    team: its roster (team.json), the names already tagged in this album's
    .FaceReco, and the names in that team's system face DB folder (which is
    where tagged non-roster people such as coaches end up). Other teams'
    folders are never read."""
    names: set[str] = set()
    team_id = _album_team_id(album_path)
    team = _resolve_team(team_id)
    for player in team.players:
        name = (player.name or "").strip()
        if name:
            names.add(name)
    for source in (_team_face_db_dir(team_id), _facereco_dir(album_path)):
        if source is None or not source.is_dir():
            continue
        for child in source.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            name = (_load_face_json(child).get("name") or "").strip()
            if name:
                names.add(name)
    return sorted(names, key=str.casefold)


# --------------------------------------------------------------------------- #
# Players page — curate one team's face DB (.FaceReco/<team id>/<Person>/).
# A person is every folder whose face.json carries the same name (alias
# folders like "Ansel" → "Ansel Chang" are merged), plus roster players who
# have no folder yet. Images are addressed as (folder, set, crop). Moving or
# deleting a crop moves its face.json embedding with it, so matching picks the
# change up immediately without a rebuild. The profile picture lives in the
# primary folder's face.json (PROFILE_CROP_KEY) and falls back to the first
# positive crop.
# --------------------------------------------------------------------------- #
PLAYER_IMAGE_SETS: dict[str, str] = {FACE_SUBDIR: "faces", NEGATIVE_SUBDIR: "negative_faces"}
_PLAYER_CROP_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
players_lock = threading.Lock()


class PlayerOp(BaseModel):
    op: str                   # "move" | "delete" | "profile"
    folder: str
    crop: str
    to: Optional[str] = None  # "move" only: "Face" | "Negative"


class PlayersCommitRequest(BaseModel):
    teamId: str
    name: str
    ops: list[PlayerOp] = Field(default_factory=list)


class PlayersRebuildRequest(BaseModel):
    teamId: str


def _team_or_404(team_id: str) -> TeamData:
    team = _find_team(_load_teams_file(), team_id) if team_id else None
    if team is None:
        raise HTTPException(status_code=404, detail="Unknown team.")
    return team


def _default_players_team(teams_file: TeamsFile) -> TeamData:
    """The current album's team when one is selected, else the first team."""
    if _has_active_album():
        try:
            team = _find_team(teams_file, _album_team_id(_current_album_path()))
        except Exception:  # noqa: BLE001 — any album problem just means "no default"
            team = None
        if team is not None:
            return team
    return teams_file.Teams[0]


def _team_face_db_path(team: TeamData) -> Path:
    return _system_face_db_root() / team.id


def _is_person_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and not path.name.startswith((".", "Annotated."))
        and (path / "face.json").is_file()
    )


def _list_crops(set_dir: Path, entries: list[dict]) -> list[str]:
    """Crop files on disk in *set_dir* (sub-folders such as Face/Retired/
    excluded), in face.json order first, then by name."""
    if not set_dir.is_dir():
        return []
    order = {f.get("cropFileName"): i for i, f in enumerate(entries) if f.get("cropFileName")}
    names = [p.name for p in set_dir.iterdir() if p.is_file() and p.suffix.lower() in _PLAYER_CROP_EXTS]
    return sorted(names, key=lambda n: (order.get(n, len(order)), n.casefold()))


def _team_people(team: TeamData) -> list[dict]:
    """Every person of *team*: {name, number, inRoster, folders}, where
    folders is [(dir, face.json payload)] with the primary folder first."""
    team_dir = _team_face_db_path(team)
    groups: dict[str, dict] = {}

    def group(name: str) -> dict:
        return groups.setdefault(name.casefold(), {"name": name, "number": "", "inRoster": False, "folders": []})

    if team_dir.is_dir():
        for d in sorted(team_dir.iterdir(), key=lambda p: p.name.casefold()):
            if not _is_person_dir(d):
                continue
            try:
                payload = _load_face_json(d)
            except ValueError as exc:
                log.warning("Players: unreadable %s/face.json skipped: %s", d.name, exc)
                continue
            g = group(str(payload.get("name") or "").strip() or d.name)
            g["folders"].append((d, payload))
            if not g["number"] and payload.get("playernum") not in (None, ""):
                g["number"] = str(payload["playernum"])
    for player in team.players:
        name = (player.name or "").strip()
        if not name:
            continue
        g = group(name)
        g["name"], g["inRoster"] = name, True
        if player.number:
            g["number"] = str(player.number)
    for g in groups.values():
        primary = _sanitize_player_dirname(g["name"])
        g["folders"].sort(key=lambda fp: fp[0].name != primary)
    return sorted(groups.values(), key=lambda g: g["name"].casefold())


def _find_person(team: TeamData, name: str) -> dict:
    key = (name or "").strip().casefold()
    for person in _team_people(team):
        if person["name"].casefold() == key:
            return person
    raise HTTPException(status_code=404, detail=f"Player {name!r} not found.")


def _player_sets(person: dict) -> tuple[list[dict], list[dict]]:
    positives: list[dict] = []
    negatives: list[dict] = []
    for d, payload in person["folders"]:
        for image_set, key, bucket in ((FACE_SUBDIR, "faces", positives), (NEGATIVE_SUBDIR, "negative_faces", negatives)):
            bucket.extend({"folder": d.name, "crop": c} for c in _list_crops(d / image_set, payload.get(key) or []))
    return positives, negatives


def _resolve_profile(person: dict, positives: list[dict]) -> Optional[dict]:
    """The chosen profile crop while it's still a positive, else the first
    positive, else None (the UI shows the secret-player icon)."""
    if not positives:
        return None
    wanted = person["folders"][0][1].get(PROFILE_CROP_KEY) if person["folders"] else None
    if wanted:
        for image in positives:
            if image["crop"] == wanted:
                return image
    return positives[0]


def _player_view(person: dict) -> dict:
    positives, negatives = _player_sets(person)
    return {
        "name": person["name"],
        "number": person["number"],
        "inRoster": person["inRoster"],
        "profile": _resolve_profile(person, positives),
        "positives": positives,
        "negatives": negatives,
    }


def _unique_dest(directory: Path, name: str) -> Path:
    dest = directory / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 1
    while dest.exists():
        dest = directory / f"{stem}_{n}{suffix}"
        n += 1
    return dest


def _pop_face_set_entry(payload: dict, key: str, crop: str) -> Optional[dict]:
    entries = payload.setdefault(key, [])
    for i, entry in enumerate(entries):
        if entry.get("cropFileName") == crop:
            return entries.pop(i)
    return None


def _write_face_json_atomic(person_dir: Path, payload: dict) -> None:
    tmp = person_dir / "face.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, person_dir / "face.json")


def _apply_player_ops(person: dict, ops: list[PlayerOp]) -> None:
    """Apply *ops* in order to one person's folders. Every face.json touched
    is written once at the end — also when an op fails part-way, so the
    files already moved and their embeddings never disagree."""
    folders = {d.name: (d, payload) for d, payload in person["folders"]}
    touched: dict[str, tuple[Path, dict]] = {}
    try:
        for op in ops:
            if op.folder not in folders:
                raise HTTPException(status_code=400, detail="Image doesn't belong to this player.")
            d, payload = folders[op.folder]
            crop = _safe_component(op.crop)
            current = next((s for s in PLAYER_IMAGE_SETS if (d / s / crop).is_file()), None)
            if current is None:
                raise HTTPException(status_code=404, detail=f"Image {crop!r} not found.")
            src = d / current / crop

            if op.op == "move":
                if op.to not in PLAYER_IMAGE_SETS:
                    raise HTTPException(status_code=400, detail="Move target must be Face or Negative.")
                if op.to == current:
                    continue
                (d / op.to).mkdir(parents=True, exist_ok=True)
                dest = _unique_dest(d / op.to, crop)
                shutil.move(str(src), str(dest))
                entry = _pop_face_set_entry(payload, PLAYER_IMAGE_SETS[current], crop)
                if entry is not None:
                    entry["cropFileName"] = dest.name
                    payload.setdefault(PLAYER_IMAGE_SETS[op.to], []).append(entry)
            elif op.op == "delete":
                trash = d / DELETED_SUBDIR
                trash.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(_unique_dest(trash, crop)))
                anno = d / FACE_ANNOTATED_SUBDIR / crop
                if anno.is_file():
                    shutil.move(str(anno), str(_unique_dest(trash, f"annotated_{crop}")))
                _pop_face_set_entry(payload, PLAYER_IMAGE_SETS[current], crop)
                deleted = payload.setdefault(DELETED_CROPS_KEY, [])
                if crop not in deleted:
                    deleted.append(crop)
            elif op.op == "profile":
                if current != FACE_SUBDIR:
                    raise HTTPException(status_code=400, detail="Only positive images can be the profile picture.")
                primary_dir, primary = person["folders"][0]
                primary[PROFILE_CROP_KEY] = crop
                touched[primary_dir.name] = (primary_dir, primary)
                continue
            else:
                raise HTTPException(status_code=400, detail=f"Unknown op {op.op!r}.")
            touched[d.name] = (d, payload)

        if person["folders"]:
            primary_dir, primary = person["folders"][0]
            profile = _resolve_profile(person, _player_sets(person)[0])
            settled = profile["crop"] if profile else None
            if primary.get(PROFILE_CROP_KEY) != settled:
                if settled:
                    primary[PROFILE_CROP_KEY] = settled
                else:
                    primary.pop(PROFILE_CROP_KEY, None)
                touched[primary_dir.name] = (primary_dir, primary)
    finally:
        for d, payload in touched.values():
            _write_face_json_atomic(d, payload)


def _ensure_face_db_idle() -> None:
    with export_state.lock:
        if export_state.running:
            raise HTTPException(status_code=409, detail="An export is updating the face database — try again when it finishes.")
    with face_db_rebuild_state.lock:
        if face_db_rebuild_state.running:
            raise HTTPException(status_code=409, detail="The face database is being rebuilt — try again when it finishes.")


def _flush_team_face_db(team_dir: Path, say) -> dict:
    """Permanently empty every person's Deleted/ folder and remove the
    Annotated.<timestamp>/ debug folders left by earlier rebuilds."""
    files = annotated = 0
    for d in sorted(team_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("Annotated."):
            shutil.rmtree(d)
            annotated += 1
            continue
        trash = d / DELETED_SUBDIR
        if trash.is_dir():
            count = sum(1 for p in trash.rglob("*") if p.is_file())
            shutil.rmtree(trash)
            files += count
            if count:
                say(f"  {d.name}: {count} deleted file(s) removed.")
    say(f"Flushed {files} deleted file(s) and {annotated} old Annotated folder(s).")
    return {"flushedFiles": files, "flushedAnnotated": annotated}


def _run_face_db_rebuild(team_dir: Path) -> None:
    state = face_db_rebuild_state

    def say(msg: str) -> None:
        state.log(msg)
        log.info("[FaceDbRebuild] %s", msg)

    try:
        say("Step 1/2: flushing deleted faces...")
        result = _flush_team_face_db(team_dir, say)
        if not any(_is_person_dir(d) for d in team_dir.iterdir()):
            say("Step 2/2: this team's face database has no players yet — nothing to rebuild.")
        else:
            say("Step 2/2: rebuilding embeddings and calibrating (RebuildFaceDB.py)...")
            rc = _stream_subprocess([sys.executable, str(REPO_ROOT / "RebuildFaceDB.py"), str(team_dir)], log_fn=say)
            if rc != 0:
                raise RuntimeError(f"RebuildFaceDB.py failed with exit code {rc}")
        say("Face DB rebuild complete.")
        with state.lock:
            state.done = True
            state.result = result
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[FaceDbRebuild] failed")
        with state.lock:
            state.error = str(exc)
    finally:
        state.mark_stopped()


class ClusterOperation(BaseModel):
    type: str            # "assign" | "delete"
    sourceCluster: str
    crop: str
    name: Optional[str] = None


# --------------------------------------------------------------------------- #
# On-demand annotated previews
#
# An album's annotated previews (boxes, scores, names) are a cache rendered
# the first time one is requested -- not pre-generated at import. Rendering
# goes through algo/stages/annotation.py::render_album_preview, the same code
# that has always drawn them, fed from the Album's persisted per-person data
# plus the CURRENT face-recognition names. Deleting a cached file (see
# _invalidate_cached_previews) is all it takes to have it redrawn.
# --------------------------------------------------------------------------- #

# Decoding + drawing one full-size photo costs tens of MB and real CPU, and
# the browser asks for many previews/thumbnails at once (the culling nav strip,
# the album cards), so only a few may render concurrently.
_PREVIEW_RENDER_SLOTS = threading.BoundedSemaphore(max(1, min(4, os.cpu_count() or 2)))
# One lock per preview file (striped), so concurrent requests for the SAME
# photo render it once, and an invalidation can't be overwritten by a render
# that started before it.
_PREVIEW_LOCKS = tuple(threading.Lock() for _ in range(64))


def _preview_lock(preview: Path) -> threading.Lock:
    return _PREVIEW_LOCKS[hash(str(preview)) % len(_PREVIEW_LOCKS)]


def _warm_preview_renderer() -> None:
    """Import, in the background at startup, the image reader previews are
    decoded with. It pulls in the whole detection stack (several seconds), and
    the first preview request shouldn't be the one that pays for it."""
    try:
        import algo.stages.image_analysis  # noqa: F401
    except Exception as exc:  # noqa: BLE001 -- only an optimisation
        log.debug("Preview renderer warm-up skipped: %s", exc)


# album dir -> (signature of its named clusters' face.json files, name index)
_face_name_index_cache: dict[str, tuple[tuple, dict[str, list[tuple[dict, str]]]]] = {}
_face_name_index_lock = threading.Lock()


def _face_name_index(album_path: Path) -> dict[str, list[tuple[dict, str]]]:
    """``{photo key: [(body_bbox wire dict, player name), ...]}`` for every face
    the album's face recognition has put in a NAMED cluster -- automatic
    database matches and hand-assigned names alike. Unnamed (numeric) clusters
    contribute nothing.

    Rebuilt only when a named cluster's face.json changes (compared by
    mtime+size), so rendering many previews doesn't re-read every cluster."""
    fr = _facereco_dir(album_path)
    if fr is None:
        return {}

    clusters: list[tuple[Path, tuple]] = []
    for child in sorted(fr.iterdir()):
        if not child.is_dir() or child.name.startswith(".") or _is_pending_cluster(child.name):
            continue
        try:
            stat = (child / "face.json").stat()
        except OSError:
            continue
        clusters.append((child, (child.name, stat.st_mtime_ns, stat.st_size)))
    signature = tuple(sig for _child, sig in clusters)

    cache_key = str(album_path)
    with _face_name_index_lock:
        cached = _face_name_index_cache.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]

    index: dict[str, list[tuple[dict, str]]] = {}
    for child, _sig in clusters:
        payload = _load_face_json(child)
        name = (payload.get("name") or child.name).strip()
        if not name:
            continue
        for face in payload.get("faces", []):
            origin = face.get("origFilename")
            body_bbox = (face.get("Body") or {}).get("body_bbox")
            if origin and body_bbox:
                index.setdefault(origin, []).append((body_bbox, name))

    with _face_name_index_lock:
        _face_name_index_cache[cache_key] = (signature, index)
    return index


def _player_name_resolver(album_path: Path, image: AlbumImage):
    """A ``PersonRecord -> name`` lookup for *image*: the name of the
    face-recognition cluster holding that person's face, "" if none."""
    faces = _face_name_index(album_path).get(image.key, [])

    def resolve(record: PersonRecord) -> str:
        bbox = record.body_bbox
        if bbox is None:
            return ""
        wire = bbox.to_wire()
        return next((name for body_bbox, name in faces if _boxes_match(body_bbox, wire)), "")

    return resolve


def _ensure_preview(image: Optional[AlbumImage]) -> Optional[Path]:
    """The annotated preview file for *image*, rendering and caching it first
    if it doesn't exist yet. None if the photo is unknown or can't be rendered
    (its source file is gone or unreadable)."""
    if image is None:
        return None
    preview = image.preview_path
    if preview is None:
        return None
    if preview.is_file():
        return preview

    album_path = image.album.path
    with _preview_lock(preview):
        if preview.is_file():  # another request rendered it while we waited
            return preview
        with _PREVIEW_RENDER_SLOTS:
            try:
                rendered = render_album_preview(
                    image, player_name_for=_player_name_resolver(album_path, image),
                )
            except Exception:  # noqa: BLE001 -- one bad photo mustn't 500 the whole page
                log.exception("Could not render the preview for %s", image.key)
                return None
    return preview if rendered and preview.is_file() else None


def _ensure_thumbnail(image: Optional[AlbumImage]) -> Optional[Path]:
    """The nav-strip thumbnail for *image*, generating it first if it doesn't
    exist. A thumbnail already on disk -- written at import, or an older
    album's own -- is served as is. A missing one is made straight from the
    source photo, never by rendering the annotated preview. Only when the
    source is gone does a cached preview serve as the last resort. None if the
    photo is unknown or no thumbnail can be produced."""
    if image is None:
        return None
    existing = image.existing_thumbnail()
    if existing is not None:
        return existing

    thumbnail = image.thumbnail_path
    # Lazy, like the preview reader: it pulls in the detection stack.
    from algo.stages.image_analysis import render_album_thumbnail

    with _preview_lock(thumbnail):
        if thumbnail.is_file():  # another request made it while we waited
            return thumbnail
        with _PREVIEW_RENDER_SLOTS:
            try:
                made = render_album_thumbnail(image)
            except Exception:  # noqa: BLE001 -- one bad photo mustn't 500 the whole strip
                log.exception("Could not generate the thumbnail for %s", image.key)
                made = False
    if made and thumbnail.is_file():
        return thumbnail
    return image.ensure_thumbnail()


def _invalidate_cached_previews(album_path: Path, keys) -> int:
    """Delete the cached previews of the photos *keys*, so the next request
    re-renders them from current data. Returns the file count. Thumbnails
    are kept: they show the plain photo, which these changes don't affect."""
    album = album_for(album_path)
    removed = 0
    for key in sorted(set(keys)):
        image = album.image(key)
        preview = image.preview_path if image else None
        if image is None or preview is None:
            continue
        with _preview_lock(preview):
            removed += invalidate_previews(album_path, [image.preview_stem])
    return removed


class FacesCommitRequest(BaseModel):
    operations: list[ClusterOperation]


def _next_numeric_id(fr: Path) -> str:
    highest = 0
    for child in fr.iterdir():
        if child.is_dir() and child.name.isdigit():
            highest = max(highest, int(child.name))
    return f"{highest + 1:04d}"


def _ensure_target_cluster(fr: Path, name: str, template: dict, new_id: str) -> Path:
    """Return the target cluster dir for ``name`` inside ``fr``, creating it
    (folder + skeleton face.json) if it does not yet exist."""
    target = fr / name
    if not target.exists():
        (target / FACE_SUBDIR).mkdir(parents=True, exist_ok=True)
        (target / "Negative").mkdir(parents=True, exist_ok=True)
        _write_face_json(target, {
            "name": name,
            "playernum": None,
            "cluster": new_id,
            "provider": template.get("provider"),
            "aligned": template.get("aligned"),
            "faces": [],
        })
    else:
        (target / FACE_SUBDIR).mkdir(parents=True, exist_ok=True)
    return target


def _unique_crop_name(face_dir: Path, crop: str) -> str:
    """Avoid clobbering an existing crop of the same name in the target."""
    if not (face_dir / crop).exists():
        return crop
    stem = Path(crop).stem
    suffix = Path(crop).suffix
    i = 1
    while (face_dir / f"{stem}_{i}{suffix}").exists():
        i += 1
    return f"{stem}_{i}{suffix}"


def _move_crop_files(src_cluster: Path, dst_cluster: Path, crop: str) -> str:
    """Move the crop PNG (and its Face.annotated twin if present). Returns the
    possibly-renamed crop filename in the destination."""
    dst_face = dst_cluster / FACE_SUBDIR
    dst_face.mkdir(parents=True, exist_ok=True)
    new_name = _unique_crop_name(dst_face, crop)

    src_png = src_cluster / FACE_SUBDIR / crop
    if src_png.is_file():
        shutil.move(str(src_png), str(dst_face / new_name))

    src_anno = src_cluster / FACE_ANNOTATED_SUBDIR / crop
    if src_anno.is_file():
        dst_anno = dst_cluster / FACE_ANNOTATED_SUBDIR
        dst_anno.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_anno), str(dst_anno / new_name))
    return new_name


def _delete_crop_files(cluster: Path, crop: str) -> None:
    for sub in (FACE_SUBDIR, FACE_ANNOTATED_SUBDIR):
        fp = cluster / sub / crop
        if fp.is_file():
            fp.unlink()


def _pop_face_entry(payload: dict, crop: str) -> Optional[dict]:
    faces = payload.get("faces", [])
    for i, face in enumerate(faces):
        if face.get("cropFileName") == crop:
            return faces.pop(i)
    return None


def _boxes_match(a: Optional[dict], b: Optional[dict]) -> bool:
    if not a or not b:
        return False
    return all(
        abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) <= BBOX_EPS
        for k in ("x1", "y1", "x2", "y2")
    )


def _update_results_json(album_path: Path, assignments: list[dict], cleared: Optional[list[dict]] = None) -> None:
    """Write player_name / player_number onto matching Album bodies.

    ``assignments`` is a list of {origFilename, body_bbox, name, playernum}.
    ``cleared`` lists {origFilename, body_bbox} of faces the user removed from
    a cluster: those people lose any name a previous assignment gave them.
    """
    cleared = cleared or []
    if not assignments and not cleared:
        return
    album = album_for(album_path)
    if not album.exists:
        return

    # One key can map to several photos only in malformed albums; keep the
    # list form so an unexpected duplicate is still updated rather than
    # silently skipped.
    by_name: dict[str, list] = {}
    for image in album.image_list:
        by_name.setdefault(image.key, []).append(image)

    changed = False
    for a in cleared:
        for image in by_name.get(a["origFilename"], []):
            for body in image.bodies:
                bbox = body.body_bbox
                if bbox and _boxes_match(bbox.to_wire(), a["body_bbox"]) and (body.player_name or body.player_number):
                    body.player_name = ""
                    body.player_number = ""
                    changed = True
    for a in assignments:
        for image in by_name.get(a["origFilename"], []):
            for body in image.bodies:
                bbox = body.body_bbox
                if bbox and _boxes_match(bbox.to_wire(), a["body_bbox"]):
                    body.player_name = a["name"]
                    body.player_number = a["playernum"]
                    changed = True

    if changed:
        album.save()


def _commit_cluster_operations(album_path: Path, req: FacesCommitRequest) -> dict:
    fr = _facereco_dir(album_path)
    if fr is None:
        raise HTTPException(status_code=400, detail="Album has no .FaceReco folder")

    # Cache of cluster payloads we mutate, keyed by absolute path.
    payloads: dict[Path, dict] = {}

    def payload_for(cluster_dir: Path) -> dict:
        if cluster_dir not in payloads:
            payloads[cluster_dir] = _load_face_json(cluster_dir)
        return payloads[cluster_dir]

    # For fresh named clusters created in this batch, remember their dir so a
    # second assignment to the same name reuses it.
    target_dirs: dict[str, Path] = {}
    results_updates: list[dict] = []
    results_cleared: list[dict] = []
    # Photos whose face-recognition result the user just confirmed (a name
    # assigned, or a face rejected): their cached previews show the old
    # result, so they are dropped and re-rendered on next view.
    touched_keys: set[str] = set()

    # Process deletions first, then assignments.
    ops = sorted(req.operations, key=lambda o: 0 if o.type == "delete" else 1)
    deleted_keys: set[tuple[str, str]] = set()

    for op in ops:
        src_name = _safe_component(op.sourceCluster)
        crop = _safe_component(op.crop)
        src_dir = fr / src_name
        if not src_dir.is_dir():
            continue
        src_payload = payload_for(src_dir)

        if op.type == "delete":
            entry = _pop_face_entry(src_payload, crop)
            _delete_crop_files(src_dir, crop)
            deleted_keys.add((src_name, crop))
            if entry is not None:
                record_manual_override(fr, deleted={
                    "file": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                })
                results_cleared.append({
                    "origFilename": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                })
                touched_keys.add(entry.get("origFilename", ""))
            continue

        if op.type == "assign":
            if (src_name, crop) in deleted_keys:
                continue  # delete wins
            name = (op.name or "").strip()
            if not name or name == src_name:
                continue  # nothing to do / already in this named cluster

            if name in target_dirs:
                dst_dir = target_dirs[name]
            else:
                dst_dir = _ensure_target_cluster(
                    fr, name, src_payload, _next_numeric_id(fr)
                )
                target_dirs[name] = dst_dir
            dst_payload = payload_for(dst_dir)

            entry = _pop_face_entry(src_payload, crop)
            new_crop = _move_crop_files(src_dir, dst_dir, crop)
            if entry is not None:
                entry["cropFileName"] = new_crop
                dst_payload.setdefault("faces", []).append(entry)
                results_updates.append({
                    "origFilename": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                    "name": name,
                    "playernum": dst_payload.get("playernum"),
                })
                record_manual_override(fr, assigned={
                    "file": entry.get("origFilename", ""),
                    "body_bbox": (entry.get("Body") or {}).get("body_bbox"),
                    "name": name,
                    "playernum": dst_payload.get("playernum"),
                })
                touched_keys.add(entry.get("origFilename", ""))

    # Rewrite every touched face.json.
    for cluster_dir, payload in payloads.items():
        if cluster_dir.is_dir():
            _write_face_json(cluster_dir, payload)

    # Dissolve emptied pending (numeric) clusters.
    for cluster_dir, payload in payloads.items():
        if (
            cluster_dir.is_dir()
            and _is_pending_cluster(cluster_dir.name)
            and not payload.get("faces")
        ):
            shutil.rmtree(cluster_dir, ignore_errors=True)

    _update_results_json(album_path, results_updates, results_cleared)
    # After the cluster files and album are written, so a preview re-rendered
    # from here on sees the confirmed names.
    _invalidate_cached_previews(album_path, touched_keys - {""})

    return {"ok": True, "clusters": _build_clusters(album_path)}


# --------------------------------------------------------------------------- #
# Export (page 5 — Album Tools). Three steps, run in a background thread while the
# browser polls /api/album-tools/export-status the same way it polls processing
# output:
#
#   1. Sync every recognized/tagged player's face crops (from kept images
#      only) into the permanent, cross-album system face database at
#      REPO_ROOT/.FaceReco/<team id>/ — uncapped; RebuildFaceDB.py's retire step (below)
#      is what curates it, not this step.
#   2. Run RebuildFaceDB.py against that system database to regenerate
#      embeddings and retire redundant crops.
#   3. Run apply_export.py to copy every kept photo (with edits + metadata
#      preserved) plus players.csv into the user-chosen destination — a flat
#      folder, no subdirectories. albums/<album>/ (and its own .FaceReco) is
#      the persistent "working album"; the destination is just the final
#      deliverable.
# --------------------------------------------------------------------------- #
def _export_log(msg: str) -> None:
    export_state.log(msg)
    log.info("[Export] %s", msg)


def _stream_subprocess(cmd: list[str], on_line=None, log_fn=None) -> int:
    """Run *cmd*, forwarding each combined stdout/stderr line to *log_fn*
    (default _export_log) and optionally *on_line* for progress-counter
    bookkeeping. Returns the process's exit code."""
    log_fn = log_fn or _export_log
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        log_fn(line)
        if on_line:
            on_line(line)
    proc.wait()
    return proc.returncode


def _sync_named_clusters_to_system_db(album_path: Path, kept: set[str]) -> int:
    """Copy every recognized/tagged person's face crops from *kept* images in
    this album's own .FaceReco into the album's team folder of the system
    face DB (REPO_ROOT/.FaceReco/<team id>/, creating a person folder there
    the first time we see them). Any tagged name counts -- roster member or
    not -- so coaches/guests tagged by hand are matched next time. Copies
    everything uncapped — RebuildFaceDB.py's retire step decides afterward
    what's actually worth keeping. Returns the number of tagged people
    processed."""
    fr = _facereco_dir(album_path)
    if fr is None:
        _export_log("No .FaceReco folder in this album — nothing to sync.")
        return 0

    named_clusters = [
        child for child in sorted(fr.iterdir())
        if child.is_dir() and not child.name.startswith(".") and not _is_pending_cluster(child.name)
    ]
    if not named_clusters:
        _export_log("No tagged players found in this album — nothing to sync.")
        return 0

    system_fr = _team_face_db_dir(_album_team_id(album_path))
    system_fr.mkdir(parents=True, exist_ok=True)

    processed = 0
    for cluster_dir in named_clusters:
        payload = _load_face_json(cluster_dir)
        name = (payload.get("name") or cluster_dir.name).strip() or cluster_dir.name

        candidates = [
            f for f in payload.get("faces", [])
            if f.get("cropFileName") and Path(f.get("origFilename", "")).name in kept
        ]
        with export_state.lock:
            export_state.processed_players = processed + 1
            export_state.total_players = len(named_clusters)
        if not candidates:
            _export_log(f"  {name}: no kept faces to sync.")
            processed += 1
            continue

        dest_dir = system_fr / _sanitize_player_dirname(name)
        dest_face_dir = dest_dir / FACE_SUBDIR
        dest_face_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "Negative").mkdir(parents=True, exist_ok=True)

        if not (dest_dir / "face.json").is_file():
            _write_face_json(dest_dir, {
                "name": name,
                "playernum": payload.get("playernum"),
                "provider": payload.get("provider"),
                "cluster": cluster_dir.name,
                "faces": [],
                "negative_faces": [],
            })

        copied = 0
        skipped = 0
        # Crops the user moved to Negative/ or deleted on the Players page
        # must not come back as positives on the next export.
        curated_out = set(_load_face_json(dest_dir).get(DELETED_CROPS_KEY) or [])
        for f in candidates:
            src_crop = cluster_dir / FACE_SUBDIR / f["cropFileName"]
            if not src_crop.is_file():
                continue
            dest_crop = dest_face_dir / f["cropFileName"]
            if f["cropFileName"] in curated_out or (dest_dir / NEGATIVE_SUBDIR / f["cropFileName"]).is_file():
                skipped += 1
                continue
            if dest_crop.is_file():
                # Already synced in a previous export run — skip instead of
                # copying a renamed duplicate (crop file names are
                # deterministic per source photo/body, so an existing file
                # of the same name means this exact face is already there).
                skipped += 1
                continue
            shutil.copy2(str(src_crop), str(dest_crop))
            src_anno = cluster_dir / FACE_ANNOTATED_SUBDIR / f["cropFileName"]
            if src_anno.is_file():
                dest_anno_dir = dest_dir / FACE_ANNOTATED_SUBDIR
                dest_anno_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(src_anno), str(dest_anno_dir / f["cropFileName"]))
            copied += 1
        _export_log(
            f"  {name}: {copied} new face crop(s) synced, {skipped} already present "
            f"(of {len(candidates)} kept face crop(s))."
        )
        processed += 1

    _export_log(f"Face-DB sync complete: {processed}/{len(named_clusters)} player(s) processed.")
    return processed


_FACE_TAGGING_CSV_HEADER = ["Player Name", "Player Number", "Image Path"]


def _collect_all_face_tagging_rows(album_path: Path) -> list[list[str]]:
    """One row per tagged face across the ENTIRE album, sourced from its own
    .FaceReco database the same way as apply_export.py's players.csv rows —
    but with no "kept"/star-rating filter, since this is the complete
    face-tagging record rather than a photo-export side effect."""
    rows: list[list[str]] = []
    fr = _facereco_dir(album_path)
    if fr is None:
        return rows
    for cluster_dir in sorted(fr.iterdir()):
        if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
            continue
        if _is_pending_cluster(cluster_dir.name):
            continue  # unnamed cluster -- nobody to tag
        payload = _load_face_json(cluster_dir)
        name = (payload.get("name") or "").strip()
        if not name:
            continue
        playernum = payload.get("playernum")
        number_str = "" if playernum is None else str(playernum)
        for face in payload.get("faces", []):
            rows.append([name, number_str, face.get("origFilename", "")])
    rows.sort(key=lambda row: (row[0], row[2]))
    return rows


def _collect_player_frame_paths(album_path: Path) -> dict[str, list[Path]]:
    """{player name: every distinct photo they're tagged in}, resolved to
    on-disk paths via the Album — always the ORIGINAL photo, never an
    accepted AI edit, same as algo/burst_video.py's detect_bursts — the
    source data for rendering one highlight MP4 per player. Faces on
    clusters sharing the same name (e.g. after a merge) are pooled and
    de-duplicated by resolved path."""
    fr = _facereco_dir(album_path)
    if fr is None:
        return {}
    album = album_for(album_path)
    by_name: dict[str, dict[str, Path]] = {}
    for cluster_dir in sorted(fr.iterdir()):
        if not cluster_dir.is_dir() or cluster_dir.name.startswith("."):
            continue
        if _is_pending_cluster(cluster_dir.name):
            continue  # unnamed cluster -- nobody to tag
        payload = _load_face_json(cluster_dir)
        name = (payload.get("name") or "").strip()
        if not name:
            continue
        bucket = by_name.setdefault(name, {})
        for face in payload.get("faces", []):
            orig = face.get("origFilename", "")
            if not orig:
                continue
            image = album.image(orig)
            path = image.original_path if image else None
            if path is None or not path.is_file():
                continue
            bucket[str(path)] = path
    return {name: list(paths.values()) for name, paths in by_name.items()}


def _open_in_file_explorer(path: Path) -> None:
    """Open *path* in the OS's native file explorer. Only meaningful when the
    server and browser run on the same machine (see api_browse_folder's
    docstring for the same assumption)."""
    if sys.platform == "win32":
        os.startfile(str(path))  # noqa: S606 — local-machine convenience only
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _run_export(album_path: Path, dest_dir: Path, export_face_tagging: bool, min_stars: int) -> None:
    try:
        kept = _kept_image_basenames(album_path, min_stars)
        with export_state.lock:
            export_state.total_images = len(kept)
            export_state.dest_dir = str(dest_dir)

        team_id = _album_team_id(album_path)
        _export_log("Step 1/3: syncing tagged face crops into this team's face database (.FaceReco/<team id>)...")
        _sync_named_clusters_to_system_db(album_path, kept)

        system_fr = _team_face_db_dir(team_id)
        if system_fr.is_dir() and any(p.is_dir() and not p.name.startswith("Annotated.") for p in system_fr.iterdir()):
            _export_log("Step 2/3: rebuilding face database embeddings (RebuildFaceDB.py)...")
            rc = _stream_subprocess([
                sys.executable, str(REPO_ROOT / "RebuildFaceDB.py"), str(system_fr), "--skip-calibration",
            ])
            if rc != 0:
                raise RuntimeError(f"RebuildFaceDB.py failed with exit code {rc}")
        else:
            _export_log("Step 2/3: team face database is empty — skipping rebuild.")

        _export_log(f"Step 3/3: exporting {len(kept)} kept photo(s) + players.csv to {dest_dir}...")
        dest_dir.mkdir(parents=True, exist_ok=True)

        def _bump_copied(line: str) -> None:
            # apply_export.py logs "  Copied: <name>" but the subprocess's
            # own logging.Formatter prepends "<time> [LEVEL   ] " to every
            # line, so match the message as a substring rather than a
            # prefix (startswith would never match and copied_images would
            # stay stuck at 0 despite the export succeeding).
            if "  Copied: " in line:
                with export_state.lock:
                    export_state.copied_images += 1

        cmd = [sys.executable, str(REPO_ROOT / "apply_export.py"), str(album_path), str(dest_dir), "--min-stars", str(min_stars)]
        if export_face_tagging:
            cmd.append("--export-face-tagging")
        rc = _stream_subprocess(cmd, on_line=_bump_copied)
        if rc != 0:
            raise RuntimeError(f"apply_export.py failed with exit code {rc}")

        _export_log("Export complete.")
        with export_state.lock:
            export_state.done = True
        state = _load_state()
        state["lastExportPath"] = str(dest_dir.resolve())
        _save_state(state)
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[Export] failed")
        with export_state.lock:
            export_state.error = str(exc)
    finally:
        export_state.mark_stopped()


def _run_burst_video(album_path: Path, dest_dir: Path, min_frames: int, mode: str, fps: float) -> None:
    try:
        result = render_album_bursts(
            album_path, dest_dir, min_frames=min_frames, mode=mode, fps=fps, on_line=burst_video_state.log
        )
        with burst_video_state.lock:
            burst_video_state.output_dir = str(dest_dir)
            burst_video_state.result = result
            burst_video_state.done = True
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[BurstVideo] failed")
        with burst_video_state.lock:
            burst_video_state.error = str(exc)
    finally:
        burst_video_state.mark_stopped()


def _run_face_tagging_csv_export(album_path: Path, dest_dir: Path) -> None:
    def _log_line(msg: str) -> None:
        face_tagging_csv_state.log(msg)
        log.info("[FaceTaggingCSV] %s", msg)

    try:
        _log_line("Collecting tagged faces from .FaceReco...")
        rows = _collect_all_face_tagging_rows(album_path)
        _log_line(f"Found {len(rows)} tagged face row(s).")
        dest_dir.mkdir(parents=True, exist_ok=True)
        csv_path = dest_dir / f"{album_path.name}_face_tagging.csv"
        _log_line(f"Writing {csv_path}...")
        with open(csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(_FACE_TAGGING_CSV_HEADER)
            writer.writerows(rows)
        _log_line("Done.")
        with face_tagging_csv_state.lock:
            face_tagging_csv_state.output_dir = str(dest_dir)
            face_tagging_csv_state.result = {"rows": len(rows), "csvPath": str(csv_path)}
            face_tagging_csv_state.done = True
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[FaceTaggingCSV] failed")
        with face_tagging_csv_state.lock:
            face_tagging_csv_state.error = str(exc)
    finally:
        face_tagging_csv_state.mark_stopped()


def _run_player_video(album_path: Path, dest_dir: Path, fps: float) -> None:
    try:
        player_frames = _collect_player_frame_paths(album_path)
        result = render_player_videos(player_frames, dest_dir, fps=fps, on_line=player_video_state.log)
        with player_video_state.lock:
            player_video_state.output_dir = str(dest_dir)
            player_video_state.result = result
            player_video_state.done = True
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        log.exception("[PlayerVideo] failed")
        with player_video_state.lock:
            player_video_state.error = str(exc)
    finally:
        player_video_state.mark_stopped()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(title="Photo Culling UI")
templates = Jinja2Templates(directory=str(WEBUI_DIR / "templates"))


def _static_url(path: str) -> str:
    """Build a `/static/...` URL with a cache-busting `?v=<mtime>` query
    param, so browsers pick up edited JS/CSS immediately instead of serving
    a stale cached copy after the file changes on disk."""
    fp = WEBUI_DIR / "static" / path
    try:
        version = int(fp.stat().st_mtime)
    except OSError:
        version = 0
    return f"/static/{path}?v={version}"


templates.env.globals["static_url"] = _static_url
templates.env.globals["has_active_album"] = _has_active_album
templates.env.globals["has_teams"] = _has_teams

PAGE_STEPS: tuple[str, ...] = ("team", "players", "select-album", "culling", "faces", "album-tools")


@app.get("/")
def root() -> RedirectResponse:
    # Send first-time users straight to the Team page until at least one
    # team has been created — every other page (Select Album, Review, etc.)
    # assumes a team already exists.
    if not _has_teams():
        return RedirectResponse(url="/team")
    return RedirectResponse(url="/select-album")


@app.post("/api/heartbeat")
def api_heartbeat() -> dict:
    with heartbeat_state.lock:
        heartbeat_state.last_heartbeat = time.time()
    return {"ok": True}


def _render_page(step: str, request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, f"{step}.html", {"active_step": step})


for _step in PAGE_STEPS:
    def _make_route(step: str):
        def _route(request: Request):
            # Select Album needs a team (every import is tied to one), and so
            # does Players (it edits a team's face DB); first-time users are
            # sent to the Team page to create one.
            if step in ("select-album", "players") and not _has_teams():
                return RedirectResponse(url="/team")
            return _render_page(step, request)
        return _route
    app.add_api_route(f"/{_step}", _make_route(_step), methods=["GET"], response_class=HTMLResponse)


@app.get("/add-new-album")
def add_new_album_page(request: Request):
    # Importing needs a team to pick; send first-time users to create one.
    if not _has_teams():
        return RedirectResponse(url="/team")
    # Sub-page of "Select Album" (reached via the "+" polaroid card there),
    # not a numbered step of its own — highlight the "Select Album" step.
    return templates.TemplateResponse(request, "add-new-album.html", {"active_step": "select-album"})


@app.post("/api/team")
def api_save_team(team: TeamData) -> dict:
    # A nameless team would later be auto-named "Team N" on load; require one.
    team.name = team.name.strip()
    if not team.name:
        raise HTTPException(status_code=400, detail="Team name is required.")
    saved = _save_team(team)
    return {"ok": True, "team": saved.model_dump()}


@app.get("/api/teams/query")
def api_query_teams() -> dict:
    """QueryTeams — returns settings (jersey colours, roster, API key) for
    every team stored in team.json, plus the shared Global settings block.
    Teams are ordered most-recently-used-to-import first (never-used teams
    keep their on-disk order, after all used ones)."""
    teams_file = _load_teams_file()
    ordered = sorted(teams_file.Teams, key=lambda t: t.lastUsedAt or "", reverse=True)
    return {
        "teams": [team.model_dump() for team in ordered],
        "global": teams_file.Global.model_dump(),
    }


@app.get("/api/sensitivity-presets")
def api_sensitivity_presets() -> dict:
    return SENSITIVITY_PRESETS


@app.get("/api/albums")
def api_list_albums() -> dict:
    return {"albums": _list_albums()}


@app.post("/api/albums/select")
def api_select_album(req: AlbumSelectRequest) -> dict:
    path = _album_dir_by_id(req.id)
    if not _is_album_complete(path):
        raise HTTPException(status_code=400, detail="Not a fully processed album.")
    state = _load_state()
    state["currentAlbum"] = str(path.resolve())
    _save_state(state)
    log.info("Current album set: %s", path.resolve())
    return {"ok": True, "album": _read_album_summary(path)}


@app.get("/api/albums/thumb")
def api_album_thumb(id: str = Query(...), file: str = Query(...)) -> FileResponse:
    """Small thumbnail of one photo in *any* album (not just the current
    one) — the rotating sample images on the album-picker cards.

    Served straight from the album's ``thumbnails/<key>.jpg`` file written at
    import, without loading album.json: that file is tens of MB per album, and
    parsing one per card on every visit is what made the page slow. Only a
    photo with no thumbnail yet (an older album) goes through the album to
    have one generated, once. The picture never changes, so browsers may
    reuse it for a while."""
    album_path = _album_dir_by_id(id)
    headers = {"Cache-Control": "private, max-age=3600"}
    direct = album_path / THUMBNAILS_SUBDIR / f"{_safe_component(file)}.jpg"
    if direct.is_file():
        return FileResponse(direct, headers=headers)
    fp = _ensure_thumbnail(album_for(album_path).image(file))
    if fp is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fp, headers=headers)


@app.post("/api/albums/delete")
def api_delete_album(req: AlbumDeleteRequest) -> dict:
    album_path = _album_dir_by_id(req.id)
    shutil.rmtree(album_path)
    state = _load_state()
    if state.get("currentAlbum") == str(album_path.resolve()):
        state["currentAlbum"] = ""
        _save_state(state)
    log.info("Album deleted: %s", album_path.resolve())
    return {"ok": True}


@app.get("/api/current-album")
def api_current_album() -> dict:
    state = _load_state()
    album_path = state.get("currentAlbum", "")
    if not album_path or not _is_album_complete(Path(album_path)):
        return {"album": None}
    return {"album": _read_album_summary(Path(album_path))}


@app.get("/api/culling/summary")
def api_culling_summary() -> dict:
    """Count of reviewable images at each effective star level (0 = skipped,
    1-5 = graded), for the star-filter checkboxes' live counts."""
    album_path = _current_album_path()
    star_counts = {str(n): 0 for n in range(6)}
    for image in _all_culling_images(album_path, with_timestamp=False):
        star_counts[str(image["stars"])] += 1
    return {"starCounts": star_counts}


@app.get("/api/culling/data")
def api_culling_data(stars: str = Query(...), sort: str = Query("size")) -> dict:
    """Images whose effective star rating is one of the independently
    selected *stars* levels (comma-separated, e.g. "1,3,5"), grouped into
    bursts (or rating-sorted chunks) exactly like the un-filtered page did —
    grouping only ever sees the already-filtered pool."""
    star_set = _parse_star_filter(stars)
    album_path = _current_album_path()
    images = [im for im in _all_culling_images(album_path) if im["stars"] in star_set]
    if sort == "rating":
        groups = _rating_groups(images)
    else:
        groups = _sort_groups(_group_bursts(images), sort)
    return {
        "stars": sorted(star_set),
        "groups": [
            {"images": [
                {"file": im["file"], "path": im["path"], "keep": im["keep"], "stars": im["stars"], "burstRanking": im["burstRanking"], "llmGrade": im["llmGrade"]}
                for im in group
            ]}
            for group in groups
        ],
    }


@app.get("/api/culling/thumb/{file:path}")
def api_culling_thumb(file: str) -> FileResponse:
    """Small square thumbnail of *file*, for the culling page's nav strip.

    Served from the thumbnail written at import (or an older album's own),
    and generated from the source photo if it is missing -- never by
    rendering the annotated preview (see `_ensure_thumbnail`)."""
    thumb_fp = _ensure_thumbnail(album_for(_current_album_path()).image(file))
    if thumb_fp is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(thumb_fp)


@app.get("/api/anno_img/{file:path}")
def api_anno_img(file: str) -> FileResponse:
    """Full-resolution annotated preview image — shared by the culling page's
    main pane (toggle-able against /api/original) and any other consumer
    that wants the annotated (not cropped/resized) version of a photo.
    Rendered on first request, then served from the album's preview cache.
    `no-cache` because the file is redrawn at the same URL when an edit is
    accepted or reverted (see render_album_preview)."""
    fp = _ensure_preview(album_for(_current_album_path()).image(file))
    if fp is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fp, headers={"Cache-Control": "no-cache"})


@app.post("/api/culling/apply")
def api_culling_apply(req: CullingApplyRequest) -> dict:
    """Commit pending star-rating decisions (staged client-side, across every
    star-filter view) into the Album. Every reviewable entry gets an
    explicit "stars" field written — the user's override if they touched it
    this session, else its current effective (explicit-or-default) value —
    with "keep" always derived from the final star rating (3+ = keep) so it
    can never drift out of sync with the star-driven color scheme. Written
    atomically (temp file + os.replace, with the previous contents backed
    up) so a crash mid-write can never corrupt the Album."""
    album_path = _current_album_path()
    info = AlbumInfo(album_path)
    album = album_for(album_path)
    album_images = album.images

    for category in CULLING_CATEGORIES:
        default_stars = _CULLING_DEFAULT_STARS[category]
        for entry in info.entries(category):
            src_name = entry.key
            image = album_images.get(src_name)
            if image is None:
                continue
            current_stars = image.stars
            current_stars = current_stars if current_stars is not None else default_stars
            stars = int(req.starOverrides.get(src_name, current_stars))
            image.stars = stars
            image.keep = stars >= 3
            # Marks a rating the user actually chose, so a later regrade can
            # re-baseline everyone else's stars without discarding it. Only
            # keys present in starOverrides were genuinely touched -- every
            # other entry is rewritten above merely to persist its current
            # effective value.
            if src_name in req.starOverrides:
                image.stars_manual = True

    album.save()
    log.info("Review changes applied: %s (%d star overrides)",
             album_path.resolve(), len(req.starOverrides))
    return {"ok": True}


@app.get("/api/import-state")
def api_get_import_state() -> dict:
    state = _load_state()
    last_path = state.get("lastImportPath", "")
    return {
        "lastImportPath": last_path,
        "lastImportPathExists": bool(last_path) and Path(last_path).is_dir(),
        "sensitivityMode": state.get("sensitivityMode", "medium"),
        "sensitivityCustomValue": state.get("sensitivityCustomValue", 0.50),
        "recognizeFaces": state.get("recognizeFaces", True),
        "selectedTeamId": _canonical_team_id(state.get("selectedTeamId", "")),
    }


@app.post("/api/import-path")
def api_set_import_path(req: ImportPathRequest) -> dict:
    path = Path(req.path)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")
    state = _load_state()
    state["lastImportPath"] = str(path.resolve())
    _save_state(state)
    log.info("Import path set: %s", path.resolve())
    return {"ok": True, "path": str(path.resolve())}


@app.post("/api/browse-folder")
def api_browse_folder(req: BrowseFolderRequest = BrowseFolderRequest()) -> dict:
    """Open a native OS folder-picker dialog on the server machine.

    Only meaningful when the server and browser run on the same machine
    (the normal case for this tool — see InitPhotoProcessing.bat / launcher
    pattern). Returns {"path": null} if the user cancels.
    """
    state = _load_state()
    if req.context == "export":
        last_export = state.get("lastExportPath", "")
        initial_dir = str(Path(last_export).parent) if last_export else ""
    else:
        last_import = state.get("lastImportPath", "")
        initial_dir = str(Path(last_import).parent) if last_import else ""
    if not initial_dir or not Path(initial_dir).is_dir():
        initial_dir = str(Path.home())

    result: dict[str, Optional[str]] = {"path": None}

    def _run_dialog() -> None:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(initialdir=initial_dir, title=req.title)
        root.destroy()
        result["path"] = chosen or None

    # tkinter must run on the main thread on some platforms; FastAPI's sync
    # endpoints already execute in a worker thread, so calling it directly
    # here is safe (no separate thread needed) and keeps the dialog modal.
    _run_dialog()

    if result["path"]:
        log.info("Folder picked via dialog: %s", result["path"])
    else:
        log.info("Folder picker cancelled")
    return result


@app.post("/api/start-processing")
def api_start_processing(req: StartProcessingRequest) -> dict:
    path = Path(req.path)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")
    if not _has_teams():
        raise HTTPException(status_code=409, detail="No team exists yet — create one on the Team page first.")

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")

    # Remember these choices for next time, same pattern as lastImportPath.
    state = _load_state()
    state["sensitivityMode"] = req.sensitivityMode
    state["sensitivityCustomValue"] = req.sensitivityCustomValue
    state["recognizeFaces"] = req.recognizeFaces
    state["selectedTeamId"] = req.teamId
    _save_state(state)
    _touch_team_last_used(req.teamId)

    thread = threading.Thread(
        target=_run_processing,
        args=(str(path.resolve()), req.albumName, req.sensitivityMode, req.sensitivityCustomValue, req.recognizeFaces, req.noTeam, req.teamColor, req.teamId),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.post("/api/import-more")
def api_import_more(req: ImportMoreRequest) -> dict:
    """Import additional photos into the currently-selected album (page 5's
    "Import more images" button). Reuses the same processing_state/polling
    machinery as /api/start-processing — only one processing job (initial
    import OR import-more) can run at a time."""
    path = Path(req.path)
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.path}")

    album_dir = _current_album_path()

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")

    state = _load_state()
    state["lastImportPath"] = str(path.resolve())
    _save_state(state)

    thread = threading.Thread(
        target=_run_import_more,
        args=(str(path.resolve()), album_dir),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.post("/api/album-tools/rerun-facereco")
def api_rerun_facereco() -> dict:
    """Re-run face detection/clustering for the current album (Album Tools
    page's "Re-run Face Detection" button). Reuses the same
    processing_state/polling machinery as /api/start-processing and
    /api/import-more — only one processing job can run at a time."""
    album_dir = _current_album_path()

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")

    thread = threading.Thread(
        target=_run_rerun_facereco,
        args=(album_dir,),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


def _encode_jpeg(image, quality: int = _EDIT_PREVIEW_JPEG_QUALITY) -> bytes:
    import cv2  # lazy: pulls in the CV stack

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode preview")
    return buf.tobytes()


def _open_edit_session(key: str):
    """Decode *key*'s editable source (an already accepted edit if there is
    one, else the original — RAW decoded exactly as autoedit.py does) and
    cache a downscaled copy as the edit session. Returns (album_dir,
    full-size (width, height), autoedit.propose_tone() values)."""
    import cv2  # lazy: pulls in the CV stack
    import autoedit

    album_dir = _current_album_path()
    source_path, _pending, _final = _resolve_ai_edit_paths(album_dir, key)
    full = autoedit.read_image(source_path)
    if full is None:
        raise HTTPException(status_code=422, detail=f"Could not read {source_path.name}")

    proposed = autoedit.propose_tone(full)
    height, width = full.shape[:2]
    scale = EDIT_PREVIEW_LONG_EDGE / max(height, width)
    base = full
    if scale < 1.0:
        base = cv2.resize(full, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
    del full
    edit_session_state.open(album_dir, key, base, _encode_jpeg(base), (width, height))
    return album_dir, (width, height), proposed


def _require_edit_session(key: Optional[str]) -> dict:
    """The open edit session for *key*, transparently reopening it if the
    server lost it (restart) or another tab opened a different photo since."""
    session = edit_session_state.snapshot()
    if key and (session is None or session["key"] != key):
        _open_edit_session(key)
        session = edit_session_state.snapshot()
    if session is None:
        raise HTTPException(status_code=404, detail="No photo is open for editing")
    return session


@app.post("/api/edit/ai-edit/session")
def api_ai_edit_session(req: AiEditRequest) -> dict:
    """Open *req.file* in edit mode (Review/Faces page "AI edit" button):
    caches a downscaled copy of the exact pixels autoedit.py will edit for
    live previews (see _open_edit_session), and returns the auto
    ("suggested") tone values plus every detected body for the Sharpen
    players overlay (mapped onto the accepted edit if it was cropped).
    Calling it again for the same photo re-reads it from disk (the panel's
    Reload button)."""
    import autoedit

    album_dir, (width, height), proposed = _open_edit_session(req.file)
    image = album_for(album_dir).image(req.file)
    return {
        "file": req.file,
        "width": width,
        "height": height,
        "edited": image is not None and image.edited_path is not None,
        "proposed": {
            "highlights": proposed.highlights,
            "midtones": proposed.midtones,
            "shadows": proposed.shadows,
            "contrast": proposed.contrast,
        },
        "limits": {"ev": autoedit.TONE_EV_LIMIT, "contrast": autoedit.TONE_CONTRAST_LIMIT},
        "bodies": _edit_session_bodies(image.bodies, image.edited_crop) if image is not None else [],
    }


@app.get("/api/edit/ai-edit/session/before")
def api_ai_edit_session_before(file: Optional[str] = None) -> Response:
    """The open photo (*file*), unedited, at preview size."""
    session = _require_edit_session(file)
    return Response(session["before_jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/edit/ai-edit/session/preview")
def api_ai_edit_session_preview(
    file: Optional[str] = None,
    highlights: float = 0.0, midtones: float = 0.0, shadows: float = 0.0, contrast: float = 0.0,
) -> Response:
    """The open photo (*file*) at preview size with the given
    brightness/contrast, rendered by autoedit.apply_tone() — the same
    function the final render runs. Sharpening is not previewed."""
    import autoedit

    session = _require_edit_session(file)
    params = autoedit.ToneParams(highlights=highlights, midtones=midtones, shadows=shadows, contrast=contrast)
    out = autoedit.apply_tone(session["base"], params)
    return Response(_encode_jpeg(out), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.delete("/api/edit/ai-edit/session")
def api_ai_edit_session_close() -> dict:
    """Close edit mode without applying anything (Cancel)."""
    edit_session_state.clear()
    return {"ok": True}


@app.post("/api/edit/ai-edit/apply")
def api_ai_edit_apply(req: AiEditApplyRequest) -> dict:
    """Render the edit-mode panel's settings at full resolution via
    autoedit.py (see _run_ai_edit). Reuses the same processing_state/polling
    machinery as every other background job here — only one can run at a
    time. The result is parked as a pending edit for the before/after
    compare view; it does not become the photo's edited_image until
    accepted."""
    album_dir = _current_album_path()
    source_path, pending_path, final_path = _resolve_ai_edit_paths(album_dir, req.file)

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")
    session = edit_session_state.snapshot()
    source_size = session["size"] if session is not None and session["key"] == req.file else None
    edit_session_state.clear()

    thread = threading.Thread(
        target=_run_ai_edit,
        args=(album_dir, req.file, source_path, pending_path, final_path, req, source_size),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


def _require_pending_edit() -> dict:
    pending = ai_edit_state.snapshot()
    if pending is None:
        raise HTTPException(status_code=404, detail="No pending AI edit")
    return pending


@app.get("/api/edit/ai-edit/pending")
def api_ai_edit_pending() -> dict:
    """Whether a generated-but-undecided AI edit is waiting for the user's
    accept/reject verdict, and which photo it belongs to."""
    pending = ai_edit_state.snapshot()
    if pending is None:
        return {"pending": False, "file": None}
    return {"pending": True, "file": pending["key"]}


@app.get("/api/edit/ai-edit/image")
def api_ai_edit_image(side: str = Query(...)) -> FileResponse:
    """Serve the "before" (what the edit started from) or "after" (the
    pending, not-yet-accepted result) image of the current AI edit."""
    pending = _require_pending_edit()
    if side == "before":
        path = pending["before_path"]
    elif side == "after":
        path = pending["pending_path"]
    else:
        raise HTTPException(status_code=400, detail="side must be 'before' or 'after'")
    if path is None or not Path(path).is_file():
        raise HTTPException(status_code=404, detail=f"No {side} image available")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.post("/api/edit/ai-edit/accept")
def api_ai_edit_accept() -> dict:
    """Promote the pending edit to the photo's edited_image, replacing any
    previously accepted edit."""
    pending = _require_pending_edit()
    album_dir: Path = pending["album_dir"]
    final_path: Path = pending["final_path"]
    key: str = pending["key"]

    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(pending["pending_path"], final_path)
    try:
        album = album_for(album_dir)
        image = album.image(key)
        if image is not None:
            image.set_edited_image(final_path)
            image.set_edited_crop(pending["crop"], pending["crop_px"])
            album.save()
            # The annotated preview shows the cropped-out area darkened, so
            # redraw it with the crop that was just recorded (or dropped).
            _invalidate_cached_previews(album_dir, [key])
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("AI edit accepted but failed to record edited_image for %s: %s", key, exc)
    ai_edit_state.clear(delete_files=True)
    log.info("AI edit accepted for %s -> %s", key, final_path)
    return {"ok": True}


@app.post("/api/edit/ai-edit/reject")
def api_ai_edit_reject() -> dict:
    """Discard the pending edit, deleting the generated image. Whatever the
    photo's edited_image was before the run is left untouched."""
    pending = _require_pending_edit()
    ai_edit_state.clear(delete_files=True)
    log.info("AI edit rejected for %s", pending["key"])
    return {"ok": True}


# Where "Save Copy" keeps the processed photos, inside the album folder.
_SAVE_COPY_SUBDIR = "SaveCopy"
_save_copy_lock = threading.Lock()


def _next_copy_path(copy_dir: Path, stem: str) -> Path:
    """``copy_dir/<stem>_<N>.jpg`` with N one past the highest number already
    used for *stem* (1 for the first copy), so a deleted copy's number is not
    handed out again; also never an existing file."""
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$", re.IGNORECASE)
    highest = 0
    if copy_dir.is_dir():
        for existing in copy_dir.iterdir():
            match = pattern.match(existing.name)
            if match:
                highest = max(highest, int(match.group(1)))
    n = highest + 1
    while (copy_dir / f"{stem}_{n}.jpg").exists():
        n += 1
    return copy_dir / f"{stem}_{n}.jpg"


@app.post("/api/edit/ai-edit/save-copy")
def api_ai_edit_save_copy() -> dict:
    """Save the pending edit as a NEW photo instead of replacing the current
    one (the review's "Save Copy"): the processed image moves to
    <album>/SaveCopy/<stem>_<N>.jpg, a copy of the photo's album entry
    (detections re-framed to the crop) is added to album.json pointing at it,
    and it is listed in info.json next to the original. The photo being
    edited keeps whatever edit it had."""
    pending = _require_pending_edit()
    album_dir: Path = pending["album_dir"]
    key: str = pending["key"]
    album = album_for(album_dir)
    image = album.image(key)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in album")

    with _save_copy_lock:
        dest = _next_copy_path(album_dir / _SAVE_COPY_SUBDIR, Path(key).stem)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending["pending_path"], dest)

        # The copy has no EXIF, so file times are all the Culling page has to
        # place it: give it the original's capture time so it lands in the
        # same burst, right beside the photo it came from.
        original = image.original_path
        if original is not None and original.is_file():
            capture = image_capture_timestamp(original)
            if capture:
                os.utime(dest, (capture, capture))

        new_key = album.unused_key(dest.name, dest)
        entry = image.copy_entry(file=dest, key=new_key, crop=pending["crop"])
        album.add_entry(entry)

        info = AlbumInfo(album_dir)
        if info.exists:
            category = category_for_status(entry.get("status", "")) or info.category_of(key)
            if category:
                info.add(category, new_key, dest)
                info.save()

    ai_edit_state.clear(delete_files=True)
    log.info("AI edit of %s saved as a copy: %s (key %s)", key, dest, new_key)
    return {"ok": True, "key": new_key, "path": str(dest)}


def _require_edit_idle() -> None:
    """409 while a background job (e.g. an AI edit render) is running —
    the Other tab's actions must not race it over the edited file."""
    with processing_state.lock:
        if processing_state.running:
            raise HTTPException(status_code=409, detail="Processing is already running.")


@app.post("/api/edit/revert")
def api_edit_revert(req: AiEditRequest) -> dict:
    """Other tab → Revert: irreversibly discard *req.file*'s accepted edit
    (deleting the edited file and forgetting it, crop included) so the photo
    is its original again. The panel asks for confirmation first, then
    reopens the session on the original."""
    _require_edit_idle()
    album_dir = _current_album_path()
    album = album_for(album_dir)
    image = album.image(req.file)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in album")
    pending = ai_edit_state.snapshot()
    if pending is not None and pending["key"] == req.file:
        ai_edit_state.clear(delete_files=True)

    edited = image.clear_edited()
    album.save()
    _invalidate_cached_previews(album_dir, [req.file])  # redraw without the crop shading
    deleted = False
    # Only ever delete inside the album's editedimages folder.
    if edited is not None and edited.parent == album_dir / _EDITEDIMAGES_SUBDIR and edited.is_file():
        edited.unlink()
        deleted = True
    edit_session_state.clear()
    log.info("Edit reverted for %s (deleted %s)", req.file, edited if deleted else "nothing")
    return {"ok": True, "deleted": deleted}


def _ensure_edited_copy(album_dir: Path, key: str) -> tuple[Path, bool]:
    """*key*'s edited file, creating it from the original first if the photo
    has no edit yet (JPEGs copied byte for byte, anything else — RAW, TIFF,
    PNG — decoded like autoedit.py does and saved as JPEG) and recording it
    as the accepted edit. Returns (path, whether it was just created)."""
    album = album_for(album_dir)
    image = album.image(key)
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found in album")
    edited = image.edited_path
    if edited is not None:
        return edited, False

    source = image.original_path
    if source is None or not source.is_file():
        raise HTTPException(status_code=404, detail="Source image not found on disk")
    final = album_dir / _EDITEDIMAGES_SUBDIR / f"{Path(key).stem}.jpg"
    final.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(source, final)
    else:
        import autoedit  # lazy: pulls in the CV stack

        decoded = autoedit.read_image(source)
        if decoded is None or not autoedit.write_image(decoded, final, "jpg", 95):
            raise HTTPException(status_code=422, detail=f"Could not convert {source.name} to JPEG")
    image.set_edited_image(final)
    image.set_edited_crop(None)
    album.save()
    log.info("Created edited copy of %s at %s", key, final)
    return final, True


def _open_with_chooser(path: Path) -> None:
    """Show Windows' "Open with" app chooser for *path* (ShellExecute's
    "openas" verb). Like _open_in_file_explorer this only makes sense with
    the server on the user's own desktop."""
    try:
        os.startfile(str(path), "openas")  # noqa: S606 — local-machine convenience only
    except OSError:
        subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", str(path)])


@app.post("/api/edit/open-external")
def api_edit_open_external(req: AiEditRequest) -> dict:
    """Other tab → Edit with custom app: make sure *req.file* has an edited
    copy (see _ensure_edited_copy) and open THAT — never the original — with
    an app the user picks. Unapplied panel settings are not included."""
    if sys.platform != "win32":
        raise HTTPException(status_code=501, detail="Editing with another app is only supported on Windows.")
    _require_edit_idle()
    album_dir = _current_album_path()
    path, created = _ensure_edited_copy(album_dir, req.file)
    try:
        _open_with_chooser(path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not open {path.name}: {exc}") from exc
    return {"ok": True, "path": str(path), "created": created}


@app.get("/api/processing-output")
def api_processing_output(since: int = 0) -> dict:
    with processing_state.lock:
        new_lines = processing_state.lines[since:]
        return {
            "lines": new_lines,
            "next": since + len(new_lines),
            "running": processing_state.running,
            "returnCode": processing_state.return_code,
            "result": processing_state.result,
        }


@app.get("/api/faces/data")
def api_faces_data() -> dict:
    album_path = _current_album_path()
    return {
        "album": _read_album_summary(album_path),
        "names": _collect_names(album_path),
        "clusters": _build_clusters(album_path),
    }


@app.get("/api/faces/thumb/{cluster}/{crop}")
def api_faces_thumb(cluster: str, crop: str) -> FileResponse:
    album_path = _current_album_path()
    fr = _facereco_dir(album_path)
    if fr is None:
        raise HTTPException(status_code=404, detail="No .FaceReco")
    fp = fr / _safe_component(cluster) / FACE_SUBDIR / _safe_component(crop)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(fp)


@app.get("/api/original/{file:path}")
def api_original(file: str) -> FileResponse:
    """Current photo for *file* — shared by the faces page's face-crop
    context view and the culling page's anno/original toggle (see
    /api/anno_img for the annotated counterpart). An accepted AI edit always
    wins over the original, so the user sees the current state of the photo.
    `no-cache` makes the browser revalidate every time: the same URL serves a
    different file once an edit is accepted or reverted."""
    album_path = _current_album_path()
    image = album_for(album_path).image(file)
    headers = {"Cache-Control": "no-cache"}
    if image is not None:
        fp = image.image_path
        if fp is not None and fp.is_file():
            return FileResponse(fp, headers=headers)
    # Fall back to the legacy single-SrcDir + basename join for albums
    # written before multi-source-directory import support existed.
    src_dir = _read_album_summary(album_path).get("srcDir")
    if not src_dir:
        raise HTTPException(status_code=404, detail="Source directory unknown")
    fp = Path(src_dir) / _safe_component(file)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Original not found")
    return FileResponse(fp, headers=headers)


@app.post("/api/faces/commit")
def api_faces_commit(req: FacesCommitRequest) -> dict:
    album_path = _current_album_path()
    return _commit_cluster_operations(album_path, req)


@app.get("/api/players/data")
def api_players_data(teamId: str = "") -> dict:
    teams_file = _load_teams_file()
    if not teams_file.Teams:
        raise HTTPException(status_code=409, detail="No team exists yet — create one on the Team page first.")
    team = _find_team(teams_file, teamId) if teamId else None
    if team is None:
        team = _default_players_team(teams_file)
    return {
        "teams": [{"id": t.id, "name": t.name} for t in teams_file.Teams],
        "teamId": team.id,
        "players": [_player_view(p) for p in _team_people(team)],
    }


@app.get("/api/players/img/{team_id}/{folder}/{image_set}/{crop}")
def api_players_img(team_id: str, folder: str, image_set: str, crop: str) -> FileResponse:
    team = _team_or_404(team_id)
    if image_set not in PLAYER_IMAGE_SETS:
        raise HTTPException(status_code=404, detail="Unknown image set")
    fp = _team_face_db_path(team) / _safe_component(folder) / image_set / _safe_component(crop)
    if not fp.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(fp)


@app.post("/api/players/commit")
def api_players_commit(req: PlayersCommitRequest) -> dict:
    team = _team_or_404(req.teamId)
    with players_lock:
        _ensure_face_db_idle()
        _apply_player_ops(_find_person(team, req.name), req.ops)
        return {"player": _player_view(_find_person(team, req.name))}


@app.post("/api/players/rebuild")
def api_players_rebuild(req: PlayersRebuildRequest) -> dict:
    """Flush the team's Deleted/ crops and re-run RebuildFaceDB.py (with
    calibration) in the background; progress via /api/players/rebuild/status."""
    team = _team_or_404(req.teamId)
    team_dir = _team_face_db_path(team)
    if not team_dir.is_dir():
        raise HTTPException(status_code=400, detail="This team has no face database yet.")
    with players_lock:
        if not face_db_rebuild_state.claim():
            raise HTTPException(status_code=409, detail="A face DB rebuild is already running.")
        with export_state.lock:
            exporting = export_state.running
        if exporting:
            face_db_rebuild_state.mark_stopped()
            raise HTTPException(status_code=409, detail="An export is updating the face database — try again when it finishes.")
        with face_db_rebuild_state.lock:
            face_db_rebuild_state.team_id = team.id
    threading.Thread(target=_run_face_db_rebuild, args=(team_dir,), daemon=True).start()
    return {"started": True}


@app.get("/api/players/rebuild/status")
def api_players_rebuild_status(since: int = 0) -> dict:
    return face_db_rebuild_state.status(since)


@app.get("/api/album-tools/summary")
def api_album_tools_summary() -> dict:
    album_path = _current_album_path()
    info = AlbumInfo(album_path)
    total_images = info.total
    kept = len(_kept_image_basenames(album_path))
    clusters = _build_clusters(album_path)

    album = album_for(album_path)
    star_breakdown = {str(n): 0 for n in range(5, 0, -1)}
    unrated = 0
    for image in album.image_list:
        key = str(image.stars)
        if key in star_breakdown:
            star_breakdown[key] += 1
        else:
            unrated += 1

    raw_sensitivity = str(album.run_settings.get("sensitivity", "medium"))
    if raw_sensitivity in SENSITIVITY_PRESETS:
        sensitivity_mode, sensitivity_custom_value = raw_sensitivity, 0.50
    else:
        try:
            sensitivity_mode, sensitivity_custom_value = "custom", float(raw_sensitivity)
        except ValueError:
            sensitivity_mode, sensitivity_custom_value = "medium", 0.50

    return {
        "name": album_path.name,
        "imagesKept": kept,
        "imagesDropped": max(0, total_images - kept),
        "blurCount": info.count("blur"),
        "sharpCount": info.count("sharp"),
        "facesDetected": sum(len(c["faces"]) for c in clusters),
        "playersDetected": sum(1 for c in clusters if not c["pending"]),
        "starBreakdown": star_breakdown,
        "unrated": unrated,
        "sensitivityMode": sensitivity_mode,
        "sensitivityCustomValue": sensitivity_custom_value,
    }


@app.post("/api/album-tools/regrade-sensitivity")
def api_regrade_sensitivity(req: RegradeSensitivityRequest) -> dict:
    """Re-grade the current album's Blur/Sharp verdicts at a NEW sensitivity.

    ``mode="shallow"`` re-buckets from the per-body sharpness scores already
    stored on the Album (see algo/regrade.py). It re-reads a source photo only
    to measure a jersey colour that was never recorded, which can take a
    while on a big album, so it runs as a background job like the deep one.

    ``mode="deep"`` re-runs the FULL analysis pipeline over the album's
    source photos (1_prep_review.py --regrade-only) in a background thread.

    Either way progress streams through the shared processing_state that
    /api/processing-output serves (a shallow regrade's summary arrives as
    that endpoint's ``result``), and this mutates the Album + info.json, so
    it's blocked while any other operation touching the same album is
    running."""
    album_path = _current_album_path()
    with processing_state.lock:
        if processing_state.running:
            raise HTTPException(status_code=409, detail="An import/re-run is already in progress.")
    with export_state.lock:
        if export_state.running:
            raise HTTPException(status_code=409, detail="An export is already in progress.")

    threshold = _sensitivity_threshold(req.sensitivityMode, req.sensitivityCustomValue)

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")

    if req.mode == "deep":
        threading.Thread(
            target=_run_deep_regrade,
            args=(album_path, _sensitivity_arg(req.sensitivityMode, req.sensitivityCustomValue)),
            daemon=True,
        ).start()
        return {"mode": "deep", "started": True, "threshold": threshold}

    threading.Thread(target=_run_quick_regrade, args=(album_path, threshold), daemon=True).start()
    return {"mode": "shallow", "started": True, "threshold": threshold}


@app.get("/api/album-tools/jersey-options")
def api_jersey_options() -> dict:
    """Colours offerable as a manual team-colour pin for the current album:
    the album team's own registered jersey colours, minus the forced ones
    (goalkeeper/alternate kits, which always pass regardless of the team
    colour and so make no sense as the team colour itself)."""
    album_path = _current_album_path()
    album = album_for(album_path)
    run_settings = album.run_settings
    team = _resolve_team(_album_team_id(album_path))
    return {
        "options": [jc.color for jc in team.jerseyColors if not jc.forced],
        "current": (run_settings.get("team_color_override") or ""),
        "detected": album.our_jersey_color,
        "noTeam": bool(run_settings.get("noteam")),
    }


@app.post("/api/album-tools/jersey-color")
def api_set_jersey_color(req: JerseyColorRequest) -> dict:
    """Pin (or un-pin) the album's team jersey colour and propagate it.

    Re-runs the shallow regrade at the album's current sensitivity with the
    new colour, which re-applies the jersey filter to every eligible body and
    rewrites verdicts, info.json buckets, star baselines and previews. Face
    recognition is deliberately NOT re-run: FaceRecoPipeline selects bodies
    purely on sharpness_score, so jersey filtering cannot change its input.

    When ``rerunLlmCulling`` is set, LLM burst ranking is re-run afterwards
    in the same background job (burst membership follows the sharp set).
    Runs as a background job so its log streams to the page; the summary
    arrives as /api/processing-output's ``result``."""
    album_path = _current_album_path()
    with processing_state.lock:
        if processing_state.running:
            raise HTTPException(status_code=409, detail="An import/re-run is already in progress.")
    with export_state.lock:
        if export_state.running:
            raise HTTPException(status_code=409, detail="An export is already in progress.")

    run_settings = album_for(album_path).run_settings
    if run_settings.get("noteam"):
        raise HTTPException(
            status_code=400,
            detail="This album was imported with jersey-colour filtering disabled.",
        )

    threshold = _album_sharpness_threshold(run_settings)

    if not processing_state.claim():
        raise HTTPException(status_code=409, detail="Processing is already running.")
    threading.Thread(
        target=_run_quick_regrade,
        args=(album_path, threshold, req.teamColor, req.rerunLlmCulling),
        daemon=True,
    ).start()
    return {"started": True, "threshold": threshold, "llmRerunQueued": bool(req.rerunLlmCulling)}


@app.post("/api/album-tools/export")
def api_start_export(req: ExportRequest) -> dict:
    album_path = _current_album_path()
    destination = req.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination folder is required.")
    dest_path = Path(destination)
    if dest_path.is_file():
        raise HTTPException(status_code=400, detail="Destination is a file, not a folder.")

    if not export_state.claim():
        raise HTTPException(status_code=409, detail="Export is already running.")
    with face_db_rebuild_state.lock:
        rebuild_running = face_db_rebuild_state.running
    if rebuild_running:
        export_state.mark_stopped()
        raise HTTPException(status_code=409, detail="A face DB rebuild is running — try again when it finishes.")

    min_stars = max(1, min(5, req.minStars))
    thread = threading.Thread(
        target=_run_export,
        args=(album_path, dest_path, req.exportFaceTagging, min_stars),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.get("/api/album-tools/export-status")
def api_export_status(since: int = 0) -> dict:
    return export_state.status(since)


@app.post("/api/album-tools/face-tagging-csv/render")
def api_face_tagging_csv_render(req: FaceTaggingCsvRequest) -> dict:
    """Write a CSV of every tagged face in the current album's .FaceReco
    database to a user-chosen destination folder — unlike players.csv
    (written only for kept/exported photos during a folder export), this
    covers the complete face-tagging record regardless of star rating."""
    album_path = _current_album_path()
    destination = req.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination folder is required.")
    dest_path = Path(destination)
    if dest_path.is_file():
        raise HTTPException(status_code=400, detail="Destination is a file, not a folder.")

    if not face_tagging_csv_state.claim():
        raise HTTPException(status_code=409, detail="A face-tagging CSV export is already running.")

    thread = threading.Thread(target=_run_face_tagging_csv_export, args=(album_path, dest_path), daemon=True)
    thread.start()
    return {"ok": True}


@app.get("/api/album-tools/face-tagging-csv/status")
def api_face_tagging_csv_status(since: int = 0) -> dict:
    return face_tagging_csv_state.status(since)


@app.post("/api/album-tools/face-tagging-csv/open-folder")
def api_face_tagging_csv_open_folder() -> dict:
    with face_tagging_csv_state.lock:
        output_dir = face_tagging_csv_state.output_dir
    if not output_dir or not Path(output_dir).is_dir():
        raise HTTPException(status_code=400, detail="No output folder yet — export at least once first.")
    _open_in_file_explorer(Path(output_dir))
    return {"ok": True}


@app.get("/api/album-tools/player-video/summary")
def api_player_video_summary() -> dict:
    album_path = _current_album_path()
    player_frames = _collect_player_frame_paths(album_path)
    return {
        "playersFound": len(player_frames),
        "totalPhotos": sum(len(v) for v in player_frames.values()),
        "ffmpegAvailable": shutil.which("ffmpeg") is not None,
    }


@app.post("/api/album-tools/player-video/render")
def api_player_video_render(req: PlayerVideoRenderRequest) -> dict:
    album_path = _current_album_path()
    destination = req.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination folder is required.")
    dest_path = Path(destination)
    if dest_path.is_file():
        raise HTTPException(status_code=400, detail="Destination is a file, not a folder.")

    if not player_video_state.claim():
        raise HTTPException(status_code=409, detail="Player-video rendering is already running.")

    thread = threading.Thread(
        target=_run_player_video,
        args=(album_path, dest_path, max(0.1, req.fps)),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.get("/api/album-tools/player-video/status")
def api_player_video_status(since: int = 0) -> dict:
    return player_video_state.status(since)


@app.post("/api/album-tools/player-video/open-folder")
def api_player_video_open_folder() -> dict:
    with player_video_state.lock:
        output_dir = player_video_state.output_dir
    if not output_dir or not Path(output_dir).is_dir():
        raise HTTPException(status_code=400, detail="No output folder yet — render at least once first.")
    _open_in_file_explorer(Path(output_dir))
    return {"ok": True}


@app.post("/api/album-tools/open-destination")
def api_open_export_destination() -> dict:
    """Open the just-exported destination folder in the OS's native file
    explorer (Windows Explorer / macOS Finder / Linux file manager)."""
    with export_state.lock:
        dest_dir = export_state.dest_dir
    if not dest_dir or not Path(dest_dir).is_dir():
        raise HTTPException(status_code=400, detail="Destination folder not found.")
    _open_in_file_explorer(Path(dest_dir))
    return {"ok": True}


@app.get("/api/album-tools/burst-video/summary")
def api_burst_video_summary(minFrames: int = Query(DEFAULT_MIN_FRAMES)) -> dict:
    album_path = _current_album_path()
    return summarize_bursts(album_path, min_frames=max(2, minFrames))


@app.get("/api/album-tools/burst-video/settings")
def api_burst_video_settings() -> dict:
    """Last-used min-frames/fps/timing-mode render settings, persisted in
    state.json (see api_burst_video_render below) so the Album Tools page
    pre-fills them on load instead of always resetting to the hardcoded
    defaults."""
    state = _load_state()
    return {
        "minFrames": state.get("burstMinFrames", DEFAULT_MIN_FRAMES),
        "fps": state.get("burstFps", DEFAULT_FPS),
        "mode": state.get("burstTimingMode", "fps"),
    }


@app.post("/api/album-tools/burst-video/render")
def api_burst_video_render(req: BurstVideoRenderRequest) -> dict:
    album_path = _current_album_path()
    if req.mode not in ("fps", "timestamp"):
        raise HTTPException(status_code=400, detail="mode must be 'fps' or 'timestamp'")
    destination = req.destination.strip()
    if not destination:
        raise HTTPException(status_code=400, detail="Destination folder is required.")
    dest_path = Path(destination)
    if dest_path.is_file():
        raise HTTPException(status_code=400, detail="Destination is a file, not a folder.")

    min_frames = max(2, req.minFrames)
    fps = max(0.1, req.fps)
    state = _load_state()
    state["burstMinFrames"] = min_frames
    state["burstFps"] = fps
    state["burstTimingMode"] = req.mode
    _save_state(state)

    if not burst_video_state.claim():
        raise HTTPException(status_code=409, detail="Burst-video rendering is already running.")

    thread = threading.Thread(
        target=_run_burst_video,
        args=(album_path, dest_path, min_frames, req.mode, fps),
        daemon=True,
    )
    thread.start()
    return {"ok": True}


@app.get("/api/album-tools/burst-video/status")
def api_burst_video_status(since: int = 0) -> dict:
    return burst_video_state.status(since)


@app.post("/api/album-tools/burst-video/open-folder")
def api_burst_video_open_folder() -> dict:
    with burst_video_state.lock:
        output_dir = burst_video_state.output_dir
    if not output_dir or not Path(output_dir).is_dir():
        raise HTTPException(status_code=400, detail="No output folder yet — render at least once first.")
    _open_in_file_explorer(Path(output_dir))
    return {"ok": True}


if (WEBUI_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEBUI_DIR / "static"), name="static")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Photo culling & organization web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open a browser tab")
    parser.add_argument("--heartbeat-timeout", type=float, default=HEARTBEAT_TIMEOUT_SECONDS,
                        help="Seconds without a browser heartbeat before the "
                             "server logs a reminder (with the relaunch link) "
                             "that it's still running. Use 0 to disable "
                             f"(default: {HEARTBEAT_TIMEOUT_SECONDS:g}).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("uvicorn.access").addFilter(_HeartbeatAccessLogFilter())

    url = f"http://{args.host}:{args.port}/"
    heartbeat_state.timeout = args.heartbeat_timeout
    heartbeat_state.last_heartbeat = time.time()
    heartbeat_state.url = url

    log.info("team.json:  %s", TEAM_JSON_PATH)
    log.info("state file: %s", STATE_JSON_PATH)
    log.info("Open %s in your browser", url)

    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if heartbeat_state.timeout > 0:
        log.info("Heartbeat watchdog: notify after %.0fs without a browser ping", heartbeat_state.timeout)
        threading.Thread(target=_heartbeat_watchdog, daemon=True).start()
    else:
        log.info("Heartbeat watchdog disabled")

    threading.Thread(target=_warm_preview_renderer, daemon=True).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
