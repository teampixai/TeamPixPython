"use strict";

// Page 5 — Tools & More. Two groups of tools for the current album:
//   Photo Utilities:   Batch Auto Edit, per-player highlight videos, Burst -> MP4.
//   Album Maintenance: Adjust Blur Sensitivity (quick/deep regrade), Change Jersey
//                      Color, Re-run Face Detection.
//
// Every long-running action below shares the same "launch job -> poll server
// log into a modal -> react when done" UX via LogDialog (see common.js) —
// each `*Job` is created once in the init block at the bottom of this file.
let deepRegradeJob, quickRegradeJob, jerseyColorJob, rerunFacerecoJob;
let burstVideoJob, playerVideoJob, batchAutoEditJob;

let rerunFacerecoRunning = false;

let regrading = false;

let burstVideoRunning = false;

// Regrading, changing the jersey colour and re-running face detection all
// rewrite the album, so keep them mutually exclusive from the UI side (each is
// also guarded server-side via the shared processing_state lock).
function setRerunFacerecoUIEnabled(enabled) {
  rerunFacerecoRunning = !enabled;
  $("#rerunFacerecoBtn").prop("disabled", !enabled);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled);
}

// ------------------------------------------------------------------ //
// Adjust Blur Sensitivity — re-bucket Blur/Sharp using already-measured
// sharpness scores (see /api/album-tools/regrade-sensitivity), no re-import.
// ------------------------------------------------------------------ //
function applyRegradeSensitivityToUI(mode, customValue) {
  $(`input[name="regradeSensMode"][value="${mode || "medium"}"]`).prop("checked", true);
  const custom = customValue ?? 0.50;
  $("#regradeSensitivitySlider").val(custom);
  $("#regradeSensitivityValue").text(Number(custom).toFixed(2));
}

function readRegradeSensitivityFromUI() {
  const mode = $('input[name="regradeSensMode"]:checked').val() || "medium";
  const customValue = parseFloat($("#regradeSensitivitySlider").val()) || 0;
  return { mode, customValue };
}

function setRegradeUIEnabled(enabled) {
  regrading = !enabled;
  $('input[name="regradeSensMode"], #regradeSensitivitySlider').prop("disabled", !enabled);
  $("#regradeBtn, #deepRegradeBtn").prop("disabled", !enabled || rerunFacerecoRunning);
  $("#rerunFacerecoBtn").prop("disabled", !enabled);
}

// Deep regrade streams 1_prep_review.py's stdout into a non-closable modal
// (deepRegradeJob, created in the init block) that dismisses itself once
// the run finishes.

// Re-run-face-detection progress dialog (rerunFacerecoJob, created in the
// init block) — same pattern as the other jobs.

async function loadSensitivity() {
  try {
    const data = await apiGet("/api/album-tools/summary");
    applyRegradeSensitivityToUI(data.sensitivityMode, data.sensitivityCustomValue);
  } catch (err) {
    // Non-fatal — the controls just stay at their defaults.
  }
}

// ------------------------------------------------------------------ //
// Change Jersey Color — see /api/album-tools/jersey-color. Rendered as
// round chip buttons ( "Auto" + each registered colour), same interaction
// pattern as the team/jersey-colour pickers on the Add Album page.
// ------------------------------------------------------------------ //
let jerseyColorOptions = [];
let selectedJerseyColor = "";
let jerseyColorControlsDisabled = false;

// Non-closable modal (jerseyColorJob, created in the init block) streaming
// this operation's output (mirrors deep regrade) — auto-dismisses on
// success, stays open with [x] enabled if it fails so the user can read
// the error.

function renderJerseyColorChips() {
  const $container = $("#jerseyColorPicker").empty();
  const makeChip = (label, value) => {
    const chip = $("<button>", {
      type: "button",
      class: "color-chip" + (selectedJerseyColor === value ? " selected" : ""),
      disabled: jerseyColorControlsDisabled,
    }).text(label);
    chip.on("click", () => { selectedJerseyColor = value; renderJerseyColorChips(); });
    $container.append(chip);
  };
  makeChip("Auto (detect)", "");
  jerseyColorOptions.forEach((color) => makeChip(color, color));
}

async function loadJerseyOptions() {
  try {
    const data = await apiGet("/api/album-tools/jersey-options");
    jerseyColorOptions = data.options || [];
    selectedJerseyColor = data.current || "";
    jerseyColorControlsDisabled = !!data.noTeam;
    renderJerseyColorChips();
    if (data.noTeam) {
      $("#jerseyColorPanel .row, #jerseyColorPanel .row.actions").find("input, button").prop("disabled", true);
      $("#jerseyColorStatus").text("This album has jersey-colour filtering disabled (--noteam).");
    }
    $("#jerseyColorDetected").text(
      data.current ? "" : (data.detected ? `Currently detected: ${data.detected}` : "")
    );
  } catch (err) {
    // Non-fatal — the panel just stays at its default "Auto" state.
  }
}

// ------------------------------------------------------------------ //
// Burst → MP4 (experimental) — render each detected photo burst (incl.
// blur/skipped shots) into its own MP4 via /api/album-tools/burst-video/*,
// saved to a user-chosen destination folder (same /api/browse-folder
// picker as Export/per-player videos below). Same background-job/poll
// pattern as the other dialogs on this page (burstVideoJob, created in the
// init block).
// ------------------------------------------------------------------ //
function readBurstMinFrames() {
  return Math.max(2, parseInt($("#burstMinFramesInput").val(), 10) || 3);
}

// Pre-fills the min-frames/fps/timing-mode controls from the last render's
// settings (persisted server-side in state.json — see api_burst_video_render
// in culling_app.py) instead of always resetting to the HTML defaults.
async function loadBurstVideoSettings() {
  try {
    const data = await apiGet("/api/album-tools/burst-video/settings");
    $("#burstMinFramesInput").val(data.minFrames);
    $("#burstFpsInput").val(data.fps);
    const mode = data.mode === "timestamp" ? "timestamp" : "fps";
    $(`input[name="burstTimingMode"][value="${mode}"]`).prop("checked", true);
    $("#burstFpsInput").prop("disabled", mode !== "fps");
  } catch (err) {
    // Keep the HTML defaults if this fails.
  }
}

async function refreshBurstSummary() {
  try {
    const data = await apiGet(`/api/album-tools/burst-video/summary?minFrames=${readBurstMinFrames()}`);
    const ffmpegNote = data.ffmpegAvailable ? "" : " (ffmpeg not found \u2014 using fallback encoder)";
    $("#burstSummaryLabel").text(
      `${data.qualifyingBursts} of ${data.totalBursts} burst(s) qualify (${data.qualifyingFrames} photo(s))${ffmpegNote}`
    );
  } catch (err) {
    $("#burstSummaryLabel").text("");
  }
}

// ------------------------------------------------------------------ //
// Per-player highlight videos — one MP4 per tagged player via
// /api/album-tools/player-video/*. Same background-job/poll/modal pattern
// as the burst-video dialog above (playerVideoJob, created in the init
// block).
// ------------------------------------------------------------------ //
async function refreshPlayerVideoSummary() {
  try {
    const data = await apiGet("/api/album-tools/player-video/summary");
    const ffmpegNote = data.ffmpegAvailable ? "" : " (ffmpeg not found \u2014 using fallback encoder)";
    $("#playerVideoSummaryLabel").text(`${data.playersFound} player(s), ${data.totalPhotos} tagged photo(s)${ffmpegNote}`);
  } catch (err) {
    $("#playerVideoSummaryLabel").text("");
  }
}

// ------------------------------------------------------------------ //
// Batch Auto Edit (experimental) — AI Edit's Auto crop + Sharpen players
// over every photo at or above a star rating, saved to a chosen folder via
// /api/album-tools/batch-auto-edit/*. Same background-job/poll/modal
// pattern as the dialogs above (batchAutoEditJob, created in the init
// block).
// ------------------------------------------------------------------ //
let batchResizeChoice = "instagram";
let batchCount = 0;
let batchRunning = false;
let batchCountRequest = 0;

function updateBatchControls() {
  const resize = $("#batchResizeInput").prop("checked");
  const sharpen = $("#batchSharpenInput").prop("checked");
  $("#batchResizeChoices button").each(function () {
    $(this).prop("disabled", !resize || batchRunning)
      .attr("aria-pressed", String($(this).data("choice") === batchResizeChoice));
  });
  $("#batchResizeInput, #batchSharpenInput, #batchMinStarsInput, #batchPlayerInput").prop("disabled", batchRunning);
  $("#batchStartBtn").prop("disabled", batchRunning || !(resize || sharpen) || batchCount < 1);
}

// Fills the Player picker with the album's tagged players ("All Players"
// stays first and means no player filter).
async function loadBatchPlayers() {
  try {
    const data = await apiGet("/api/album-tools/batch-auto-edit/players");
    const $select = $("#batchPlayerInput");
    $select.find("option:gt(0)").remove();
    data.players.forEach((name) => $select.append($("<option>").val(name).text(name)));
  } catch (err) {
    // Keep just "All Players" if this fails.
  }
}

function batchPlayer() {
  return $("#batchPlayerInput").val() || "";
}

async function refreshBatchCount() {
  const minStars = parseInt($("#batchMinStarsInput").val(), 10) || 5;
  const player = batchPlayer();
  const request = ++batchCountRequest;
  try {
    const data = await apiGet(
      `/api/album-tools/batch-auto-edit/summary?minStars=${minStars}&player=${encodeURIComponent(player)}`
    );
    if (request !== batchCountRequest) return;
    batchCount = data.count;
    const stars = minStars === 5 ? "5 stars" : `${minStars}+ stars`;
    $("#batchCountLabel").text(
      data.count
        ? `${data.count} photo(s) will be processed`
        : (player ? `No photos of ${player} have ${stars}` : `No photos have ${stars}`)
    );
  } catch (err) {
    if (request !== batchCountRequest) return;
    batchCount = 0;
    $("#batchCountLabel").text("");
  }
  updateBatchControls();
}

// Start asks for the destination folder first (like Export); canceling the
// picker leaves everything as it was.
async function startBatchAutoEdit() {
  batchRunning = true;
  updateBatchControls();
  $("#batchStatus").text("Choose a destination folder…");
  $("#openBatchFolderBtn").hide();
  let res;
  try {
    res = await apiPost("/api/browse-folder", { title: "Select folder for the edited photos", context: "export" });
  } catch (err) {
    $("#batchStatus").text("Browse failed: " + err.message);
    batchRunning = false;
    updateBatchControls();
    return;
  }
  if (!res.path) {
    $("#batchStatus").text("");
    batchRunning = false;
    updateBatchControls();
    return;
  }
  $("#batchStatus").text("");
  batchAutoEditJob.reset();
  batchAutoEditJob.open("Batch Auto Edit…");
  const finish = () => {
    batchRunning = false;
    updateBatchControls();
  };
  try {
    await apiPost("/api/album-tools/batch-auto-edit/start", {
      destination: res.path,
      resize: $("#batchResizeInput").prop("checked") ? batchResizeChoice : null,
      sharpen: $("#batchSharpenInput").prop("checked"),
      player: batchPlayer() || null,
      minStars: parseInt($("#batchMinStarsInput").val(), 10) || 5,
    });
  } catch (err) {
    $("#batchStatus").text("Failed to start: " + err.message);
    batchAutoEditJob.close();
    finish();
    return;
  }
  batchAutoEditJob.startPolling((data) => {
    if (data.running) return;
    finish();
    const success = !data.error;
    batchAutoEditJob.setFinished(success ? "Batch complete" : "Batch failed");
    const r = data.result;
    if (success) {
      $("#batchStatus").text(r ? `Wrote ${r.processed} photo(s) (${r.cropped} cropped, ${r.sharpened} sharpened).` : "Done.");
    } else {
      $("#batchStatus").text("Failed: " + data.error);
    }
    if (r && r.processed) $("#openBatchFolderBtn").show();
  });
}

$(function () {
  loadSensitivity();
  loadJerseyOptions();

  deepRegradeJob = LogDialog.create({
    dialogId: "deepRegradeDialog",
    outputId: "deepRegradeOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    dialogOptions: { width: 720 },
    label: "deep regrade",
  });
  // No markup in the page for this one: LogDialog builds the dialog itself.
  quickRegradeJob = LogDialog.create({
    dialogId: "quickRegradeDialog",
    outputId: "quickRegradeOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    dialogOptions: { width: 720 },
    label: "quick regrade",
  });
  jerseyColorJob = LogDialog.create({
    dialogId: "jerseyColorDialog",
    outputId: "jerseyColorOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    dialogOptions: { width: 720 },
    label: "jersey colour apply",
  });
  rerunFacerecoJob = LogDialog.create({
    dialogId: "rerunFacerecoDialog",
    outputId: "rerunFacerecoOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    label: "face detection re-run",
  });
  burstVideoJob = LogDialog.create({
    dialogId: "burstVideoDialog",
    outputId: "burstVideoOutput",
    statusUrl: (since) => `/api/album-tools/burst-video/status?since=${since}`,
    label: "burst-video rendering",
  });
  playerVideoJob = LogDialog.create({
    dialogId: "playerVideoDialog",
    outputId: "playerVideoOutput",
    statusUrl: (since) => `/api/album-tools/player-video/status?since=${since}`,
    label: "player-video rendering",
  });
  batchAutoEditJob = LogDialog.create({
    dialogId: "batchAutoEditDialog",
    outputId: "batchAutoEditOutput",
    statusUrl: (since) => `/api/album-tools/batch-auto-edit/status?since=${since}`,
    dialogOptions: { width: 720 },
    label: "batch auto edit",
  });

  loadBatchPlayers();
  refreshBatchCount();
  $("#batchResizeInput, #batchSharpenInput").on("change", updateBatchControls);
  $("#batchMinStarsInput, #batchPlayerInput").on("change", refreshBatchCount);
  $("#batchResizeChoices").on("click", "button", function () {
    batchResizeChoice = $(this).data("choice");
    updateBatchControls();
  });
  $("#batchStartBtn").on("click", startBatchAutoEdit);
  $("#openBatchFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/batch-auto-edit/open-folder", {});
    } catch (err) {
      $("#batchStatus").text("Could not open folder: " + err.message);
    }
  });

  loadBurstVideoSettings().then(refreshBurstSummary);
  refreshPlayerVideoSummary();

  $("#burstMinFramesInput").on("change", refreshBurstSummary);

  $('input[name="burstTimingMode"]').on("change", function () {
    $("#burstFpsInput").prop("disabled", $('input[name="burstTimingMode"]:checked').val() !== "fps");
  });

  $("#openBurstsFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/burst-video/open-folder", {});
    } catch (err) {
      $("#burstVideoStatus").text("Could not open folder: " + err.message);
    }
  });

  $("#renderBurstsBtn").on("click", async () => {
    $("#renderBurstsBtn").prop("disabled", true);
    $("#burstVideoStatus").text("Choose a destination folder…");
    $("#openBurstsFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for burst videos", context: "export" });
    } catch (err) {
      $("#burstVideoStatus").text("Browse failed: " + err.message);
      $("#renderBurstsBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#burstVideoStatus").text("");
      $("#renderBurstsBtn").prop("disabled", false);
      return;
    }

    burstVideoRunning = true;
    $("#burstVideoStatus").text("");
    burstVideoJob.reset();
    burstVideoJob.open("Rendering bursts\u2026");
    const mode = $('input[name="burstTimingMode"]:checked').val() || "fps";
    const fps = parseFloat($("#burstFpsInput").val()) || 6;
    const minFrames = readBurstMinFrames();
    try {
      await apiPost("/api/album-tools/burst-video/render", { destination: res.path, minFrames, mode, fps });
      burstVideoJob.startPolling((data) => {
        if (data.running) return;
        burstVideoRunning = false;
        $("#renderBurstsBtn").prop("disabled", false);
        const success = !data.error;
        burstVideoJob.setFinished(success ? "Rendering complete" : "Rendering failed");
        if (success) {
          $("#burstVideoStatus").text(data.result ? `Rendered ${data.result.rendered} video(s).` : "Done.");
          $("#openBurstsFolderBtn").show();
        } else {
          $("#burstVideoStatus").text("Failed: " + data.error);
        }
        refreshBurstSummary();
      });
    } catch (err) {
      $("#burstVideoStatus").text("Failed to start: " + err.message);
      $("#renderBurstsBtn").prop("disabled", false);
      burstVideoRunning = false;
      burstVideoJob.close();
    }
  });

  $("#regradeSensitivitySlider").on("input", function () {
    $("#regradeSensitivityValue").text(Number($(this).val()).toFixed(2));
  });

  // Shown in a job's log when it ends without a usable result, so the page says
  // what it actually saw instead of only "see the log".
  function describeMissingResult(job, data) {
    job.appendLines([
      `The job finished with exit code ${data.returnCode ?? "none"} and returned no result.`,
      "If the app was just updated, restart the culling server: it may still be running the old code.",
    ]);
  }

  $("#regradeBtn").on("click", async () => {
    setRegradeUIEnabled(false);
    $("#regradeStatus").text("Regrading…");
    quickRegradeJob.reset();
    quickRegradeJob.open("Quick regrade…");
    const { mode, customValue } = readRegradeSensitivityFromUI();
    const finish = (data) => {
      setRegradeUIEnabled(true);
      const res = data.result;
      if (data.returnCode === 0 && res) {
        const parts = [];
        if (res.recovered) parts.push(`${res.recovered} moved to Sharp`);
        if (res.demoted) parts.push(`${res.demoted} moved to Blur`);
        if (res.teamColor) parts.push(`team colour ${res.teamColor}`);
        if (res.teammates) parts.push(`${res.teammates} teammate(s) (+${res.teammatesAdded} / −${res.teammatesRemoved})`);
        $("#regradeStatus").text(
          parts.length
            ? `Regraded ${res.imagesConsidered} image(s) — ${parts.join(", ")}.`
            : `Regraded ${res.imagesConsidered} image(s) — no changes.`
        );
        quickRegradeJob.close();
        loadSensitivity();
      } else {
        $("#regradeStatus").text("Regrade failed — see the log.");
        if (!res) describeMissingResult(quickRegradeJob, data);
        quickRegradeJob.setFinished("Quick regrade failed");
      }
    };
    let started;
    try {
      started = await apiPost("/api/album-tools/regrade-sensitivity", {
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        mode: "shallow",
      });
    } catch (err) {
      $("#regradeStatus").text("Regrade failed to start: " + err.message);
      quickRegradeJob.appendLines(["Error: " + err.message]);
      quickRegradeJob.setFinished("Quick regrade failed");
      setRegradeUIEnabled(true);
      return;
    }
    // A server that predates background regrades answers with the finished
    // summary right away instead of starting a job: accept that as the result.
    if (!started.started) {
      finish({ running: false, returnCode: 0, result: started });
      return;
    }
    quickRegradeJob.startPolling((data) => {
      if (data.running) return;
      finish(data);
    });
  });

  $("#deepRegradeBtn").on("click", async () => {
    setRegradeUIEnabled(false);
    $("#regradeStatus").text("Starting deep regrade…");
    deepRegradeJob.reset();
    const { mode, customValue } = readRegradeSensitivityFromUI();
    try {
      await apiPost("/api/album-tools/regrade-sensitivity", {
        sensitivityMode: mode,
        sensitivityCustomValue: customValue,
        mode: "deep",
      });
      deepRegradeJob.open();
      $("#regradeStatus").text("Deep regrade running…");
      deepRegradeJob.startPolling((data) => {
        if (data.running) return;
        setRegradeUIEnabled(true);
        deepRegradeJob.close();
        if (data.returnCode === 0) {
          $("#regradeStatus").text("Deep regrade complete.");
          loadSensitivity();
        } else {
          $("#regradeStatus").text(`Deep regrade failed (exit code ${data.returnCode}).`);
        }
      });
    } catch (err) {
      $("#regradeStatus").text("Deep regrade failed to start: " + err.message);
      setRegradeUIEnabled(true);
    }
  });

  $("#jerseyColorBtn").on("click", async () => {
    const teamColor = selectedJerseyColor;
    const rerunLlmCulling = $("#jerseyRerunLlmInput").is(":checked");
    const reenable = () => {
      $("#jerseyColorBtn, #jerseyRerunLlmInput").prop("disabled", false);
      jerseyColorControlsDisabled = false;
      renderJerseyColorChips();
      setRegradeUIEnabled(true);
    };
    $("#jerseyColorBtn, #jerseyRerunLlmInput").prop("disabled", true);
    jerseyColorControlsDisabled = true;
    renderJerseyColorChips();
    setRegradeUIEnabled(false);
    $("#jerseyColorStatus").text("Applying…");
    jerseyColorJob.reset();
    jerseyColorJob.open("Applying jersey colour…");
    let started;
    try {
      started = await apiPost("/api/album-tools/jersey-color", { teamColor, rerunLlmCulling });
    } catch (err) {
      $("#jerseyColorStatus").text("Failed: " + err.message);
      jerseyColorJob.appendLines(["Error: " + err.message]);
      jerseyColorJob.setFinished("Jersey colour apply failed");
      reenable();
      return;
    }
    const finish = async (data) => {
      reenable();
      const res = data.result;
      if (data.returnCode === 0 && res) {
        const parts = [];
        if (res.recovered) parts.push(`${res.recovered} moved to Sharp`);
        if (res.demoted) parts.push(`${res.demoted} moved to Blur`);
        if (res.starsRebaselined) parts.push(`${res.starsRebaselined} star rating(s) reset`);
        const summary =
          `Team colour: ${res.teamColor || "none detected"} (${res.pinned ? "pinned" : "auto"})` +
          (parts.length ? ` — ${parts.join(", ")}.` : " — no changes.") +
          (res.llmRerun ? " LLM re-cull complete." : "");
        $("#jerseyColorStatus").text(summary);
        jerseyColorJob.close();
      } else {
        const msg = res && res.llmRerun
          ? `LLM re-cull exited with code ${data.returnCode}.`
          : "Jersey colour apply failed — see the log.";
        $("#jerseyColorStatus").text(msg);
        if (!res) describeMissingResult(jerseyColorJob, data);
        jerseyColorJob.setFinished("Jersey colour apply failed");
      }
      loadSensitivity();
      await loadJerseyOptions();
    };
    // A server that predates background jobs answers with the finished summary
    // right away (no "started"): accept that as the result.
    if (!started.started) {
      await finish({ running: false, returnCode: 0, result: started });
      return;
    }
    // The regrade (and the optional LLM re-rank after it) runs as one
    // background job; its log streams into the dialog until it is done.
    jerseyColorJob.startPolling(async (data) => {
      if (data.running) return;
      await finish(data);
    });
  });

  $("#rerunFacerecoBtn").on("click", async () => {
    setRerunFacerecoUIEnabled(false);
    $("#rerunFacerecoStatus").text("Starting…");
    rerunFacerecoJob.reset();
    rerunFacerecoJob.open("Re-running face detection…");
    $("#rerunFacerecoStatus").text("Re-running face detection…");
    try {
      await apiPost("/api/album-tools/rerun-facereco", {});
      rerunFacerecoJob.startPolling((data) => {
        if (data.running) return;
        setRerunFacerecoUIEnabled(true);
        const success = data.returnCode === 0;
        rerunFacerecoJob.setFinished(success ? "Face detection complete" : `Face detection failed (exit code ${data.returnCode})`);
        if (success) {
          $("#rerunFacerecoStatus").text("Face detection re-run complete.");
          refreshPlayerVideoSummary();
          loadBatchPlayers();
        } else {
          $("#rerunFacerecoStatus").text(`Face detection re-run exited with code ${data.returnCode}.`);
        }
      });
    } catch (err) {
      $("#rerunFacerecoStatus").text("Failed: " + err.message);
      setRerunFacerecoUIEnabled(true);
      rerunFacerecoJob.close();
    }
  });

  $("#renderPlayerVideosBtn").on("click", async () => {
    $("#renderPlayerVideosBtn").prop("disabled", true);
    $("#playerVideoStatus").text("Choose a destination folder…");
    $("#openPlayerVideoFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for player highlight videos", context: "export" });
    } catch (err) {
      $("#playerVideoStatus").text("Browse failed: " + err.message);
      $("#renderPlayerVideosBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#playerVideoStatus").text("");
      $("#renderPlayerVideosBtn").prop("disabled", false);
      return;
    }

    const fps = parseFloat($("#playerVideoFpsInput").val()) || 1;
    $("#playerVideoStatus").text("");
    playerVideoJob.reset();
    playerVideoJob.open("Rendering player videos…");
    try {
      await apiPost("/api/album-tools/player-video/render", { destination: res.path, fps });
      playerVideoJob.startPolling((data) => {
        if (data.running) return;
        $("#renderPlayerVideosBtn").prop("disabled", false);
        const success = !data.error;
        playerVideoJob.setFinished(success ? "Rendering complete" : "Rendering failed");
        if (success) {
          $("#playerVideoStatus").text(data.result ? `Rendered ${data.result.rendered} video(s).` : "Done.");
          $("#openPlayerVideoFolderBtn").show();
        } else {
          $("#playerVideoStatus").text("Failed: " + data.error);
        }
        refreshPlayerVideoSummary();
      });
    } catch (err) {
      $("#playerVideoStatus").text("Failed to start: " + err.message);
      $("#renderPlayerVideosBtn").prop("disabled", false);
      playerVideoJob.close();
    }
  });

  $("#openPlayerVideoFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/player-video/open-folder", {});
    } catch (err) {
      $("#playerVideoStatus").text("Could not open folder: " + err.message);
    }
  });
});
