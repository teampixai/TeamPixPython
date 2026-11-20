"use strict";

// Page 4 — Export: show a quick album summary (stars, faces, players), then
// let the user pick a destination folder and export kept photos
// (+ players.csv) — syncing tagged face crops into the system-wide
// .FaceReco database and rebuilding it first — with progress polling. Also
// hosts "Import more images" and the face-tagging CSV export.
//
// Every long-running action below shares the same "launch job -> poll server
// log into a modal -> react when done" UX via LogDialog (see common.js) —
// each `*Job` is created once in the init block at the bottom of this file.
let exportJob, importMoreJob, faceTaggingCsvJob;

let lastStarBreakdown = null;

let importMoreRunning = false;

// Exporting and importing more images mutate/read the album at the same time —
// keep them mutually exclusive from the UI side (each is also guarded
// server-side).
function setExportUIEnabled(enabled) {
  $("#exportFaceTaggingInput, #minStarsInput, #exportBtn").prop("disabled", !enabled);
  $("#importMoreBtn").prop("disabled", !enabled || importMoreRunning);
}

function setImportMoreUIEnabled(enabled) {
  importMoreRunning = !enabled;
  $("#importMoreBtn").prop("disabled", !enabled);
  $("#exportBtn").prop("disabled", !enabled);
}

async function openExportDestination() {
  try {
    await apiPost("/api/album-tools/open-destination", {});
  } catch (err) {
    $("#exportStatus").text("Could not open folder: " + err.message);
  }
}

const STAR_TIERS = [5, 4, 3, 2, 1];

function renderStarBreakdown(breakdown, unrated) {
  const $el = $("#starBreakdown").empty();
  breakdown = breakdown || {};
  STAR_TIERS.forEach((n) => {
    const count = breakdown[String(n)] || 0;
    const $block = $("<div>", { class: `star-block star-block-${n}` });
    $block.append($("<span>", { class: "star-icons" }).text("★".repeat(n) + "☆".repeat(5 - n)));
    $block.append($("<span>", { class: "star-count" }).text(count));
    $el.append($block);
  });
  if (unrated) {
    const $block = $("<div>", { class: "star-block star-block-unrated", title: "Unrated" });
    $block.append($("<span>", { class: "star-icons" }).text("—"));
    $block.append($("<span>", { class: "star-count" }).text(unrated));
    $el.append($block);
  }
}

async function loadSummary() {
  try {
    const data = await apiGet("/api/album-tools/summary");
    $("#summaryAlbumName").text(data.name || "Album");
    lastStarBreakdown = data.starBreakdown || {};
    renderStarBreakdown(data.starBreakdown, data.unrated);
    $("#statFaces").text(data.facesDetected);
    $("#statPlayers").text(data.playersDetected);
    updateExportCount();
  } catch (err) {
    $("#summaryAlbumName").text("Album");
  }
}

// Number of images that will be exported at the currently-selected minimum
// star rating, computed client-side from the star breakdown already loaded
// by loadSummary() — no extra round-trip needed when the selector changes.
function updateExportCount() {
  const minStars = parseInt($("#minStarsInput").val(), 10) || 3;
  let count = 0;
  if (lastStarBreakdown) {
    for (let n = minStars; n <= 5; n++) {
      count += lastStarBreakdown[String(n)] || 0;
    }
  }
  $("#exportCountLabel").text(`${count} image(s) will be exported`);
}

$(function () {
  loadSummary();

  exportJob = LogDialog.create({
    dialogId: "exportDialog",
    outputId: "exportOutput",
    statusUrl: (since) => `/api/album-tools/export-status?since=${since}`,
    label: "export",
  });
  importMoreJob = LogDialog.create({
    dialogId: "importMoreDialog",
    outputId: "importMoreOutput",
    statusUrl: (since) => `/api/processing-output?since=${since}`,
    label: "import",
  });
  faceTaggingCsvJob = LogDialog.create({
    dialogId: "faceTaggingCsvDialog",
    outputId: "faceTaggingCsvOutput",
    statusUrl: (since) => `/api/album-tools/face-tagging-csv/status?since=${since}`,
    label: "face-tagging CSV export",
  });

  $("#minStarsInput").on("change", updateExportCount);

  $("#importMoreBtn").on("click", async () => {
    setImportMoreUIEnabled(false);
    $("#importMoreStatus").text("Choose a folder…");
    let path;
    try {
      const res = await apiPost("/api/browse-folder", { title: "Select folder to import more images from", context: "import" });
      path = res.path;
    } catch (err) {
      $("#importMoreStatus").text("Folder picker failed: " + err.message);
      setImportMoreUIEnabled(true);
      return;
    }
    if (!path) {
      // Cancelled — quietly return to the idle state.
      $("#importMoreStatus").text("");
      setImportMoreUIEnabled(true);
      return;
    }

    importMoreJob.reset();
    importMoreJob.open("Importing…");
    $("#importMoreStatus").text("Importing…");
    try {
      await apiPost("/api/import-more", { path });
      importMoreJob.startPolling((data) => {
        if (data.running) return;
        setImportMoreUIEnabled(true);
        const success = data.returnCode === 0;
        importMoreJob.setFinished(success ? "Import complete" : `Import failed (exit code ${data.returnCode})`);
        if (success) {
          $("#importMoreStatus").text("Import complete.");
          loadSummary();
        } else {
          $("#importMoreStatus").text(`Import exited with code ${data.returnCode}.`);
        }
      });
    } catch (err) {
      $("#importMoreStatus").text("Failed: " + err.message);
      setImportMoreUIEnabled(true);
      importMoreJob.close();
    }
  });

  $("#openDestBtn").on("click", openExportDestination);

  $("#exportBtn").on("click", async () => {
    setExportUIEnabled(false);
    $("#exportStatus").text("Choose a destination folder…");
    $("#openDestBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select export destination folder", context: "export" });
    } catch (err) {
      $("#exportStatus").text("Browse failed: " + err.message);
      setExportUIEnabled(true);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#exportStatus").text("");
      setExportUIEnabled(true);
      return;
    }

    const exportFaceTagging = $("#exportFaceTaggingInput").is(":checked");
    const minStars = parseInt($("#minStarsInput").val(), 10) || 3;
    $("#exportStatus").text("");
    $("#exportDialogStatus").text("Starting export…");
    exportJob.reset();
    exportJob.open("Exporting…");
    try {
      await apiPost("/api/album-tools/export", { destination: res.path, exportFaceTagging, minStars });
      exportJob.startPolling((data) => {
        if (data.totalImages) {
          $("#exportDialogStatus").text(
            `Copying photos: ${data.copiedImages}/${data.totalImages} · ` +
            `Faces: ${data.processedPlayers}/${data.totalPlayers} player(s)`
          );
        }
        if (data.running) return;
        setExportUIEnabled(true);
        if (data.error) {
          $("#exportDialogStatus").text("Export failed: " + data.error);
          exportJob.setFinished("Export failed");
        } else {
          $("#exportDialogStatus").text(`Done — ${data.copiedImages} photo(s) exported.`);
          exportJob.setFinished("Export complete");
          if (data.destDir) {
            $("#openDestBtn").show();
            openExportDestination();
          }
        }
      });
    } catch (err) {
      $("#exportDialogStatus").text("Failed: " + err.message);
      exportJob.setFinished("Export failed");
      setExportUIEnabled(true);
    }
  });

  $("#faceTaggingCsvBtn").on("click", async () => {
    $("#faceTaggingCsvBtn").prop("disabled", true);
    $("#faceTaggingCsvStatus").text("Choose a destination folder…");
    $("#openFaceTaggingCsvFolderBtn").hide();
    let res;
    try {
      res = await apiPost("/api/browse-folder", { title: "Select folder for face-tagging CSV", context: "export" });
    } catch (err) {
      $("#faceTaggingCsvStatus").text("Browse failed: " + err.message);
      $("#faceTaggingCsvBtn").prop("disabled", false);
      return;
    }
    if (!res.path) {
      // User canceled the folder picker — quietly return to the idle state.
      $("#faceTaggingCsvStatus").text("");
      $("#faceTaggingCsvBtn").prop("disabled", false);
      return;
    }

    $("#faceTaggingCsvStatus").text("");
    $("#faceTaggingCsvDialogStatus").text("Starting export…");
    faceTaggingCsvJob.reset();
    faceTaggingCsvJob.open("Exporting…");
    try {
      await apiPost("/api/album-tools/face-tagging-csv/render", { destination: res.path });
      faceTaggingCsvJob.startPolling((data) => {
        if (data.running) return;
        $("#faceTaggingCsvBtn").prop("disabled", false);
        const success = !data.error;
        faceTaggingCsvJob.setFinished(success ? "Export complete" : "Export failed");
        if (success) {
          $("#faceTaggingCsvStatus").text(data.result ? `Wrote ${data.result.rows} row(s) to ${data.result.csvPath}` : "Done.");
          $("#openFaceTaggingCsvFolderBtn").show();
        } else {
          $("#faceTaggingCsvStatus").text("Failed: " + data.error);
        }
      });
    } catch (err) {
      $("#faceTaggingCsvDialogStatus").text("Failed: " + err.message);
      faceTaggingCsvJob.setFinished("Export failed");
      $("#faceTaggingCsvBtn").prop("disabled", false);
    }
  });

  $("#openFaceTaggingCsvFolderBtn").on("click", async () => {
    try {
      await apiPost("/api/album-tools/face-tagging-csv/open-folder", {});
    } catch (err) {
      $("#faceTaggingCsvStatus").text("Could not open folder: " + err.message);
    }
  });
});
