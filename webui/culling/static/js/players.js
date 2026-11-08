"use strict";

// Players page: curate one team's face database. The left nav lists every
// player (profile picture + name). The main area shows the selected player's
// Positive crops (left) and Negative crops (right). Users drag crops between
// the two sets, delete them, or pick a profile picture. Edits are applied
// locally right away and queued as ops; the queue is committed when another
// player or team is selected, before a rebuild, and when the page is left.

const SETS = ["Face", "Negative"];
const SECRET_ICON = window.SECRET_PLAYER_ICON || "/static/img/secret-player.png";

let teamId = null;
let players = [];
let current = null;            // working copy: { name, number, inRoster, profile, Face:[], Negative:[] }
let pendingOps = [];
let selection = new Set();     // keys "set|folder|crop"
let anchor = null;             // { set, index } for shift-click ranges
let dragKeys = null;
let flushing = null;
let rebuildJob = null;

const keyOf = (set, img) => `${set}|${img.folder}|${img.crop}`;
const sameImg = (a, b) => !!a && !!b && a.folder === b.folder && a.crop === b.crop;

// img.disk is the set the file is actually in on disk; it differs from the
// pane it's shown in while a move is still queued (not yet committed).
function imgUrl(set, img) {
  const onDisk = img.disk || set;
  return `/api/players/img/${encodeURIComponent(teamId)}/${encodeURIComponent(img.folder)}/${onDisk}/${encodeURIComponent(img.crop)}`;
}

function errorText(err) {
  try { return JSON.parse(err.message).detail || err.message; } catch (_) { return err.message; }
}

function setStatus(text, isError) {
  $("#playersStatus").text(text || "").toggleClass("players-status-error", !!isError);
}

function toWorking(p) {
  const tag = (disk) => (img) => Object.assign({}, img, { disk });
  return { name: p.name, number: p.number, inRoster: p.inRoster,
           profile: p.profile ? Object.assign({}, p.profile, { disk: "Face" }) : null,
           Face: p.positives.map(tag("Face")), Negative: p.negatives.map(tag("Negative")) };
}

// Same rule as the server: the chosen crop while it's still positive, else
// the first positive, else none (secret-player icon).
function settleProfile(w) {
  if (!w.Face.some((img) => sameImg(img, w.profile))) w.profile = w.Face[0] || null;
}

function profileSrc(p) {
  return p.profile ? imgUrl("Face", p.profile) : SECRET_ICON;
}

// ------------------------------------------------------------------ //
// Rendering
// ------------------------------------------------------------------ //
function renderNav() {
  const $nav = $("#playersNav").empty();
  if (!players.length) {
    $nav.append($("<p class='muted'>").text("No players yet. Add them on the Team page."));
    return;
  }
  players.forEach((p) => {
    const shown = current && current.name === p.name ? current : p;
    const $row = $("<div>", { class: "players-nav-row" + (current && current.name === p.name ? " active" : "") });
    $row.append($("<img>", { class: "players-avatar", src: profileSrc(shown), alt: "" }));
    const $text = $("<div class='players-nav-text'>");
    $text.append($("<div class='players-nav-name'>").text(p.name));
    const meta = [];
    if (p.number) meta.push("#" + p.number);
    if (!p.inRoster) meta.push("not on roster");
    if (meta.length) $text.append($("<div class='players-nav-meta muted'>").text(meta.join(" · ")));
    $row.append($text);
    $row.on("click", () => selectPlayer(p.name));
    $nav.append($row);
  });
}

function renderGrid(set) {
  const $grid = $(set === "Face" ? "#playersPosGrid" : "#playersNegGrid").empty();
  $(set === "Face" ? "#playersPosCount" : "#playersNegCount").text(current ? current[set].length : 0);
  if (!current) return;
  if (!current[set].length) {
    $grid.append($("<p class='muted players-empty'>").text(
      set === "Face" ? "No positive photos yet — drag some here from Negative." : "No negative photos. Drag look-alikes here."));
    return;
  }
  current[set].forEach((img, index) => {
    const key = keyOf(set, img);
    const isProfile = set === "Face" && sameImg(img, current.profile);
    const $tile = $("<div>", {
      class: "players-tile" + (selection.has(key) ? " selected" : "") + (isProfile ? " profile" : ""),
      title: img.crop,
      "data-key": key,
    });
    // Not via the props object above: jQuery would call jQuery UI's
    // .draggable() plugin instead of setting the HTML5 attribute.
    $tile.attr("draggable", "true");
    $tile.append($("<img>", { src: imgUrl(set, img), loading: "lazy", alt: "" })
      .attr("draggable", "false")
      .on("load", function () { $tile.attr("title", `${img.crop} — ${this.naturalWidth}×${this.naturalHeight}px`); }));
    if (isProfile) $tile.append($("<span class='players-profile-badge' title='Profile picture'>★</span>"));
    $tile.on("click", (e) => { e.stopPropagation(); onTileClick(e, set, index); });
    $tile.on("dragstart", (e) => { hidePreview(); onDragStart(e, set, index); });
    $tile.on("dragend", onDragEnd);
    $tile.on("mouseenter", () => schedulePreview($tile));
    $tile.on("mouseleave mousedown", hidePreview);
    $grid.append($tile);
  });
}

function renderMain() {
  hidePreview();
  SETS.forEach(renderGrid);
  updateToolbar();
}

// ------------------------------------------------------------------ //
// Hover preview: after HOVER_PREVIEW_MS on a thumbnail, show the crop at
// full size, centered on the thumbnail's center. It is scaled down only if
// it can't fit in the window.
// ------------------------------------------------------------------ //
const HOVER_PREVIEW_MS = 750;
const PREVIEW_MARGIN = 8;
let previewTimer = null;

function schedulePreview($tile) {
  hidePreview();
  previewTimer = setTimeout(() => showPreview($tile), HOVER_PREVIEW_MS);
}

function hidePreview() {
  if (previewTimer) { clearTimeout(previewTimer); previewTimer = null; }
  $(".players-hover-preview").remove();
}

function showPreview($tile) {
  previewTimer = null;
  const thumb = $tile.find("img")[0];
  if (!thumb || !thumb.naturalWidth || !document.body.contains(thumb)) return;
  const pad = 10; // preview padding + border, both sides
  const maxW = window.innerWidth - 2 * PREVIEW_MARGIN - pad;
  const maxH = window.innerHeight - 2 * PREVIEW_MARGIN - pad;
  const scale = Math.min(1, maxW / thumb.naturalWidth, maxH / thumb.naturalHeight);
  const w = Math.round(thumb.naturalWidth * scale);
  const h = Math.round(thumb.naturalHeight * scale);
  const r = thumb.getBoundingClientRect();
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const left = clamp(r.left + r.width / 2 - (w + pad) / 2, PREVIEW_MARGIN, window.innerWidth - w - pad - PREVIEW_MARGIN);
  const top = clamp(r.top + r.height / 2 - (h + pad) / 2, PREVIEW_MARGIN, window.innerHeight - h - pad - PREVIEW_MARGIN);
  $("<div class='players-hover-preview'>")
    .css({ left, top })
    .append($("<img>", { src: thumb.currentSrc || thumb.src, alt: "" }).css({ width: w, height: h }))
    .appendTo("body");
}

// Selection-only change: toggle classes on the existing tiles. Never rebuild
// the grid here — rebuilding during dragstart removes the dragged element and
// the browser cancels the drag.
function refreshSelection() {
  $(".players-tile").each(function () {
    $(this).toggleClass("selected", selection.has(this.dataset.key));
  });
  updateToolbar();
}

function updateToolbar() {
  const items = selectedItems();
  const single = items.length === 1 ? items[0] : null;
  const canProfile = !!single && single.set === "Face";
  $("#playersProfileBtn").prop("disabled", !canProfile).attr("title",
    single && single.set !== "Face" ? "Only positive images can be the profile picture" : "Use the selected photo as this player's picture");
  $("#playersDeleteBtn").prop("disabled", items.length === 0);
}

// ------------------------------------------------------------------ //
// Selection
// ------------------------------------------------------------------ //
function selectedItems() {
  if (!current) return [];
  const out = [];
  SETS.forEach((set) => current[set].forEach((img) => {
    if (selection.has(keyOf(set, img))) out.push({ set, img });
  }));
  return out;
}

function onTileClick(e, set, index) {
  const key = keyOf(set, current[set][index]);
  if (e.shiftKey && anchor && anchor.set === set) {
    if (!(e.ctrlKey || e.metaKey)) selection.clear();
    const [a, b] = [Math.min(anchor.index, index), Math.max(anchor.index, index)];
    for (let i = a; i <= b; i++) selection.add(keyOf(set, current[set][i]));
  } else if (e.ctrlKey || e.metaKey) {
    if (selection.has(key)) selection.delete(key); else selection.add(key);
    anchor = { set, index };
  } else {
    selection = new Set([key]);
    anchor = { set, index };
  }
  refreshSelection();
}

function clearSelection() {
  if (!selection.size) return;
  selection.clear();
  anchor = null;
  refreshSelection();
}

// ------------------------------------------------------------------ //
// Edits (applied locally, queued for commit)
// ------------------------------------------------------------------ //
function moveItems(items, to) {
  const moving = items.filter((it) => it.set !== to);
  if (!moving.length) return;
  moving.forEach(({ set, img }) => {
    current[set] = current[set].filter((x) => !sameImg(x, img));
    current[to].push(img);
    pendingOps.push({ op: "move", folder: img.folder, crop: img.crop, to });
  });
  selection = new Set(moving.map(({ img }) => keyOf(to, img)));
  anchor = null;
  afterEdit();
}

function deleteSelected() {
  const items = selectedItems();
  if (!items.length) return;
  items.forEach(({ set, img }) => {
    current[set] = current[set].filter((x) => !sameImg(x, img));
    pendingOps.push({ op: "delete", folder: img.folder, crop: img.crop });
  });
  selection.clear();
  anchor = null;
  afterEdit();
}

function setProfileFromSelection() {
  const items = selectedItems();
  if (items.length !== 1 || items[0].set !== "Face") return;
  const img = items[0].img;
  current.profile = img;
  pendingOps.push({ op: "profile", folder: img.folder, crop: img.crop });
  afterEdit();
}

function afterEdit() {
  settleProfile(current);
  setStatus("Unsaved changes — saved when you switch player");
  renderMain();
  renderNav();
}

// Commit the queued ops for the current player. Resolves true on success.
// On failure the page reloads from disk, so it never shows edits that were
// not saved.
function flush() {
  if (flushing) return flushing.then(() => flush());
  if (!pendingOps.length || !current) return Promise.resolve(true);
  const ops = pendingOps;
  const name = current.name;
  pendingOps = [];
  setStatus("Saving…");
  flushing = apiPost("/api/players/commit", { teamId, name, ops })
    .then((res) => {
      const idx = players.findIndex((p) => p.name === name);
      if (idx >= 0) players[idx] = res.player;
      if (current && current.name === name && !pendingOps.length) {
        current = toWorking(res.player);
        selection.clear();
      }
      setStatus("Saved.");
      return true;
    })
    .catch(async (err) => {
      setStatus("Save failed: " + errorText(err) + " — reloaded from disk.", true);
      await loadTeam(teamId, name).catch(() => {});
      return false;
    })
    .finally(() => { flushing = null; renderMain(); renderNav(); });
  return flushing;
}

function flushOnUnload() {
  if (!pendingOps.length || !current) return;
  const body = JSON.stringify({ teamId, name: current.name, ops: pendingOps });
  pendingOps = [];
  fetch("/api/players/commit", {
    method: "POST", headers: { "Content-Type": "application/json" }, body, keepalive: true,
  }).catch(() => {});
}

// ------------------------------------------------------------------ //
// Drag & drop between the Positive / Negative panes
// ------------------------------------------------------------------ //
function onDragStart(e, set, index) {
  const key = keyOf(set, current[set][index]);
  if (!selection.has(key)) {
    selection = new Set([key]);
    anchor = { set, index };
    refreshSelection();
  }
  dragKeys = new Set(selection);
  const dt = e.originalEvent.dataTransfer;
  dt.effectAllowed = "move";
  dt.setData("text/plain", "players-faces");
}

function onDragEnd() {
  dragKeys = null;
  $(".players-pane").removeClass("drop-target");
}

function bindPane($pane) {
  const set = $pane.data("set");
  const accepts = () => !!dragKeys && [...dragKeys].some((k) => !k.startsWith(set + "|"));
  $pane.on("dragover", (e) => {
    if (!accepts()) return;
    e.preventDefault();
    e.originalEvent.dataTransfer.dropEffect = "move";
    $pane.addClass("drop-target");
  });
  $pane.on("dragleave", (e) => {
    if (!$pane[0].contains(e.originalEvent.relatedTarget)) $pane.removeClass("drop-target");
  });
  $pane.on("drop", (e) => {
    e.preventDefault();
    $pane.removeClass("drop-target");
    if (!accepts()) return;
    const items = selectedItems().filter(({ set: s, img }) => dragKeys.has(keyOf(s, img)));
    dragKeys = null;
    moveItems(items, set);
  });
  $pane.find(".players-grid").on("click", clearSelection);
}

// ------------------------------------------------------------------ //
// Loading / switching
// ------------------------------------------------------------------ //
async function loadTeam(requestedTeamId, keepName) {
  const data = await apiGet("/api/players/data" + (requestedTeamId ? "?teamId=" + encodeURIComponent(requestedTeamId) : ""));
  teamId = data.teamId;
  players = data.players || [];
  const $sel = $("#playersTeamSelect").empty();
  (data.teams || []).forEach((t) => $sel.append($("<option>", { value: t.id }).text(t.name || "Unnamed Team")));
  $sel.val(teamId);
  const keep = players.find((p) => p.name === keepName) || players[0] || null;
  current = keep ? toWorking(keep) : null;
  pendingOps = [];
  selection.clear();
  anchor = null;
  renderNav();
  renderMain();
}

async function selectPlayer(name) {
  if (current && current.name === name) return;
  if (!(await flush())) return;
  const p = players.find((x) => x.name === name);
  if (!p) return;
  current = toWorking(p);
  selection.clear();
  anchor = null;
  renderNav();
  renderMain();
}

async function changeTeam(newTeamId) {
  if (!(await flush())) { $("#playersTeamSelect").val(teamId); return; }
  setStatus("");
  await loadTeam(newTeamId).catch((err) => setStatus("Failed to load team: " + errorText(err), true));
}

async function rebuildFaceDb() {
  if (!(await flush())) return;
  const $btn = $("#playersRebuildBtn").prop("disabled", true);
  rebuildJob.reset();
  rebuildJob.open("Rebuilding face database…");
  try {
    await apiPost("/api/players/rebuild", { teamId });
  } catch (err) {
    rebuildJob.appendLines(["Failed to start: " + errorText(err)]);
    rebuildJob.setFinished("Rebuild failed");
    $btn.prop("disabled", false);
    return;
  }
  rebuildJob.startPolling(async (data) => {
    if (data.running) return;
    rebuildJob.setFinished(data.error ? "Rebuild failed" : "Rebuild complete");
    $btn.prop("disabled", false);
    setStatus(data.error ? "Rebuild failed: " + data.error : "Face database rebuilt.", !!data.error);
    await loadTeam(teamId, current && current.name).catch(() => {});
  });
}

$(function () {
  rebuildJob = LogDialog.create({
    dialogId: "playersRebuildDialog",
    outputId: "playersRebuildOutput",
    statusUrl: (since) => `/api/players/rebuild/status?since=${since}`,
    dialogOptions: { width: 760 },
    label: "face DB rebuild",
  });

  $(".players-pane").each((_, el) => bindPane($(el)));
  $(".players-grid").on("scroll", hidePreview);
  $(window).on("blur resize", hidePreview);
  $("#playersTeamSelect").on("change", function () { changeTeam($(this).val()); });
  $("#playersProfileBtn").on("click", setProfileFromSelection);
  $("#playersDeleteBtn").on("click", deleteSelected);
  $("#playersRebuildBtn").on("click", rebuildFaceDb);

  $(document).on("keydown", (e) => {
    if (e.key !== "Delete") return;
    if ($(e.target).is("input, textarea, select, [contenteditable]")) return;
    if ($(".ui-dialog:visible").length) return;
    if (!selection.size) return;
    e.preventDefault();
    deleteSelected();
  });

  window.addEventListener("pagehide", flushOnUnload);

  loadTeam(null).catch((err) => setStatus("Failed to load players: " + errorText(err), true));
});
