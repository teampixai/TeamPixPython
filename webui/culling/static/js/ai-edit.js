"use strict";

// Edit mode — triggered by the "AI edit" button on both the Culling page
// (next to the file-path box) and the Faces page (in the original-image
// preview window, see viewport.js). Three steps:
//
//   1. Editor: a before/after split view (Viewport.showCompareWindow) with a
//      control panel on the right — Brightness (Highlights / Midtones /
//      Shadows, -2..+2 EV) and Contrast sliders, each section with an
//      "Auto" checkbox that fills in the server's suggested values and locks
//      its sliders, plus "Sharpen players", which shows a box around every
//      detected player to pick who gets sharpened. The "Preview" side is
//      rendered by the server with autoedit.apply_tone() — the same code the
//      final render uses — so what you see is what Apply produces (minus
//      sharpening, which is too slow to preview).
//   2. Apply: autoedit.py runs at full resolution, streaming its output into
//      a non-closable jQuery UI modal via LogDialog (see common.js), reusing
//      the same processing_state/`/api/processing-output` polling plumbing as
//      import-more/rerun-facereco/deep-regrade (see album-tools.js) — only
//      one such background job can run at a time.
//   3. Review: the log gives way to the before/after compare window whose
//      Accept/Reject buttons decide whether the generated image is kept.
//
// The three checkboxes are remembered in a cookie (all on by default), but
// only updated when a result is accepted.
const AiEdit = (function () {
  const PREFS_COOKIE = "aiEditPrefs";
  // Wait this long after the last slider change (user drag or an Auto
  // checkbox toggle) before re-rendering the preview.
  const PREVIEW_DEBOUNCE_MS = 500;
  const ZONES = [
    { key: "highlights", label: "Highlights" },
    { key: "midtones", label: "Midtones" },
    { key: "shadows", label: "Shadows" },
  ];

  let job = null;

  function dialog() {
    if (!job) {
      job = LogDialog.create({
        dialogId: "aiEditDialog",
        outputId: "aiEditOutput",
        statusUrl: (since) => `/api/processing-output?since=${since}`,
        pollIntervalMs: 1000,
        label: "edit",
      });
    }
    return job;
  }

  // ---------------------------------------------------------------- //
  // Checkbox memory
  // ---------------------------------------------------------------- //
  function readPrefs() {
    const prefs = { brightnessAuto: true, contrastAuto: true, sharpen: true };
    try {
      const saved = JSON.parse(getCookie(PREFS_COOKIE) || "{}");
      Object.keys(prefs).forEach((k) => {
        if (typeof saved[k] === "boolean") prefs[k] = saved[k];
      });
    } catch (err) {
      // Unreadable cookie: fall back to the defaults.
    }
    return prefs;
  }

  function savePrefs(prefs) {
    setCookie(PREFS_COOKIE, JSON.stringify(prefs));
  }

  // ---------------------------------------------------------------- //
  // Step 1: editor
  // ---------------------------------------------------------------- //
  function formatEv(v) {
    return (v > 0 ? "+" : v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(1);
  }

  function formatContrast(v) {
    return (v > 0 ? "+" : v < 0 ? "\u2212" : "") + Math.abs(Math.round(v));
  }

  // One labelled range slider with a live value readout.
  function sliderRow(label, min, max, step, format) {
    const $input = $("<input>", { type: "range", min, max, step, value: 0 });
    const $value = $("<output>", { class: "edit-slider-value" });
    const $row = $("<label>", { class: "edit-slider" }).append(
      $("<span>", { class: "edit-slider-name" }).text(label), $input, $value
    );
    return {
      $row,
      $input,
      get value() { return parseFloat($input.val()); },
      set(v) { $input.val(v); $value.text(format(parseFloat($input.val()))); },
      refresh() { $value.text(format(parseFloat($input.val()))); },
      setEnabled(on) { $input.prop("disabled", !on); $row.toggleClass("disabled", !on); },
    };
  }

  function autoCheckbox() {
    const $input = $("<input>", { type: "checkbox" });
    const $label = $("<label>", { class: "edit-auto", title: "Use suggested values" }).append($input, " Auto");
    return { $label, $input };
  }

  function showLoading(text) {
    const $el = $("<div>", { class: "image-window-backdrop edit-loading" })
      .append($("<div>", { class: "edit-loading-text" }).text(text));
    $("body").append($el).addClass("image-window-open");
    return () => {
      $el.remove();
      if (!$(".image-window-backdrop").length) $("body").removeClass("image-window-open");
    };
  }

  function openEditor(key, session) {
    const prefs = readPrefs();
    const proposed = session.proposed;
    const evLimit = session.limits.ev;
    const contrastLimit = session.limits.contrast;
    const bodies = session.bodies || [];
    const selected = new Set(bodies.filter((b) => b.isTeammate).map((b) => b.id));
    let hoverId = null;
    let win = null;

    // --- Brightness ---
    const brightAuto = autoCheckbox();
    const zoneSliders = {};
    const $bright = $("<section>", { class: "edit-section" }).append(
      $("<div>", { class: "edit-section-head" }).append($("<h3>").text("Brightness"), brightAuto.$label)
    );
    ZONES.forEach((z) => {
      zoneSliders[z.key] = sliderRow(z.label, -evLimit, evLimit, 0.1, formatEv);
      $bright.append(zoneSliders[z.key].$row);
    });

    // --- Contrast ---
    const contrastAuto = autoCheckbox();
    const contrastSlider = sliderRow("Contrast", -contrastLimit, contrastLimit, 1, formatContrast);
    const $contrast = $("<section>", { class: "edit-section" }).append(
      $("<div>", { class: "edit-section-head" }).append($("<h3>").text("Contrast"), contrastAuto.$label),
      contrastSlider.$row
    );

    // --- Sharpen players ---
    const $sharpen = $("<input>", { type: "checkbox" });
    const $sharpenHint = $("<p>", { class: "edit-hint" });
    const $sharpenCount = $("<p>", { class: "edit-count" });
    const $sharpenSection = $("<section>", { class: "edit-section" }).append(
      $("<div>", { class: "edit-section-head" }).append(
        $("<label>", { class: "edit-check" }).append($sharpen, $("<h3>").text("Sharpen players"))
      ),
      $sharpenHint,
      $sharpenCount
    );

    // --- Actions ---
    const $cancel = $("<button>", { type: "button", class: "btn" }).text("Cancel");
    const $apply = $("<button>", { type: "button", class: "btn btn-primary" }).text("Apply");
    const $previewStatus = $("<p>", { class: "edit-preview-status", role: "status" });
    const $panel = $("<div>", { class: "edit-panel" }).append(
      $bright, $contrast, $sharpenSection, $previewStatus,
      $("<div>", { class: "edit-panel-actions" }).append($cancel, $apply)
    );

    // --- Player boxes (laid over the photo, zoom with it) ---
    const $overlay = $("<div>");
    const boxEls = {};
    bodies.forEach((b) => {
      const [x1, y1, x2, y2] = b.box;
      boxEls[b.id] = $("<div>", { class: "edit-box" }).css({
        left: `${x1 * 100}%`, top: `${y1 * 100}%`,
        width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`,
      }).appendTo($overlay);
    });

    function renderBoxes() {
      const on = $sharpen.prop("checked");
      $overlay.toggle(on && bodies.length > 0);
      bodies.forEach((b) => {
        boxEls[b.id].toggleClass("selected", selected.has(b.id)).toggleClass("hover", hoverId === b.id);
      });
      if (!bodies.length) {
        $sharpenHint.text("No players found in this photo.");
        $sharpenCount.text("");
      } else {
        $sharpenHint.text(on ? "Click a box on the photo to include or exclude a player." : "");
        $sharpenCount.text(on ? `${selected.size} of ${bodies.length} player${bodies.length === 1 ? "" : "s"} selected` : "");
      }
    }

    // Smallest box containing the point, so a box nested inside (or mostly
    // overlapped by) a bigger one can always be picked.
    function boxAt(x, y) {
      let best = null;
      let bestArea = Infinity;
      bodies.forEach((b) => {
        const [x1, y1, x2, y2] = b.box;
        if (x < x1 || x > x2 || y < y1 || y > y2) return;
        const area = (x2 - x1) * (y2 - y1);
        if (area < bestArea) { best = b; bestArea = area; }
      });
      return best;
    }

    // --- Live preview ---
    function currentTone() {
      return {
        highlights: zoneSliders.highlights.value,
        midtones: zoneSliders.midtones.value,
        shadows: zoneSliders.shadows.value,
        contrast: contrastSlider.value,
      };
    }

    function previewUrl() {
      const q = new URLSearchParams(Object.entries(currentTone()).map(([k, v]) => [k, String(v)]));
      q.set("file", key);
      return `/api/edit/ai-edit/session/preview?${q.toString()}`;
    }

    let previewTimer = null;
    let previewSeq = 0;
    function schedulePreview() {
      clearTimeout(previewTimer);
      $previewStatus.removeClass("error").text("Preview will update shortly\u2026");
      previewTimer = setTimeout(() => {
        const seq = ++previewSeq;
        const url = previewUrl();
        $previewStatus.text("Updating preview\u2026");
        // Preload so the image only swaps once the new render is ready,
        // and drop responses that a newer request has overtaken.
        const img = new Image();
        img.onload = () => {
          if (seq !== previewSeq || !win) return;
          win.setAfterUrl(url);
          $previewStatus.text("");
        };
        img.onerror = () => {
          if (seq !== previewSeq) return;
          $previewStatus.addClass("error").text("Couldn't update the preview. Try moving a slider again.");
        };
        img.src = url;
      }, PREVIEW_DEBOUNCE_MS);
    }

    // --- Auto checkboxes ---
    // The user's own slider values, kept while Auto overrides the sliders
    // so unchecking Auto restores exactly what was there before.
    const manual = { highlights: 0, midtones: 0, shadows: 0, contrast: 0 };

    function applyBrightnessAuto() {
      const on = brightAuto.$input.prop("checked");
      ZONES.forEach((z) => {
        zoneSliders[z.key].set(on ? proposed[z.key] : manual[z.key]);
        zoneSliders[z.key].setEnabled(!on);
      });
    }

    function applyContrastAuto() {
      const on = contrastAuto.$input.prop("checked");
      contrastSlider.set(on ? proposed.contrast : manual.contrast);
      contrastSlider.setEnabled(!on);
    }

    brightAuto.$input.prop("checked", prefs.brightnessAuto).on("change", () => { applyBrightnessAuto(); schedulePreview(); });
    contrastAuto.$input.prop("checked", prefs.contrastAuto).on("change", () => { applyContrastAuto(); schedulePreview(); });
    $sharpen.prop("checked", prefs.sharpen).on("change", renderBoxes);
    ZONES.forEach((z) => zoneSliders[z.key].$input.on("input", () => {
      zoneSliders[z.key].refresh();
      manual[z.key] = zoneSliders[z.key].value;
      schedulePreview();
    }));
    contrastSlider.$input.on("input", () => {
      contrastSlider.refresh();
      manual.contrast = contrastSlider.value;
      schedulePreview();
    });

    applyBrightnessAuto();
    applyContrastAuto();
    renderBoxes();

    const stamp = Date.now();
    win = Viewport.showCompareWindow({
      title: `Edit photo \u2014 ${session.file}`,
      beforeUrl: `/api/edit/ai-edit/session/before?file=${encodeURIComponent(key)}&t=${stamp}`,
      afterUrl: previewUrl() + `&t=${stamp}`,
      beforeLabel: "Original",
      afterLabel: "Preview",
      sidePanel: $panel,
      overlay: $overlay,
      onImageClick(x, y) {
        if (!$sharpen.prop("checked")) return false;
        const b = boxAt(x, y);
        if (!b) return false;
        if (selected.has(b.id)) selected.delete(b.id);
        else selected.add(b.id);
        renderBoxes();
        return true;
      },
      onImageHover(x, y) {
        const b = x == null || !$sharpen.prop("checked") ? null : boxAt(x, y);
        const id = b ? b.id : null;
        if (id !== hoverId) { hoverId = id; renderBoxes(); }
        return !!b;
      },
    });

    function closeEditor() {
      clearTimeout(previewTimer);
      previewSeq++;
      win.close();
      win = null;
    }

    $cancel.on("click", () => {
      closeEditor();
      fetch("/api/edit/ai-edit/session", { method: "DELETE" }).catch(() => {});
    });

    $apply.on("click", () => {
      const sharpen = $sharpen.prop("checked");
      const request = Object.assign({ file: key, sharpen, bodyIds: sharpen ? [...selected] : [] }, currentTone());
      const chosenPrefs = {
        brightnessAuto: brightAuto.$input.prop("checked"),
        contrastAuto: contrastAuto.$input.prop("checked"),
        sharpen,
      };
      closeEditor();
      apply(request, chosenPrefs);
    });
  }

  // ---------------------------------------------------------------- //
  // Step 2: apply (log console)
  // ---------------------------------------------------------------- //
  async function apply(request, prefs) {
    const j = dialog();
    j.reset();
    j.open("Applying edits\u2026");
    try {
      await apiPost("/api/edit/ai-edit/apply", request);
    } catch (err) {
      j.appendLines([`Failed to start: ${err.message}`]);
      fetch("/api/edit/ai-edit/session", { method: "DELETE" }).catch(() => {});
      j.setFinished("Edit failed to start");
      return;
    }
    j.startPolling((data) => {
      if (data.running) return;
      if (data.returnCode === 0) {
        j.close();
        openCompare(prefs);
      } else {
        j.setFinished(`Edit failed (exit code ${data.returnCode})`);
      }
    });
  }

  // ---------------------------------------------------------------- //
  // Step 3: review — the edit is only recorded on the album entry once
  // accepted, and the generated file is deleted on reject (see
  // /api/edit/ai-edit/accept|reject in culling_app.py).
  // ---------------------------------------------------------------- //
  async function openCompare(prefs) {
    let state;
    try {
      state = await apiGet("/api/edit/ai-edit/pending");
    } catch (err) {
      state = null;
    }
    if (!state || !state.pending) {
      reopenLog("The edit finished, but no image was produced.", "Edit produced no image");
      return;
    }
    const stamp = Date.now();
    Viewport.showCompareWindow({
      title: `Review edit \u2014 ${state.file}`,
      beforeUrl: `/api/edit/ai-edit/image?side=before&t=${stamp}`,
      afterUrl: `/api/edit/ai-edit/image?side=after&t=${stamp}`,
      beforeLabel: "Before",
      afterLabel: "After",
      acceptLabel: "Accept",
      rejectLabel: "Reject",
      onAccept: () => apiPost("/api/edit/ai-edit/accept", {}).then(() => savePrefs(prefs)).catch((err) => {
        alert(`Failed to keep the edit: ${err.message}`);
      }),
      onReject: () => apiPost("/api/edit/ai-edit/reject", {}).catch((err) => {
        alert(`Failed to discard the edit: ${err.message}`);
      }),
    });
  }

  // Reopens the log modal in a terminal (closable) state with `message`
  // appended — used when the run ends without something to compare. Goes
  // straight to the "already finished" dialog chrome (skipping the usual
  // non-closable-while-running state LogDialog.open() sets up), so this
  // uses the raw jQuery UI dialog ($dialog, exposed by LogDialog) directly.
  function reopenLog(message, title) {
    const j = dialog();
    j.appendLines([message]);
    j.$dialog.dialog("option", "title", title);
    j.$dialog.dialog("option", "closeOnEscape", true);
    j.$dialog.dialog("open");
    j.$dialog.dialog("widget").find(".ui-dialog-titlebar-close").show();
  }

  // Opens edit mode for `key` (the Album's bookkeeping key / plain
  // filename — same identifier used by /api/original etc.).
  async function run(key) {
    const done = showLoading("Opening photo\u2026");
    let session;
    try {
      session = await apiPost("/api/edit/ai-edit/session", { file: key });
    } catch (err) {
      done();
      alert(`Could not open the photo for editing: ${err.message}`);
      return;
    }
    done();
    openEditor(key, session);
  }

  return { run };
})();
