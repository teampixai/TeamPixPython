"use strict";

// Edit mode — triggered by the "AI edit" button on both the Culling page
// (next to the file-path box) and the Faces page (in the original-image
// preview window, see viewport.js). Three steps:
//
//   1. Editor: the photo (Viewport.showCompareWindow) with a tabbed property
//      page on the right; Cancel/Apply and the preview status sit in a footer
//      shared by every tab:
//        - AI Edit: Brightness (Highlights / Midtones / Shadows, -2..+2 EV)
//          and Contrast sliders, each section with an "Auto" checkbox that
//          fills in the server's suggested values and locks its sliders,
//          plus "Sharpen players", which shows a box around every detected
//          player to pick who gets sharpened. Shown as a before/after split;
//          the "Preview" side is rendered by the server with
//          autoedit.apply_tone() — the same code the final render uses — so
//          what you see is what Apply produces (minus sharpening, which is
//          too slow to preview).
//        - Crop: ratio presets (all H x W) and Custom, plus "Auto crop" (a
//          checkbox: an OpenAI model suggests what to crop out; while checked
//          the crop controls are locked, and unchecking restores the previous
//          crop). The Level (turn the photo -180..180 degrees, positive
//          counter-clockwise; Auto crop then also applies the model's level)
//          is built but hidden while LEVEL_ENABLED is false. The preview
//          becomes a single image with a crop box that can be moved and
//          resized from its edges and corners (presets keep their ratio,
//          Custom is free). The crop is in pixels of the turned photo's
//          canvas; changing the level re-fits the box inside the photo, and
//          the window keeps the unturned photo's shape so the layout stays put.
//        - Other: Reload the photo from disk, Revert it to the original
//          (deleting the edit), or open the edited copy in another app.
//      The crop always applies: whatever it cuts off is darkened on every
//      tab. It starts as Custom covering the whole photo (= no crop).
//   2. Apply: autoedit.py runs at full resolution (tone, sharpening, then
//      the crop), streaming its output into a non-closable jQuery UI modal
//      via LogDialog (see common.js), reusing the same
//      processing_state/`/api/processing-output` polling plumbing as
//      import-more/rerun-facereco/deep-regrade (see album-tools.js) — only
//      one such background job can run at a time.
//   3. Review: the log gives way to the before/after compare window whose
//      Abort / Save Copy / Save buttons decide what happens to the result:
//      discard it, add it to the album as a new photo, or keep it as this
//      photo's edit.
//
// The three checkboxes are remembered in a cookie (all on by default), but
// only updated when a result is accepted.
//
// AiEdit.run(key, { onResult }) reports how the run ended — see run() below
// for the result object — so the page that launched it can refresh itself.
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

  // Crop presets, all H x W (h/w are only used as a ratio). The iPhone Fold
  // entries are its screens' pixel sizes (outer display: portrait, shown
  // width x height; inner display: landscape, shown H x W like the rest). Each
  // group is shown widest first, except one marked `ordered`, which keeps the
  // order written here.
  const CROP_GROUPS = [
    { title: "Instagram", presets: [
      { id: "ig-5x4", label: "5x4", h: 5, w: 4 },
      { id: "ig-1x1", label: "1x1", h: 1, w: 1 },
    ] },
    { title: "Phone", ordered: true, presets: [
      { id: "phone", label: "iPhone / Android", detail: "19.5x9", h: 19.5, w: 9 },
      { id: "fold-outer", label: "iPhone Fold outer", detail: "1398x2034", detailUnit: "W x H px", h: 2034, w: 1398 },
      { id: "fold-inner", label: "iPhone Fold inner", detail: "1878x2670", h: 1878, w: 2670, unfolded: true },
    ] },
    { title: "Print", presets: ["16x9", "3x2", "4x3", "5x4", "1x1", "4x5", "3x4", "2x3", "9x16"].map((label) => {
      const [h, w] = label.split("x").map(Number);
      return { id: `print-${label}`, label, h, w };
    }) },
  ];
  CROP_GROUPS.forEach((g) => {
    if (!g.ordered) g.presets.sort((a, b) => b.w / b.h - a.w / a.h);
  });

  // Grab distance (screen px) around the crop box's edges and corners.
  const CROP_GRAB_PX = 10;
  // Smallest crop side, in source pixels.
  const CROP_MIN_PX = 32;
  const CROP_CURSORS = {
    n: "ns-resize", s: "ns-resize", e: "ew-resize", w: "ew-resize",
    nw: "nwse-resize", se: "nwse-resize", ne: "nesw-resize", sw: "nesw-resize", move: "move",
  };

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

  let autoCropJob = null;

  // The log dialog Auto crop shows its progress in, above the editor window.
  function autoCropDialog() {
    if (!autoCropJob) {
      autoCropJob = LogDialog.create({
        dialogId: "aiAutoCropDialog",
        outputId: "aiAutoCropOutput",
        statusUrl: (since) => `/api/edit/ai-edit/auto-crop/status?since=${since}`,
        pollIntervalMs: 500,
        dialogOptions: { width: 520, dialogClass: "edit-over" },
        label: "auto crop",
      });
      // The editor window sits above jQuery UI's usual layer; see style.css.
      autoCropJob.$dialog.on("dialogopen", () => $("body").addClass("edit-dialog-over"));
      autoCropJob.$dialog.on("dialogclose", () => $("body").removeClass("edit-dialog-over"));
    }
    return autoCropJob;
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  // The Level: degrees to turn the photo, positive counter-clockwise. Two
  // decimals is all the server keeps (_clean_level in culling_app.py).
  const LEVEL_LIMIT = 180;
  // Temporarily off: the Level widgets are not shown and Auto crop ignores
  // the level the model suggests. Everything underneath still works.
  const LEVEL_ENABLED = false;

  function cleanLevel(v) {
    const n = parseFloat(v);
    if (!Number.isFinite(n)) return 0;
    return Math.round(clamp(n, -LEVEL_LIMIT, LEVEL_LIMIT) * 100) / 100 + 0;
  }

  // Size of the canvas that holds a w x h photo turned by `deg` degrees —
  // must agree with autoedit.rotated_size() (the crop is in its pixels).
  function rotatedSize(w, h, deg) {
    const rad = (deg * Math.PI) / 180;
    const c = Math.abs(Math.cos(rad));
    const s = Math.abs(Math.sin(rad));
    return { w: Math.max(1, Math.round(w * c + h * s)), h: Math.max(1, Math.round(w * s + h * c)) };
  }

  // The largest w/h = `ratio` rectangle centred on the turned canvas that
  // lies wholly inside the photo (no empty corners).
  function inscribedCrop(srcW, srcH, deg, ratio) {
    const rad = (deg * Math.PI) / 180;
    const c = Math.abs(Math.cos(rad));
    const s = Math.abs(Math.sin(rad));
    const h = Math.min(srcW / (ratio * c + s), srcH / (ratio * s + c));
    const w = h * ratio;
    const canvas = rotatedSize(srcW, srcH, deg);
    return { x: (canvas.w - w) / 2, y: (canvas.h - h) / 2, w, h };
  }

  // Whether the photo turned by `deg` degrees covers the point (x, y) of its
  // canvas (`cw` x `ch`), within half a pixel.
  function coversPoint(srcW, srcH, cw, ch, deg, x, y) {
    const rad = (deg * Math.PI) / 180;
    const c = Math.cos(rad);
    const s = Math.sin(rad);
    const rx = x - cw / 2;
    const ry = y - ch / 2;
    return Math.abs(rx * c - ry * s) <= srcW / 2 + 0.5 && Math.abs(rx * s + ry * c) <= srcH / 2 + 0.5;
  }

  function formatLevel(v) {
    return (v > 0 ? "+" : v < 0 ? "\u2212" : "") + Math.abs(v).toFixed(1) + "\u00b0";
  }

  // FastAPI errors arrive as '{"detail": "..."}' — show just the detail.
  function errorText(err) {
    try {
      const parsed = JSON.parse(err.message);
      if (parsed && parsed.detail) return String(parsed.detail);
    } catch (e) {
      // Not JSON: use the message as is.
    }
    return err.message;
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

  // A small rectangle in the preset's shape (the long side 22px). `unfolded`
  // adds the faint crease of an opened foldable.
  function ratioIcon(h, w, unfolded) {
    const long = 22;
    const ratio = w / h;
    return $("<span>", { class: "crop-icon", "aria-hidden": "true" }).append(
      $("<span>", { class: "crop-icon-shape" + (unfolded ? " crop-icon-unfolded" : "") }).css({
        width: `${Math.round(long * Math.min(1, ratio))}px`,
        height: `${Math.round(long * Math.min(1, 1 / ratio))}px`,
      })
    );
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

  // A modal Yes/No question over the editor. Resolves true for Yes; No is
  // focused (and Esc answers it) so nothing destructive happens by accident.
  function confirmDialog(title, message) {
    return new Promise((resolve) => {
      const $no = $("<button>", { type: "button", class: "btn" }).text("No");
      const $yes = $("<button>", { type: "button", class: "btn btn-danger" }).text("Yes");
      const $box = $("<div>", { class: "edit-confirm", role: "alertdialog", "aria-modal": "true" }).append(
        $("<h3>").text(title),
        $("<p>").text(message),
        $("<div>", { class: "edit-confirm-actions" }).append($no, $yes)
      );
      const $backdrop = $("<div>", { class: "edit-confirm-backdrop" }).append($box).appendTo("body");
      // Registered on window in the capture phase so it runs before the
      // compare window's own document-level key swallowing.
      function onKey(e) {
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); done(false); }
      }
      function done(answer) {
        window.removeEventListener("keydown", onKey, true);
        $backdrop.remove();
        resolve(answer);
      }
      window.addEventListener("keydown", onKey, true);
      $no.on("click", () => done(false));
      $yes.on("click", () => done(true));
      $no.trigger("focus");
    });
  }

  function openEditor(key, initialSession, flow) {
    const prefs = readPrefs();
    let session = initialSession;
    let proposed = session.proposed;
    const evLimit = session.limits.ev;
    const contrastLimit = session.limits.contrast;
    let bodies = session.bodies || [];
    let selected = new Set(bodies.filter((b) => b.isTeammate).map((b) => b.id));
    let hoverId = null;
    let win = null;
    let activeTab = "tone";
    let busy = false;

    // The photo's own size (session.width x session.height) and the Level
    // it is turned by. The crop rectangle is in full-resolution pixels of
    // the canvas that holds the turned photo (imgW x imgH — the photo's
    // own size while the level is 0). Starts as the whole photo.
    let srcW = session.width;
    let srcH = session.height;
    let level = 0;
    let imgW = srcW;
    let imgH = srcH;
    let beforeLevel = 0;  // the level the "Original" image on screen was made for
    let bodiesUnavailable = !!session.bodiesUnavailable;
    let crop = { x: 0, y: 0, w: imgW, h: imgH };
    let cropPreset = null;  // null = Custom
    let cropDrag = null;

    // ============================ AI Edit tab ============================ //
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

    // ============================== Crop tab ============================= //
    // --- Level ---
    const levelSlider = sliderRow("Angle", -LEVEL_LIMIT, LEVEL_LIMIT, 0.1, formatLevel);
    levelSlider.$row.addClass("edit-slider-level");
    const $levelNumber = $("<input>", {
      type: "number", class: "edit-level-input", min: -LEVEL_LIMIT, max: LEVEL_LIMIT, step: 0.1, value: 0,
      "aria-label": "Level in degrees",
    });
    const $levelReset = $("<button>", { type: "button", class: "btn", title: "Back to 0\u00b0" }).text("Reset");
    const $level = $("<section>", { class: "edit-section" }).append(
      $("<div>", { class: "edit-section-head" }).append(
        $("<h3>").text("Level"),
        $("<span>", { class: "edit-level-entry" }).append($levelNumber, " \u00b0", $levelReset)
      ),
      levelSlider.$row,
      $("<p>", { class: "edit-hint" }).text(
        "Positive turns the photo counter-clockwise, negative clockwise. Changing the level re-fits the crop box " +
        "inside the turned photo."
      )
    );

    // --- Auto crop ---
    const $autoCrop = $("<input>", { type: "checkbox" });
    const $autoCropNote = $("<p>", { class: "edit-hint" });
    const $autoCropSection = $("<section>", { class: "edit-section" }).append(
      $("<div>", { class: "edit-section-head" }).append(
        $("<label>", { class: "edit-check" }).append($autoCrop, $("<h3>").text("Auto crop"))
      ),
      $("<p>", { class: "edit-hint" }).text(
        "Checking it asks an AI model (OpenAI) to crop out what doesn't belong, using a small copy of the photo. " +
        "While it is checked the crop controls are locked. " +
        "Unchecking it brings back the crop you had before; checking it again re-applies the suggestion."
      ),
      $autoCropNote
    );

    const $cropX = $("<span>", { class: "crop-readout-value" });
    const $cropSize = $("<span>", { class: "crop-readout-value" });
    const $customTile = $("<button>", { type: "button", class: "crop-tile crop-tile-custom", "aria-pressed": "true" }).append(
      $("<span>", { class: "crop-tile-head" }).append(
        $("<span>", { class: "crop-icon crop-icon-free", "aria-hidden": "true" }).append($("<span>", { class: "crop-icon-shape" })),
        $("<span>", { class: "crop-tile-label" }).text("Custom size")
      ),
      $("<span>", { class: "crop-readout" }).append(
        $("<span>", { class: "crop-readout-name" }).text("Top left"), $cropX,
        $("<span>", { class: "crop-readout-name" }).text("H \u00d7 W"), $cropSize
      )
    );
    const presetTiles = [];
    const $crop = $("<div>", { class: "crop-groups" }).append(...(LEVEL_ENABLED ? [$level] : []), $autoCropSection);
    CROP_GROUPS.forEach((group) => {
      const $grid = $("<div>", { class: "crop-grid" });
      group.presets.forEach((preset) => {
        const $tile = $("<button>", {
          type: "button", class: "crop-tile", "aria-pressed": "false",
          title: preset.detail
            ? `${preset.label} (${preset.detail}, ${preset.detailUnit || "H x W"})`
            : `${preset.label} (H x W)`,
        }).append(
          ratioIcon(preset.h, preset.w, preset.unfolded),
          $("<span>", { class: "crop-tile-label" }).text(preset.label),
          preset.detail ? $("<span>", { class: "crop-tile-detail" }).text(preset.detail) : null
        );
        $tile.on("click", () => selectPreset(preset));
        presetTiles.push({ preset, $tile });
        $grid.append($tile);
      });
      $crop.append($("<section>", { class: "edit-section" }).append($("<h3>").text(group.title), $grid));
    });
    $crop.append(
      $("<section>", { class: "edit-section" }).append($("<h3>").text("Custom"), $customTile),
      $("<p>", { class: "edit-hint" }).text(
        "Drag the box or its edges and corners on the photo. Scroll to zoom; drag outside the box to pan."
      )
    );
    $customTile.on("click", () => selectPreset(null));

    // ============================= Other tab ============================= //
    function actionRow($button, hint) {
      return $("<section>", { class: "edit-section edit-action" }).append(
        $button, $("<p>", { class: "edit-hint" }).text(hint)
      );
    }
    const $reload = $("<button>", { type: "button", class: "btn" }).text("Reload");
    const $revert = $("<button>", { type: "button", class: "btn btn-danger" }).text("Revert\u2026");
    const $external = $("<button>", {
      type: "button", class: "btn",
      title: "Settings not yet applied in this window are not included.",
    }).text("Edit with custom app\u2026");
    const $revertHint = $("<p>", { class: "edit-hint" });
    const $other = $("<div>", { class: "edit-actions" }).append(
      actionRow($reload, "Read the photo from disk again, e.g. after changing it in another app. Your settings here are kept."),
      $("<section>", { class: "edit-section edit-action" }).append($revert, $revertHint),
      actionRow($external,
        "Open the edited copy (made from the original if there isn't one yet) in an app you choose. " +
        "Settings not yet applied here are not included \u2014 click Reload after saving in the other app.")
    );

    // ============================ Panel shell ============================ //
    const TABS = [
      { id: "tone", label: "AI Edit", $pane: $("<div>").append($bright, $contrast, $sharpenSection) },
      { id: "crop", label: "Crop", $pane: $crop },
      { id: "other", label: "Other", $pane: $other },
    ];
    const $tabs = $("<div>", { class: "edit-tabs", role: "tablist" });
    const $panes = $("<div>", { class: "edit-tab-panes" });
    TABS.forEach((t) => {
      t.$tab = $("<button>", { type: "button", class: "edit-tab", role: "tab" }).text(t.label)
        .on("click", () => setTab(t.id));
      t.$pane.addClass("edit-tab-pane").attr("role", "tabpanel");
      $tabs.append(t.$tab);
      $panes.append(t.$pane);
    });

    const $cancel = $("<button>", { type: "button", class: "btn" }).text("Cancel");
    const $apply = $("<button>", { type: "button", class: "btn btn-primary" }).text("Apply");
    const $previewStatus = $("<p>", { class: "edit-preview-status", role: "status" });
    const $panel = $("<div>", { class: "edit-panel edit-panel-tabbed" }).append(
      $tabs,
      $panes,
      $("<div>", { class: "edit-panel-footer" }).append(
        $previewStatus,
        $("<div>", { class: "edit-panel-actions" }).append($cancel, $apply)
      )
    );

    // ============================== Overlay ============================== //
    // Player boxes (AI Edit tab) under the crop shading, so players the crop
    // cuts off are darkened too. Everything is %-positioned and zooms with
    // the photo.
    const $playerLayer = $("<div>", { class: "edit-player-layer" });
    const boxEls = {};
    function buildPlayerBoxes() {
      $playerLayer.empty();
      Object.keys(boxEls).forEach((id) => delete boxEls[id]);
      bodies.forEach((b) => {
        const [x1, y1, x2, y2] = b.box;
        boxEls[b.id] = $("<div>", { class: "edit-box" }).css({
          left: `${x1 * 100}%`, top: `${y1 * 100}%`,
          width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%`,
        }).appendTo($playerLayer);
      });
    }

    const shades = ["top", "bottom", "left", "right"].map(() => $("<div>", { class: "crop-shade" }));
    const $cropBox = $("<div>", { class: "crop-box" }).append(
      $("<div>", { class: "crop-thirds" }),
      ["nw", "n", "ne", "e", "se", "s", "sw", "w"].map((h) => $("<div>", { class: `crop-handle crop-handle-${h}` }))
    );
    const $cropLayer = $("<div>", { class: "crop-layer" }).append(shades, $cropBox);
    const $overlay = $("<div>").append($playerLayer, $cropLayer);

    // ---------------------------- Crop logic ---------------------------- //
    // The crop as whole source pixels, clamped to the photo — exactly what
    // Apply sends and the Custom tile shows.
    function cropPixels() {
      const x = clamp(Math.round(crop.x), 0, imgW - 1);
      const y = clamp(Math.round(crop.y), 0, imgH - 1);
      return {
        x, y,
        w: clamp(Math.round(crop.w), 1, imgW - x),
        h: clamp(Math.round(crop.h), 1, imgH - y),
      };
    }

    function isCropped() {
      const c = cropPixels();
      return c.x !== 0 || c.y !== 0 || c.w !== imgW || c.h !== imgH;
    }

    function resetCrop() {
      crop = { x: 0, y: 0, w: imgW, h: imgH };
      cropPreset = null;
    }

    // The canvas that holds the photo turned by the current Level.
    function refreshCanvas() {
      const canvas = rotatedSize(srcW, srcH, level);
      imgW = canvas.w;
      imgH = canvas.h;
    }

    // The w/h the crop box takes when it is re-fitted after a level change:
    // the preset's, else the photo's own (swapped once it is turned past 45
    // degrees, when its long side lies the other way).
    function refitRatio() {
      if (cropPreset) return cropPreset.w / cropPreset.h;
      const folded = ((level % 180) + 180) % 180;
      const turned = Math.min(folded, 180 - folded) > 45;
      return turned ? srcH / srcW : srcW / srcH;
    }

    // Level changed: the crop box becomes the largest one of its shape that
    // fits inside the turned photo.
    function refitCrop() {
      crop = inscribedCrop(srcW, srcH, level, refitRatio());
    }

    function syncLevelControls() {
      levelSlider.set(level);
      $levelNumber.val(level);
    }

    function renderCrop() {
      const pct = (v) => `${v * 100}%`;
      const l = crop.x / imgW;
      const t = crop.y / imgH;
      const r = (crop.x + crop.w) / imgW;
      const b = (crop.y + crop.h) / imgH;
      const [top, bottom, left, right] = shades;
      top.css({ left: 0, top: 0, width: "100%", height: pct(t) });
      bottom.css({ left: 0, top: pct(b), width: "100%", height: pct(1 - b) });
      left.css({ left: 0, top: pct(t), width: pct(l), height: pct(b - t) });
      right.css({ left: pct(r), top: pct(t), width: pct(1 - r), height: pct(b - t) });
      $cropBox.css({ left: pct(l), top: pct(t), width: pct(r - l), height: pct(b - t) });
      // Outside the Crop tab the box is only drawn when it actually cuts
      // something off; handles only ever show on the Crop tab.
      const editing = activeTab === "crop";
      $cropLayer.toggleClass("editing", editing);
      $cropBox.toggle(editing || isCropped());

      const c = cropPixels();
      $cropX.text(`${c.x}, ${c.y} px`);
      $cropSize.text(`${c.h} \u00d7 ${c.w} px`);
      $customTile.attr("aria-pressed", String(cropPreset === null));
      presetTiles.forEach(({ preset, $tile }) => $tile.attr("aria-pressed", String(cropPreset === preset)));
    }

    // `c` reshaped to the width/height `ratio`, keeping its centre and area,
    // then shrunk/shifted as needed to stay on the canvas.
    function reshapeCrop(c, ratio) {
      const cx = c.x + c.w / 2;
      const cy = c.y + c.h / 2;
      let h = Math.sqrt((c.w * c.h) / ratio);
      let w = h * ratio;
      if (w > imgW) { w = imgW; h = w / ratio; }
      if (h > imgH) { h = imgH; w = h * ratio; }
      return { x: clamp(cx - w / 2, 0, imgW - w), y: clamp(cy - h / 2, 0, imgH - h), w, h };
    }

    // `c` shrunk about its centre until it lies wholly inside the turned
    // photo (no empty corners), keeping its shape.
    function shrinkIntoPhoto(c) {
      const inside = (k) => {
        const w = c.w * k;
        const h = c.h * k;
        const x = c.x + (c.w - w) / 2;
        const y = c.y + (c.h - h) / 2;
        return [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
          .every(([px, py]) => coversPoint(srcW, srcH, imgW, imgH, level, px, py));
      };
      if (inside(1)) return c;
      let lo = 0;
      let hi = 1;
      for (let i = 0; i < 30; i++) {
        const mid = (lo + hi) / 2;
        if (inside(mid)) lo = mid; else hi = mid;
      }
      const w = c.w * lo;
      const h = c.h * lo;
      return { x: c.x + (c.w - w) / 2, y: c.y + (c.h - h) / 2, w, h };
    }

    // Picking a preset keeps the box's center and area, reshaped to the
    // preset's ratio, then shrunk/shifted as needed to stay on the photo.
    // Custom keeps the box as it is and frees the ratio.
    function selectPreset(preset) {
      cropPreset = preset;
      if (preset) crop = shrinkIntoPhoto(reshapeCrop(crop, preset.w / preset.h));
      renderCrop();
    }

    // Which part of the crop box is at normalised (nx, ny): an edge/corner
    // ("n", "se", ...), "move" inside it, or null outside it.
    function cropHitTest(nx, ny) {
      const tol = win.pxToImage(CROP_GRAB_PX);
      const tx = tol.x * imgW;
      const ty = tol.y * imgH;
      const x = nx * imgW;
      const y = ny * imgH;
      const L = crop.x, R = crop.x + crop.w, T = crop.y, B = crop.y + crop.h;
      const withinX = x >= L - tx && x <= R + tx;
      const withinY = y >= T - ty && y <= B + ty;
      let v = "";
      let h = "";
      if (withinX && Math.abs(y - T) <= ty) v = "n";
      else if (withinX && Math.abs(y - B) <= ty) v = "s";
      if (withinY && Math.abs(x - L) <= tx) h = "w";
      else if (withinY && Math.abs(x - R) <= tx) h = "e";
      if (v || h) return v + h;
      if (x > L && x < R && y > T && y < B) return "move";
      return null;
    }

    // Resizing with a fixed width/height `ratio`: corners keep the opposite
    // corner anchored, edges keep the opposite edge anchored and the box
    // centered along it.
    function ratioResize(s, mode, dx, dy, ratio, minSide) {
      const minW = minSide * Math.max(1, ratio);
      if (mode.length === 2) {
        const right = mode[1] === "e";
        const down = mode[0] === "s";
        const ax = right ? s.x : s.x + s.w;
        const ay = down ? s.y : s.y + s.h;
        const wantW = right ? s.x + s.w + dx - ax : ax - (s.x + dx);
        const wantH = down ? s.y + s.h + dy - ay : ay - (s.y + dy);
        const maxW = Math.min(right ? imgW - ax : ax, (down ? imgH - ay : ay) * ratio);
        const w = clamp(Math.max(wantW, wantH * ratio), Math.min(minW, maxW), maxW);
        const h = w / ratio;
        return { x: right ? ax : ax - w, y: down ? ay : ay - h, w, h };
      }
      if (mode === "e" || mode === "w") {
        const right = mode === "e";
        const ax = right ? s.x : s.x + s.w;
        const cy = s.y + s.h / 2;
        const maxW = Math.min(right ? imgW - ax : ax, 2 * Math.min(cy, imgH - cy) * ratio);
        const w = clamp(right ? s.w + dx : s.w - dx, Math.min(minW, maxW), maxW);
        const h = w / ratio;
        return { x: right ? ax : ax - w, y: cy - h / 2, w, h };
      }
      const down = mode === "s";
      const ay = down ? s.y : s.y + s.h;
      const cx = s.x + s.w / 2;
      const maxH = Math.min(down ? imgH - ay : ay, (2 * Math.min(cx, imgW - cx)) / ratio);
      const h = clamp(down ? s.h + dy : s.h - dy, Math.min(minW / ratio, maxH), maxH);
      const w = h * ratio;
      return { x: cx - w / 2, y: down ? ay : ay - h, w, h };
    }

    function dragCrop(drag, dx, dy) {
      const s = drag.start;
      if (drag.mode === "move") {
        return { x: clamp(s.x + dx, 0, imgW - s.w), y: clamp(s.y + dy, 0, imgH - s.h), w: s.w, h: s.h };
      }
      const minSide = Math.min(CROP_MIN_PX, imgW, imgH);
      if (cropPreset) return ratioResize(s, drag.mode, dx, dy, cropPreset.w / cropPreset.h, minSide);
      let L = s.x, R = s.x + s.w, T = s.y, B = s.y + s.h;
      if (drag.mode.includes("w")) L = clamp(L + dx, 0, R - minSide);
      if (drag.mode.includes("e")) R = clamp(R + dx, L + minSide, imgW);
      if (drag.mode.includes("n")) T = clamp(T + dy, 0, B - minSide);
      if (drag.mode.includes("s")) B = clamp(B + dy, T + minSide, imgH);
      return { x: L, y: T, w: R - L, h: B - T };
    }

    // --------------------------- Player boxes --------------------------- //
    function renderBoxes() {
      const on = $sharpen.prop("checked");
      // Boxes are in the unturned photo's frame, so they're hidden (and not
      // clickable) while the Level turns it.
      const pickable = !level && !bodiesUnavailable;
      $playerLayer.toggle(activeTab === "tone" && on && bodies.length > 0 && pickable);
      bodies.forEach((b) => {
        boxEls[b.id].toggleClass("selected", selected.has(b.id)).toggleClass("hover", hoverId === b.id);
      });
      if (bodiesUnavailable) {
        $sharpenHint.text("This photo was turned by an earlier edit, so its players can't be located to sharpen.");
        $sharpenCount.text("");
      } else if (!bodies.length) {
        $sharpenHint.text("No players found in this photo.");
        $sharpenCount.text("");
      } else {
        $sharpenHint.text(!on ? "" : level
          ? "Set the Level back to 0\u00b0 to pick players; the current selection is still sharpened."
          : "Click a box on the photo to include or exclude a player.");
        $sharpenCount.text(on ? `${selected.size} of ${bodies.length} player${bodies.length === 1 ? "" : "s"} selected` : "");
      }
    }

    // Smallest box containing the point, so a box nested inside (or mostly
    // overlapped by) a bigger one can always be picked.
    function boxAt(x, y) {
      let best = null;
      let bestArea = Infinity;
      if (level || bodiesUnavailable) return null;
      bodies.forEach((b) => {
        const [x1, y1, x2, y2] = b.box;
        if (x < x1 || x > x2 || y < y1 || y > y2) return;
        const area = (x2 - x1) * (y2 - y1);
        if (area < bestArea) { best = b; bestArea = area; }
      });
      return best;
    }

    // ---------------------------- Live preview --------------------------- //
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
      q.set("level", String(level));
      return `/api/edit/ai-edit/session/preview?${q.toString()}`;
    }

    function beforeUrl(stamp, forLevel) {
      return `/api/edit/ai-edit/session/before?file=${encodeURIComponent(key)}` +
        `&level=${encodeURIComponent(String(forLevel === undefined ? level : forLevel))}&t=${stamp}`;
    }

    let previewTimer = null;
    let previewSeq = 0;
    function schedulePreview() {
      clearTimeout(previewTimer);
      $previewStatus.removeClass("error").text("Preview will update shortly\u2026");
      previewTimer = setTimeout(() => {
        const seq = ++previewSeq;
        const url = previewUrl();
        const forLevel = level;
        $previewStatus.text("Updating preview\u2026");
        // Preload so the image only swaps once the new render is ready,
        // and drop responses that a newer request has overtaken.
        const img = new Image();
        img.onload = () => {
          if (seq !== previewSeq || !win) return;
          win.setAfterUrl(url);
          // The original is turned with the photo so the two line up.
          if (beforeLevel !== forLevel) {
            beforeLevel = forLevel;
            win.setBeforeUrl(beforeUrl(Date.now(), forLevel));
          }
          $previewStatus.text("");
        };
        img.onerror = () => {
          if (seq !== previewSeq) return;
          $previewStatus.addClass("error").text("Couldn't update the preview. Try moving a slider again.");
        };
        img.src = url;
      }, PREVIEW_DEBOUNCE_MS);
    }

    // --------------------------- Auto checkboxes ------------------------- //
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

    // ------------------------------- Level ------------------------------- //
    function setLevel(value) {
      const next = cleanLevel(value);
      if (next !== level) {
        level = next;
        refreshCanvas();
        refitCrop();
        renderBoxes();
        renderCrop();
        schedulePreview();
      }
      syncLevelControls();
    }

    levelSlider.$input.on("input", () => setLevel(levelSlider.value));
    $levelNumber.on("change", () => setLevel($levelNumber.val()));
    $levelReset.on("click", () => setLevel(0));

    // ------------------------------ Auto crop ---------------------------- //
    // The model's box is in the photo as it is now, unturned; the crop box
    // becomes the box's bounds on the turned photo, in the selected preset's
    // shape if one is, kept inside the photo.
    function applyAutoCrop(result) {
      level = LEVEL_ENABLED ? cleanLevel(result.level) : 0;
      refreshCanvas();
      syncLevelControls();
      const [x1, y1, x2, y2] = result.box;
      const rad = (level * Math.PI) / 180;
      const c = Math.cos(rad);
      const s = Math.sin(rad);
      const xs = [];
      const ys = [];
      [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([nx, ny]) => {
        const rx = nx * srcW - srcW / 2;
        const ry = ny * srcH - srcH / 2;
        xs.push(rx * c + ry * s + imgW / 2);
        ys.push(-rx * s + ry * c + imgH / 2);
      });
      const left = clamp(Math.min(...xs), 0, imgW);
      const top = clamp(Math.min(...ys), 0, imgH);
      let box = { x: left, y: top, w: clamp(Math.max(...xs), 0, imgW) - left, h: clamp(Math.max(...ys), 0, imgH) - top };
      if (cropPreset) box = reshapeCrop(box, cropPreset.w / cropPreset.h);
      crop = shrinkIntoPhoto(box);
      renderBoxes();
      renderCrop();
      schedulePreview();
    }

    // The checkbox: its first check asks the server for a suggestion (kept
    // for later re-checks); unchecking brings back what the level and crop
    // were just before it was applied.
    let autoSuggestion = null;
    let beforeAuto = null;

    function applyAutoSuggestion() {
      beforeAuto = { level, crop: Object.assign({}, crop), preset: cropPreset };
      applyAutoCrop(autoSuggestion);
      $autoCropNote.text(autoSuggestion.reason || "");
    }

    function restoreBeforeAuto() {
      if (beforeAuto) {
        level = beforeAuto.level;
        crop = beforeAuto.crop;
        cropPreset = beforeAuto.preset;
        beforeAuto = null;
        refreshCanvas();
        syncLevelControls();
        renderBoxes();
        renderCrop();
        schedulePreview();
      }
      $autoCropNote.text("");
    }

    function forgetAutoCrop() {
      autoSuggestion = null;
      beforeAuto = null;
      $autoCrop.prop("checked", false);
      $autoCropNote.text("");
      renderAutoLock();
    }

    // While the checkbox is on the suggestion owns the level and crop: those
    // controls (and the crop box on the photo) are locked until it is unchecked.
    function autoLocked() {
      return $autoCrop.prop("checked");
    }

    function renderAutoLock() {
      const locked = autoLocked();
      levelSlider.setEnabled(!locked);
      $levelNumber.prop("disabled", locked);
      $levelReset.prop("disabled", locked);
      $customTile.prop("disabled", locked);
      presetTiles.forEach(({ $tile }) => $tile.prop("disabled", locked));
      cropDrag = null;
      $cropLayer.toggleClass("locked", locked);
    }

    async function requestAutoSuggestion() {
      const j = autoCropDialog();
      setBusy(true);
      $previewStatus.removeClass("error").text("");
      j.reset();
      try {
        await apiPost("/api/edit/ai-edit/auto-crop", {
          file: key,
          ratioH: cropPreset ? cropPreset.h : null,
          ratioW: cropPreset ? cropPreset.w : null,
        });
      } catch (err) {
        setBusy(false);
        $previewStatus.addClass("error").text(`Auto crop: ${errorText(err)}`);
        return null;
      }
      j.open("Auto crop");
      return new Promise((resolve) => {
        j.startPolling((data) => {
          if (data.running) return;
          setBusy(false);
          if (data.error || !data.result) {
            j.appendLines([`Auto crop failed: ${data.error || "no suggestion was returned."}`]);
            j.setFinished("Auto crop failed");
            resolve(null);
            return;
          }
          j.close();
          resolve(data.result);
        });
      });
    }

    $autoCrop.on("change", async () => {
      renderAutoLock();
      if (!$autoCrop.prop("checked")) {
        restoreBeforeAuto();
        return;
      }
      if (!autoSuggestion) {
        autoSuggestion = await requestAutoSuggestion();
        if (!autoSuggestion) {
          $autoCrop.prop("checked", false);
          renderAutoLock();
          return;
        }
      }
      applyAutoSuggestion();
    });

    // -------------------------------- Tabs ------------------------------- //
    function setTab(id) {
      activeTab = id;
      TABS.forEach((t) => {
        const on = t.id === id;
        t.$tab.toggleClass("active", on).attr("aria-selected", String(on));
        t.$pane.prop("hidden", !on);
      });
      if (win) win.setSplit(id !== "crop");
      hoverId = null;
      cropDrag = null;
      renderBoxes();
      renderCrop();
    }

    // ------------------------------ Other tab ---------------------------- //
    function renderOther() {
      $revert.prop("disabled", busy || !session.edited);
      $revertHint.text(session.edited
        ? "Irreversibly delete this photo's edited copy and go back to the original."
        : "This photo has no saved edits; it is already the original.");
    }

    function setBusy(on) {
      busy = on;
      $reload.prop("disabled", on);
      $external.prop("disabled", on);
      $autoCrop.prop("disabled", on);
      $apply.prop("disabled", on);
      renderOther();
    }

    // Takes over a (re)opened session: new photo size, suggested tone and
    // player boxes. The crop and level are kept unless the photo's size
    // changed or `fresh` asks for a clean start (after Revert).
    function loadSession(next, fresh) {
      const sizeChanged = next.width !== srcW || next.height !== srcH;
      session = next;
      proposed = next.proposed;
      srcW = next.width;
      srcH = next.height;
      bodiesUnavailable = !!next.bodiesUnavailable;
      if (fresh || sizeChanged) level = 0;
      if (fresh || sizeChanged) forgetAutoCrop();
      refreshCanvas();
      syncLevelControls();
      if (fresh || sizeChanged) resetCrop();
      bodies = next.bodies || [];
      const ids = new Set(bodies.map((b) => b.id));
      selected = fresh
        ? new Set(bodies.filter((b) => b.isTeammate).map((b) => b.id))
        : new Set([...selected].filter((id) => ids.has(id)));
      hoverId = null;
      buildPlayerBoxes();
      applyBrightnessAuto();
      applyContrastAuto();
      renderBoxes();
      renderCrop();
      renderOther();
      const stamp = Date.now();
      beforeLevel = level;
      if (sizeChanged) win.setViewportSize({ w: srcW, h: srcH });
      win.setBeforeUrl(beforeUrl(stamp));
      win.setAfterUrl(`${previewUrl()}&t=${stamp}`);
    }

    async function reopenSession(fresh, doneText) {
      setBusy(true);
      $previewStatus.removeClass("error").text("Reloading the photo\u2026");
      try {
        loadSession(await apiPost("/api/edit/ai-edit/session", { file: key }), fresh);
        $previewStatus.text(doneText || "");
      } catch (err) {
        $previewStatus.addClass("error").text(`Couldn't reload the photo: ${errorText(err)}`);
      } finally {
        setBusy(false);
      }
    }

    $reload.on("click", () => reopenSession(false, "Reloaded."));

    $revert.on("click", async () => {
      const yes = await confirmDialog(
        "Revert to the original?",
        "This irreversibly abandons every edit made to this photo so far and deletes the edited copy. Continue?"
      );
      if (!yes) return;
      setBusy(true);
      try {
        await apiPost("/api/edit/revert", { file: key });
        flow.changed = true;
      } catch (err) {
        setBusy(false);
        $previewStatus.addClass("error").text(`Couldn't revert: ${errorText(err)}`);
        return;
      }
      await reopenSession(true, "Reverted to the original.");
    });

    $external.on("click", async () => {
      setBusy(true);
      $previewStatus.removeClass("error").text("Opening\u2026");
      try {
        const res = await apiPost("/api/edit/open-external", { file: key });
        flow.changed = true;
        if (res.created) session.edited = true;
        $previewStatus.text("Pick an app to edit with. After saving there, click Reload.");
      } catch (err) {
        $previewStatus.addClass("error").text(`Couldn't open another app: ${errorText(err)}`);
      } finally {
        setBusy(false);
      }
    });

    // ------------------------------- Window ------------------------------ //
    buildPlayerBoxes();
    applyBrightnessAuto();
    applyContrastAuto();
    syncLevelControls();

    const stamp = Date.now();
    win = Viewport.showCompareWindow({
      title: `Edit photo \u2014 ${session.file}`,
      beforeUrl: beforeUrl(stamp),
      afterUrl: previewUrl() + `&t=${stamp}`,
      beforeLabel: "Original",
      afterLabel: "Preview",
      // The window keeps the unturned photo's shape however the Level turns it.
      viewportSize: { w: srcW, h: srcH },
      sidePanel: $panel,
      overlay: $overlay,
      onImageClick(x, y) {
        if (activeTab !== "tone" || !$sharpen.prop("checked")) return false;
        const b = boxAt(x, y);
        if (!b) return false;
        if (selected.has(b.id)) selected.delete(b.id);
        else selected.add(b.id);
        renderBoxes();
        return true;
      },
      onImageHover(x, y) {
        if (x != null && activeTab === "crop") {
          const mode = autoLocked() ? null : cropHitTest(x, y);
          return mode ? CROP_CURSORS[mode] : false;
        }
        const b = x == null || activeTab !== "tone" || !$sharpen.prop("checked") ? null : boxAt(x, y);
        const id = b ? b.id : null;
        if (id !== hoverId) { hoverId = id; renderBoxes(); }
        return !!b;
      },
      onPointerDown(x, y) {
        if (activeTab !== "crop" || autoLocked()) return false;
        const mode = cropHitTest(x, y);
        if (!mode) return false;
        cropDrag = { mode, x0: x * imgW, y0: y * imgH, start: Object.assign({}, crop) };
        return true;
      },
      onPointerMove(x, y) {
        if (!cropDrag) return;
        crop = dragCrop(cropDrag, x * imgW - cropDrag.x0, y * imgH - cropDrag.y0);
        renderCrop();
      },
      onPointerUp() {
        cropDrag = null;
      },
      onViewChange(zoom) {
        $overlay[0].style.setProperty("--edit-inv-zoom", String(1 / zoom));
      },
    });
    setTab("tone");
    renderOther();

    function closeEditor() {
      clearTimeout(previewTimer);
      previewSeq++;
      win.close();
      win = null;
    }

    $cancel.on("click", () => {
      closeEditor();
      fetch("/api/edit/ai-edit/session", { method: "DELETE" }).catch(() => {});
      finish(flow, "cancelled");
    });

    $apply.on("click", () => {
      const sharpen = $sharpen.prop("checked");
      const request = Object.assign(
        { file: key, sharpen, bodyIds: sharpen ? [...selected] : [], crop: isCropped() ? cropPixels() : null, level },
        currentTone()
      );
      const chosenPrefs = {
        brightnessAuto: brightAuto.$input.prop("checked"),
        contrastAuto: contrastAuto.$input.prop("checked"),
        sharpen,
      };
      closeEditor();
      apply(request, chosenPrefs, flow);
    });
  }

  // ---------------------------------------------------------------- //
  // Step 2: apply (log console)
  // ---------------------------------------------------------------- //
  async function apply(request, prefs, flow) {
    const j = dialog();
    j.reset();
    j.open("Applying edits\u2026");
    try {
      await apiPost("/api/edit/ai-edit/apply", request);
    } catch (err) {
      j.appendLines([`Failed to start: ${err.message}`]);
      fetch("/api/edit/ai-edit/session", { method: "DELETE" }).catch(() => {});
      j.setFinished("Edit failed to start");
      finish(flow, "failed", errorText(err));
      return;
    }
    j.startPolling((data) => {
      if (data.running) return;
      if (data.returnCode === 0) {
        j.close();
        openCompare(prefs, flow);
      } else {
        j.setFinished(`Edit failed (exit code ${data.returnCode})`);
        finish(flow, "failed", `Edit failed (exit code ${data.returnCode})`);
      }
    });
  }

  // ---------------------------------------------------------------- //
  // Step 3: review — the edit is only recorded on the album entry once
  // saved (Save), and the generated file is deleted on Abort (see
  // /api/edit/ai-edit/accept|reject in culling_app.py). Save Copy instead
  // keeps the result as a new photo in the album and leaves this one alone
  // (/api/edit/ai-edit/save-copy).
  // ---------------------------------------------------------------- //
  async function openCompare(prefs, flow) {
    let state;
    try {
      state = await apiGet("/api/edit/ai-edit/pending");
    } catch (err) {
      state = null;
    }
    if (!state || !state.pending) {
      reopenLog("The edit finished, but no image was produced.", "Edit produced no image");
      finish(flow, "failed", "The edit finished, but no image was produced.");
      return;
    }
    const stamp = Date.now();
    Viewport.showCompareWindow({
      title: `Review edit \u2014 ${state.file}`,
      beforeUrl: `/api/edit/ai-edit/image?side=before&t=${stamp}`,
      afterUrl: `/api/edit/ai-edit/image?side=after&t=${stamp}`,
      beforeLabel: "Before",
      afterLabel: "After",
      acceptLabel: "Save",
      rejectLabel: "Abort",
      extraButtons: [{
        label: "Save Copy",
        onClick: () => apiPost("/api/edit/ai-edit/save-copy", {}).then((res) => {
          finish(flow, "copied", null, { copy: { key: res.key, path: res.path } });
        }).catch((err) => {
          alert(`Failed to save a copy: ${errorText(err)}`);
          finish(flow, "failed", errorText(err));
        }),
      }],
      onAccept: () => apiPost("/api/edit/ai-edit/accept", {}).then(() => {
        savePrefs(prefs);
        finish(flow, "accepted");
      }).catch((err) => {
        alert(`Failed to keep the edit: ${err.message}`);
        finish(flow, "failed", errorText(err));
      }),
      onReject: () => apiPost("/api/edit/ai-edit/reject", {}).then(() => {
        finish(flow, "rejected");
      }).catch((err) => {
        alert(`Failed to discard the edit: ${err.message}`);
        finish(flow, "failed", errorText(err));
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

  // Reports how one run() ended to its onResult callback — exactly once, so
  // a late second report (e.g. an error after the verdict) is ignored. The
  // callback is never allowed to break the edit flow.
  function finish(flow, outcome, error, extra) {
    if (flow.finished) return;
    flow.finished = true;
    if (!flow.onResult) return;
    const result = {
      file: flow.key,
      outcome,
      // The photo's file changed on disk during this run (a kept edit, a
      // revert, or it was opened in another app): redisplay it.
      changed: outcome === "accepted" || flow.changed,
    };
    if (error) result.error = error;
    Object.assign(result, extra);
    try {
      flow.onResult(result);
    } catch (err) {
      console.error("AiEdit onResult callback failed:", err);
    }
  }

  // Opens edit mode for `key` (the Album's bookkeeping key / plain
  // filename — same identifier used by /api/original etc.).
  //
  // `opts.onResult(result)` is called once when the run ends, with
  //   { file, outcome, changed, error? }
  // where `outcome` is
  //   "accepted"  — the edit was applied and kept (Save in the review),
  //   "rejected"  — it was applied but discarded (Abort in the review),
  //   "copied"    — the edit was saved as a new photo in the album (Save
  //                 Copy); `copy: { key, path }` identifies it. The photo
  //                 that was edited is unchanged,
  //   "cancelled" — the editor was closed with Cancel (nothing applied),
  //   "failed"    — the photo couldn't be opened or the edit couldn't be
  //                 produced/saved (`error` says why),
  // and `changed` is true when the photo's displayed file may now differ
  // (accepted, or Revert / Edit with custom app was used along the way), so
  // the caller should reload it.
  async function run(key, opts) {
    const flow = { key, onResult: opts && opts.onResult, changed: false, finished: false };
    const done = showLoading("Opening photo\u2026");
    let session;
    try {
      session = await apiPost("/api/edit/ai-edit/session", { file: key });
    } catch (err) {
      done();
      alert(`Could not open the photo for editing: ${err.message}`);
      finish(flow, "failed", errorText(err));
      return;
    }
    done();
    openEditor(key, session, flow);
  }

  return { run };
})();
