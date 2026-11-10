"use strict";

// Shared "viewport" zoom/pan math + a reusable "frame" (modal image window)
// built on top of it. Used by the culling page (viewport embedded directly
// in the page, no frame) and the faces page (viewport hosted inside a
// modal frame). See static/js/culling.js and static/js/faces.js.
const Viewport = (function () {
  // Given an image's natural size and the available space to display it
  // in, compute the classic object-fit: contain box (the "zoom-to-fit"
  // size), in pixels.
  function computeContainFit(naturalW, naturalH, availW, availH) {
    if (!naturalW || !naturalH || !availW || !availH) return null;
    const scale = Math.min(availW / naturalW, availH / naturalH);
    return { width: Math.round(naturalW * scale), height: Math.round(naturalH * scale) };
  }

  // A zoom/pan controller: holds the current zoom level + pan offset and
  // applies them (via a CSS transform) to whatever set of <img> elements is
  // passed in at call time. Callers own which images are "live" (e.g. the
  // culling page re-queries its currently displayed preview cells so up to 4
  // images can share one zoom/pan state), so the same controller instance
  // can be reused across image/group navigation (and, on the culling page,
  // across the annotated/original toggle — see toggleViewMode/renderMain in
  // culling.js) without resetting.
  //
  // Pan is tracked as a `translate(tx, ty) scale(zoom)` transform (in that
  // order) with transform-origin pinned to the image's top-left corner, so
  // tx/ty are plain screen pixels regardless of zoom: dragging the mouse by
  // (dx, dy) always just adds (dx, dy) to (tx, ty), and "zoom centered on a
  // point" is solved by picking a new tx/ty that keeps that point's screen
  // position unchanged (see zoomTo below). Since the <img> itself is always
  // sized to exactly fill its (fixed-size, non-scaling) viewport parent at
  // zoom 1, that parent's rect doubles as the img's un-transformed base
  // rect for both of those calculations.
  function createZoomController({ min = 1, max = 6, step = 0.25 } = {}) {
    let zoom = min;
    let tx = 0;
    let ty = 0;
    // Share of the viewport's width the picture covers at fit zoom (below 1
    // when it is letterboxed in a fixed viewport), so "actual size" is right.
    let contentFraction = 1;

    function baseRectOf(imgEl) {
      return imgEl.parentElement.getBoundingClientRect();
    }

    // Keeps the pan offset within the range that leaves the (scaled) image
    // fully covering its viewport — i.e. no empty gap at any edge.
    function clamp(imgEl) {
      const r = baseRectOf(imgEl);
      tx = Math.min(0, Math.max(r.width * (1 - zoom), tx));
      ty = Math.min(0, Math.max(r.height * (1 - zoom), ty));
    }

    function apply($imgs) {
      if (zoom <= min) {
        tx = 0;
        ty = 0;
        $imgs.removeClass("zoomed").css({ transform: "", "transform-origin": "" });
        return;
      }
      $imgs.addClass("zoomed").css({
        transform: `translate(${tx}px, ${ty}px) scale(${zoom})`,
        "transform-origin": "0 0",
      });
    }

    function resetToFit($imgs) {
      zoom = min;
      apply($imgs);
    }

    // Changes zoom to `newZoom`, keeping the point under (clientX, clientY)
    // — relative to `imgEl` — fixed on screen, if given; otherwise zooms
    // centered on the image itself.
    function zoomTo($imgs, imgEl, newZoom, clientX, clientY) {
      newZoom = Math.min(max, Math.max(min, newZoom));
      if (imgEl) {
        const r = baseRectOf(imgEl);
        const cx = clientX != null ? clientX : r.left + r.width / 2;
        const cy = clientY != null ? clientY : r.top + r.height / 2;
        const localX = (cx - r.left - tx) / zoom;
        const localY = (cy - r.top - ty) / zoom;
        tx += localX * (zoom - newZoom);
        ty += localY * (zoom - newZoom);
      }
      zoom = newZoom;
      if (imgEl) clamp(imgEl);
      apply($imgs);
    }

    // Zoom to 100% (actual pixel size) of `imgEl`, anchored on (clientX,
    // clientY) if given, otherwise centered on the image.
    function zoomToActual($imgs, imgEl, clientX, clientY) {
      if (!imgEl || !imgEl.naturalWidth) return;
      const rect = baseRectOf(imgEl);
      const nativeZoom = imgEl.naturalWidth / (rect.width * contentFraction);
      zoomTo($imgs, imgEl, nativeZoom, clientX, clientY);
    }

    function adjustByStep($imgs, delta, imgEl, clientX, clientY) {
      zoomTo($imgs, imgEl, zoom + delta, clientX, clientY);
    }

    // Click-to-toggle: zooms to actual size if currently at fit, or back to
    // fit if already zoomed in.
    function toggleClick($imgs, imgEl, clientX, clientY) {
      if (zoom > min) {
        resetToFit($imgs);
        return;
      }
      zoomToActual($imgs, imgEl, clientX, clientY);
    }

    // Drag-to-pan. Starts tracking a drag anchored at (clientX, clientY);
    // returns false (and starts nothing) while at "fit" zoom, since there's
    // no room to pan there. `imgEl` is remembered only to know which
    // viewport to clamp against as the drag proceeds.
    let dragImg = null;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragStartTx = 0;
    let dragStartTy = 0;

    function dragStart(imgEl, clientX, clientY) {
      if (zoom <= min) return false;
      dragImg = imgEl;
      dragStartX = clientX;
      dragStartY = clientY;
      dragStartTx = tx;
      dragStartTy = ty;
      return true;
    }

    function dragMove($imgs, clientX, clientY) {
      if (!dragImg) return;
      tx = dragStartTx + (clientX - dragStartX);
      ty = dragStartTy + (clientY - dragStartY);
      clamp(dragImg);
      apply($imgs);
    }

    function dragEnd() {
      dragImg = null;
    }

    function isZoomed() {
      return zoom > min;
    }

    return {
      apply, resetToFit, zoomToActual, adjustByStep, toggleClick, isZoomed,
      dragStart, dragMove, dragEnd,
      get min() { return min; },
      get max() { return max; },
      get step() { return step; },
      set contentFraction(v) { contentFraction = v; },
      // Current zoom level and pan offset (screen px), so callers that need
      // to translate between on-screen position and an image's own local
      // coordinate space (e.g. the compare-window slider handle, which
      // stays at a fixed container-relative position while the images
      // underneath zoom/pan) can do the math themselves.
      get zoom() { return zoom; },
      get tx() { return tx; },
      get ty() { return ty; },
    };
  }

  // A self-contained modal "frame" hosting a single-image viewport: darkens
  // the background, locks page scroll, and shows a close [x] button. Zoom
  // interactions over the image match the shared viewport behavior (scroll
  // to zoom, drag to pan while zoomed, click to toggle fit/actual, '1' to
  // zoom to actual). Clicking anywhere outside the image, the close button,
  // or any other key (including Esc) closes the window immediately.
  //
  // `images` is an array of `{url, originalPath, aiEditKey}` (the latter two
  // optional; faces.js is the only caller and always passes them). `index`
  // is which one to show first. When there's more than one image, Prev/Next
  // buttons + a counter let the caller's whole group be paged through
  // in-place instead of closing/reopening the window per image — arrow keys
  // work too. Clicks/keystrokes inside the toolbar (path box / AI edit
  // button / nav buttons) are excluded from the "anything closes this
  // window" behavior below, so the path can be selected/copied and the
  // buttons clicked without the window vanishing first.
  //
  // `opts.onAiEditResult(result)` (optional) is called with AiEdit.run's
  // result when an AI edit launched from this window ends. Whenever the
  // photo changed (`result.changed`), the window itself first reloads the
  // current image so it shows the edited (or reverted) photo.
  function showImageWindow(images, index, opts) {
    images = images || [];
    opts = opts || {};
    let current = Math.max(0, Math.min(images.length - 1, index || 0));
    const zoomCtl = createZoomController({ min: 1, max: 6, step: 0.25 });

    const $viewport = $("<div>", { class: "image-window-viewport" });
    const $img = $("<img>", { src: (images[current] || {}).url || "", alt: "" });
    $viewport.append($img);
    const $closeBtn = $("<button>", { type: "button", class: "image-window-close", "aria-label": "Close" }).html("&times;");
    const $backdrop = $("<div>", { class: "image-window-backdrop" }).append($viewport, $closeBtn);

    const hasToolbar = images.some((it) => it.originalPath || it.aiEditKey);
    let $toolbar = null;
    let $pathInput = null;
    let currentAiEditKey = null;
    if (hasToolbar) {
      $toolbar = $("<div>", { class: "image-window-toolbar" });
      $pathInput = $("<input>", { type: "text", class: "image-window-path", readonly: true });
      $toolbar.append($pathInput);
      $toolbar.append(
        $("<button>", { type: "button", class: "btn btn-sm" }).text("AI edit")
          .on("click", () => { if (currentAiEditKey) AiEdit.run(currentAiEditKey, { onResult: onAiEditResult }); })
      );
      $backdrop.append($toolbar);
    }

    let $prevBtn = null;
    let $nextBtn = null;
    let $counter = null;
    if (images.length > 1) {
      $prevBtn = $("<button>", { type: "button", class: "image-window-nav-prev", "aria-label": "Previous photo" }).html("&lsaquo;");
      $nextBtn = $("<button>", { type: "button", class: "image-window-nav-next", "aria-label": "Next photo" }).html("&rsaquo;");
      // Keep focus off the nav buttons so a stray Enter/Space after a click
      // doesn't re-trigger them via the generic keydown-closes handler.
      $prevBtn.add($nextBtn).on("mousedown", (e) => e.preventDefault());
      $prevBtn.on("click", () => goTo(current - 1));
      $nextBtn.on("click", () => goTo(current + 1));
      $viewport.append($prevBtn, $nextBtn);
      $counter = $("<span>", { class: "image-window-nav-counter" });
      ($toolbar || $backdrop).append($counter);
    }

    $("body").append($backdrop).addClass("image-window-open");

    function imgs() {
      return $viewport.find("img");
    }

    function lock() {
      const box = computeContainFit($img[0].naturalWidth, $img[0].naturalHeight, window.innerWidth * 0.9, window.innerHeight * 0.9);
      if (box) $viewport.css({ width: box.width + "px", height: box.height + "px" });
    }

    function updateToolbar() {
      const item = images[current] || {};
      if ($pathInput) $pathInput.val(item.originalPath || "");
      currentAiEditKey = item.aiEditKey || null;
    }

    // The server always serves the current (edited, if any) version at the
    // same URL, so after an edit the browser has to be told to refetch: a
    // changing query string defeats its cache, and a kept edit may be a
    // different size (crop), so zoom/pan start over.
    function onAiEditResult(result) {
      if (result.changed) {
        const i = images.findIndex((it) => it.aiEditKey === result.file);
        if (i >= 0) {
          const base = images[i].url.split("?")[0];
          images[i].url = `${base}?t=${Date.now()}`;
          if (i === current) {
            $img.off("load").one("load", function () {
              lock();
              zoomCtl.resetToFit(imgs());
            });
            $img.attr("src", images[i].url);
          }
        }
      }
      if (opts.onAiEditResult) opts.onAiEditResult(result);
    }

    function updateNavUI() {
      if (!$counter) return;
      $counter.text(`${current + 1} / ${images.length}`);
      $prevBtn.prop("disabled", current === 0);
      $nextBtn.prop("disabled", current === images.length - 1);
    }

    // Clamped, no wrap-around. Preserves the current zoom/pan across
    // navigation when the new image is the same size as the old one (the
    // common case — same camera, same burst) since the transform is still
    // valid for it; resets to fit only when the size actually changes.
    // Neighbours are prefetched since /api/original serves full-resolution
    // files.
    function goTo(newIndex) {
      newIndex = Math.max(0, Math.min(images.length - 1, newIndex));
      if (newIndex === current) return;
      const priorW = $img[0].naturalWidth;
      const priorH = $img[0].naturalHeight;
      current = newIndex;
      $img.off("load").one("load", function () {
        lock();
        if (this.naturalWidth !== priorW || this.naturalHeight !== priorH) zoomCtl.resetToFit(imgs());
      });
      $img.attr("src", images[current].url);
      updateToolbar();
      updateNavUI();
      [current - 1, current + 1].forEach((i) => {
        if (images[i]) new Image().src = images[i].url;
      });
    }

    updateToolbar();
    updateNavUI();
    if ($img[0].complete) lock();
    else $img.one("load", lock);
    $(window).on("resize.imageWindow", lock);

    // Drag-to-pan while zoomed in. `dragging` tracks whether a drag started
    // on the image is still in progress; `dragMoved` distinguishes an actual
    // drag from a plain click so the click handler below (fit/actual toggle)
    // doesn't also fire once the drag ends.
    let dragging = false;
    let dragMoved = false;
    $viewport.on("mousedown", "img", function (e) {
      e.preventDefault(); // suppress the browser's native image-ghost drag
      dragMoved = false;
      dragging = zoomCtl.dragStart(this, e.clientX, e.clientY);
    });
    $(document).on("mousemove.imageWindowDrag", function (e) {
      if (!dragging) return;
      dragMoved = true;
      zoomCtl.dragMove(imgs(), e.clientX, e.clientY);
    });
    $(document).on("mouseup.imageWindowDrag", function () {
      dragging = false;
    });
    $viewport.on("wheel", "img", function (e) {
      e.preventDefault();
      const oe = e.originalEvent;
      const delta = oe.deltaY < 0 ? zoomCtl.step : -zoomCtl.step;
      zoomCtl.adjustByStep(imgs(), delta, this, oe.clientX, oe.clientY);
    });
    $viewport.on("click", "img", function (e) {
      if (dragMoved) {
        dragMoved = false;
        return;
      }
      zoomCtl.toggleClick(imgs(), this, e.clientX, e.clientY);
    });

    // Clicking outside the viewport (the backdrop) or the close button closes.
    // Clicks inside the toolbar (path box / AI edit button) are excluded too.
    $backdrop.on("click", function (e) {
      if ($(e.target).closest(".image-window-viewport, .image-window-toolbar").length) return;
      close();
    });
    $closeBtn.on("click", close);

    // Any keydown closes the window, except '1' (zoom to actual, matching
    // the shared viewport hotkey), Ctrl/Cmd+E (AI edit, matching the
    // toolbar button) and Left/A / Right/D (navigate, when there's a
    // group) — captured ahead of any other page-level keydown handler
    // (e.g. the cluster page's Delete-key handler) so it never also fires
    // while the window is open.
    function onKeydown(e) {
      // Let normal typing/selection (e.g. Ctrl+A/Ctrl+C on the readonly path
      // box) work while focus is in the toolbar, instead of closing the
      // window on the very first keystroke. Same for keys aimed at a compare
      // / edit window opened on top of this one (e.g. its sliders).
      if ($(e.target).closest(".image-window-toolbar, .compare-window-backdrop").length) return;
      e.stopPropagation();
      if (e.key === "1") {
        zoomCtl.zoomToActual(imgs(), $img[0]);
        return;
      }
      const key = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && key === "e") {
        e.preventDefault();
        if (currentAiEditKey) AiEdit.run(currentAiEditKey, { onResult: onAiEditResult });
        return;
      }
      if (images.length > 1) {
        if (key === "arrowleft" || key === "a") {
          goTo(current - 1);
          return;
        }
        if (key === "arrowright" || key === "d") {
          goTo(current + 1);
          return;
        }
      }
      e.preventDefault();
      close();
    }
    document.addEventListener("keydown", onKeydown, true);

    function close() {
      document.removeEventListener("keydown", onKeydown, true);
      $(document).off("mousemove.imageWindowDrag mouseup.imageWindowDrag");
      $(window).off("resize.imageWindow");
      $("body").removeClass("image-window-open");
      $backdrop.remove();
    }

    return { close };
  }


  // A modal "compare" window: a single viewport overlaying the before/after
  // images, with a draggable vertical handle that clips the "after" image to
  // reveal more or less of it over the "before" image underneath (the
  // classic before/after slider pattern). Both images share a SINGLE zoom
  // controller, so zooming/panning stays in lockstep between them (they're
  // always laid out at identical pixel sizes, which is what makes one
  // shared transform valid for both). Used by the AI-edit flow to review a
  // generated image against what it was made from (see static/js/ai-edit.js).
  //
  // Unlike showImageWindow this window is deliberately NOT dismissable by
  // clicking away, Esc, or any other key: the caller needs an explicit
  // verdict, so the only ways out are the two action buttons.
  // `opts`: { title, beforeUrl, afterUrl, beforeLabel, afterLabel,
  //           acceptLabel, rejectLabel, onAccept, onReject, extraButtons }
  // — the handlers may return a promise; the window closes once it settles.
  // `extraButtons` is an optional array of `{ label, onClick }` shown between
  // the reject and accept buttons, handled the same way.
  //
  // Optional extras (used by the edit mode in static/js/ai-edit.js):
  //   sidePanel      — element shown to the right of the viewport; the
  //                    Accept/Reject buttons are then omitted and the
  //                    caller closes the window via the returned close().
  //   overlay        — element laid over the images that zooms/pans with
  //                    them (fill it with %-positioned children). It never
  //                    receives pointer events itself; use the two hooks
  //                    below instead.
  //   onImageClick(x, y)  — click on the image at normalised (0-1) image
  //                    coordinates; return true if handled, which skips the
  //                    usual fit/actual zoom toggle.
  //   onImageHover(x, y)  — pointer moved over the image (x/y null when it
  //                    left); return true to show a pointer cursor, or a CSS
  //                    cursor name (e.g. "nwse-resize") to show that one.
  //   onPointerDown(x, y, e) — mouse button pressed over the image; return
  //                    true to claim the drag, which then reports every move
  //                    to onPointerMove(x, y, e) and the release to
  //                    onPointerUp(x, y, e) (coordinates not clamped to the
  //                    image) instead of panning, and swallows the click.
  //   onViewChange(zoom)  — called whenever zoom/pan/size changes, so
  //                    overlay content can keep a constant on-screen size.
  //   viewportSize     — {w, h}: keep the viewport box the shape of this size
  //                    whatever the pictures' own size, letterboxing them
  //                    (see setViewportSize below).
  // Returns { close, setAfterUrl, setBeforeUrl, setSplit, pxToImage,
  // setViewportSize, zoom }.
  function showCompareWindow(opts) {
    opts = opts || {};
    const HINT_SPLIT = "Drag the handle to compare \u2014 scroll to zoom, drag image to pan";
    const HINT_SINGLE = "Scroll to zoom, drag image to pan";
    const zoomCtl = createZoomController({ min: 1, max: 6, step: 0.25 });
    let sliderPct = 50;
    // false = single-image mode: only the after image, no slider (setSplit).
    let split = true;
    const hasPanel = !!opts.sidePanel;

    const $beforeImg = $("<img>", { src: opts.beforeUrl, alt: opts.beforeLabel || "Before", class: "compare-slider-before" });
    const $afterImg = $("<img>", { src: opts.afterUrl, alt: opts.afterLabel || "After", class: "compare-slider-after" });
    const $handle = $("<div>", { class: "compare-slider-handle", tabindex: "0", role: "slider",
      "aria-label": "Before/after position", "aria-valuemin": "0", "aria-valuemax": "100" })
      .append($("<div>", { class: "compare-slider-handle-grip" }).text("\u276e\u276f"));
    const $viewport = $("<div>", { class: "compare-window-viewport" }).append(
      $beforeImg,
      $afterImg,
      $("<div>", { class: "compare-slider-badge compare-slider-badge-before" }).text(opts.beforeLabel || "Before"),
      $("<div>", { class: "compare-slider-badge compare-slider-badge-after" }).text(opts.afterLabel || "After"),
      $handle
    );
    const $overlay = opts.overlay ? $(opts.overlay).addClass("compare-window-overlay") : $();
    $handle.before($overlay);

    const $reject = $("<button>", { type: "button", class: "btn btn-danger" }).text(opts.rejectLabel || "Reject");
    const $accept = $("<button>", { type: "button", class: "btn btn-primary" }).text(opts.acceptLabel || "Accept");
    // Optional extra buttons between the two (opts.extraButtons).
    const $extras = (opts.extraButtons || []).map((b) =>
      $("<button>", { type: "button", class: "btn" }).text(b.label).on("click", () => decide(b.onClick))
    );
    const $toolbar = $("<div>", { class: "compare-window-toolbar" }).append(
      $("<span>", { class: "compare-window-hint" }).text(HINT_SPLIT)
    );
    if (!hasPanel) $toolbar.append($reject, ...$extras, $accept);
    const $body = $("<div>", { class: "compare-window-body" }).append($viewport);
    if (hasPanel) $body.append($(opts.sidePanel).addClass("compare-window-panel"));
    const $backdrop = $("<div>", { class: "image-window-backdrop compare-window-backdrop" }).append(
      $("<div>", { class: "compare-window-title" }).text(opts.title || "Compare"),
      $body,
      $toolbar
    );
    $("body").append($backdrop).addClass("image-window-open");

    // Everything that follows the shared zoom/pan transform.
    function imgs() {
      return $viewport.find("img").add($overlay);
    }

    // Client point -> normalised (0-1) image coordinates, undoing zoom/pan
    // (and the letterbox of a fixed viewport, see `content` below).
    function toImageCoords(clientX, clientY) {
      const rect = $viewport[0].getBoundingClientRect();
      if (!rect.width || !rect.height) return null;
      const x = (clientX - rect.left - zoomCtl.tx) / zoomCtl.zoom / rect.width;
      const y = (clientY - rect.top - zoomCtl.ty) / zoomCtl.zoom / rect.height;
      return { x: (x - content.x) / content.w, y: (y - content.y) / content.h };
    }

    // Moves the handle to `pct` (0-100) of the box width and re-clips the
    // after-image so its clip boundary lines up with the handle on screen
    // (see syncClip() below) — the before-image is the full, unclipped base
    // layer underneath, so it shows through to the left of the boundary.
    function setSlider(pct) {
      sliderPct = Math.min(100, Math.max(0, pct));
      $handle.css("left", sliderPct + "%");
      $handle.attr("aria-valuenow", Math.round(sliderPct));
      syncClip();
    }

    // The handle itself is a plain sibling element, not subject to the
    // images' zoom/pan transform, so it always sits at a fixed
    // container-relative position (sliderPct% of the box). But clip-path
    // percentages on the after-image are resolved against the image's OWN
    // (untransformed) border box, and that clipped result is what then gets
    // zoomed/panned — so a naive `inset(0 0 0 sliderPct%)` clip boundary
    // drifts away from the handle as soon as zoom/pan changes. This inverts
    // the image's current transform to find which fraction of the image's
    // own box currently projects onto the handle's fixed screen position,
    // and clips there instead, keeping the visible before/after boundary
    // pinned under the handle at any zoom level or pan offset.
    function syncClip() {
      const width = $viewport.width();
      if (!width) return;
      if (!split) {
        $afterImg.css("clip-path", "none");
        notifyView();
        return;
      }
      const handlePx = (sliderPct / 100) * width;
      const localPct = ((handlePx - zoomCtl.tx) / zoomCtl.zoom / width) * 100;
      $afterImg.css("clip-path", `inset(0 0 0 ${Math.min(100, Math.max(0, localPct))}%)`);
      notifyView();
    }

    // Every zoom/pan/resize ends in syncClip(), so this is the one place
    // that tells the caller the view changed.
    function notifyView() {
      if (opts.onViewChange) opts.onViewChange(zoomCtl.zoom);
    }

    // A fixed viewport (opts.viewportSize / setViewportSize): the box keeps
    // the shape of that reference size however the picture's own size changes
    // (the edit panel's Level turns the photo), and the picture is letterboxed
    // inside it. `content` is the part of the box the picture covers at fit
    // zoom, as fractions of it, so the overlay and pointer maths still line up.
    let viewportSize = opts.viewportSize || null;
    let content = { x: 0, y: 0, w: 1, h: 1 };

    function updateContent() {
      const img = $afterImg[0];
      const vw = $viewport.width();
      const vh = $viewport.height();
      if (viewportSize && img.naturalWidth && vw && vh) {
        const scale = Math.min(vw / img.naturalWidth, vh / img.naturalHeight);
        const w = (img.naturalWidth * scale) / vw;
        const h = (img.naturalHeight * scale) / vh;
        content = { x: (1 - w) / 2, y: (1 - h) / 2, w, h };
      } else {
        content = { x: 0, y: 0, w: 1, h: 1 };
      }
      zoomCtl.contentFraction = content.w;
      $overlay.css({
        left: `${content.x * 100}%`, top: `${content.y * 100}%`,
        width: `${content.w * 100}%`, height: `${content.h * 100}%`,
      });
    }

    // The viewport box is sized from whichever image reports a natural size
    // first, so before/after stay pixel-identical even if they differ
    // slightly (the odd one out just letterboxes) — or from the fixed
    // reference size, when there is one.
    function lock() {
      const found = [$beforeImg[0], $afterImg[0]].find((el) => el.naturalWidth);
      const ref = viewportSize
        ? { naturalWidth: viewportSize.w, naturalHeight: viewportSize.h }
        : found;
      if (!ref) return;
      let availW = window.innerWidth * 0.9;
      let availH = window.innerHeight * 0.74;
      if (hasPanel) {
        const panelW = $body.children(".compare-window-panel").outerWidth() || 0;
        availW = window.innerWidth * 0.94 - panelW - 16;
        availH = window.innerHeight * 0.8;
      }
      const box = computeContainFit(ref.naturalWidth, ref.naturalHeight, availW, availH);
      if (!box) return;
      $viewport.css({ width: box.width + "px", height: box.height + "px" });
      updateContent();
      zoomCtl.apply(imgs());
      setSlider(sliderPct);
    }
    // Re-locked on every load, not just the first: setAfterUrl/setBeforeUrl
    // may swap in a photo of a different size (the edit panel's Reload).
    [$beforeImg, $afterImg].forEach(($img) => {
      $img.on("load", lock);
      if ($img[0].complete) lock();
    });
    $(window).on("resize.compareWindow", lock);

    // Handle drag: tracks the viewport box's rect at drag-start and
    // recomputes sliderPct from the pointer's X position on every move.
    // Kept entirely separate from the zoom/pan drag below (different
    // element, different flag) so the two never interfere with each other.
    let handleDragging = false;
    $handle.on("mousedown", function (e) {
      e.preventDefault();
      handleDragging = true;
    });
    $(document).on("mousemove.compareSliderDrag", function (e) {
      if (!handleDragging) return;
      const rect = $viewport[0].getBoundingClientRect();
      setSlider(((e.clientX - rect.left) / rect.width) * 100);
    });
    $(document).on("mouseup.compareSliderDrag", function () {
      handleDragging = false;
    });
    $handle.on("keydown", function (e) {
      if (e.key === "ArrowLeft") { e.preventDefault(); setSlider(sliderPct - 2); }
      else if (e.key === "ArrowRight") { e.preventDefault(); setSlider(sliderPct + 2); }
    });

    // Zoom/pan drag (scroll-to-zoom, drag-to-pan while zoomed, click to
    // toggle fit/actual) — identical to showImageWindow, just wired to both
    // images at once so they stay in lockstep.
    let dragging = false;
    let dragMoved = false;
    // A drag claimed by opts.onPointerDown (e.g. moving a crop box).
    let pointerDrag = false;
    $viewport.on("mousedown", "img", function (e) {
      e.preventDefault();
      dragMoved = false;
      if (opts.onPointerDown && e.button === 0) {
        const pt = toImageCoords(e.clientX, e.clientY);
        if (pt && opts.onPointerDown(pt.x, pt.y, e)) {
          pointerDrag = true;
          dragMoved = true;  // swallow the click that follows
          return;
        }
      }
      dragging = zoomCtl.dragStart(this, e.clientX, e.clientY);
    });
    $(document).on("mousemove.compareWindowDrag", function (e) {
      if (pointerDrag) {
        const pt = toImageCoords(e.clientX, e.clientY);
        if (pt && opts.onPointerMove) opts.onPointerMove(pt.x, pt.y, e);
        return;
      }
      if (!dragging) return;
      dragMoved = true;
      zoomCtl.dragMove(imgs(), e.clientX, e.clientY);
      syncClip();
    });
    $(document).on("mouseup.compareWindowDrag", function (e) {
      if (pointerDrag) {
        pointerDrag = false;
        const pt = toImageCoords(e.clientX, e.clientY);
        if (opts.onPointerUp) opts.onPointerUp(pt ? pt.x : null, pt ? pt.y : null, e);
      }
      dragging = false;
    });
    $viewport.on("wheel", "img", function (e) {
      e.preventDefault();
      const oe = e.originalEvent;
      const delta = oe.deltaY < 0 ? zoomCtl.step : -zoomCtl.step;
      zoomCtl.adjustByStep(imgs(), delta, this, oe.clientX, oe.clientY);
      syncClip();
    });
    $viewport.on("click", "img", function (e) {
      if (dragMoved) {
        dragMoved = false;
        return;
      }
      if (opts.onImageClick) {
        const pt = toImageCoords(e.clientX, e.clientY);
        if (pt && opts.onImageClick(pt.x, pt.y)) return;
      }
      zoomCtl.toggleClick(imgs(), this, e.clientX, e.clientY);
      syncClip();
    });
    if (opts.onImageHover) {
      const setCursor = (result) => {
        $viewport.toggleClass("compare-window-pointer", result === true);
        $viewport.find("img").css("cursor", typeof result === "string" ? result : "");
      };
      $viewport.on("mousemove", "img", function (e) {
        if (dragging || pointerDrag) return;
        const pt = toImageCoords(e.clientX, e.clientY);
        setCursor(pt ? opts.onImageHover(pt.x, pt.y) : false);
      });
      $viewport.on("mouseleave", function () {
        if (pointerDrag) return;
        opts.onImageHover(null, null);
        setCursor(false);
      });
    }

    // Swallow page-level keyboard shortcuts (e.g. the review page's star
    // hotkeys) while the window is up; '1' still zooms to actual size.
    function onKeydown(e) {
      if ($(e.target).is($handle)) return;
      e.stopPropagation();
      if (e.key === "1") {
        e.preventDefault();
        zoomCtl.zoomToActual(imgs(), $beforeImg[0]);
        syncClip();
      }
    }
    document.addEventListener("keydown", onKeydown, true);

    function close() {
      document.removeEventListener("keydown", onKeydown, true);
      $(document).off("mousemove.compareWindowDrag mouseup.compareWindowDrag mousemove.compareSliderDrag mouseup.compareSliderDrag");
      $(window).off("resize.compareWindow");
      $("body").removeClass("image-window-open");
      $backdrop.remove();
    }

    function decide(handler) {
      $accept.prop("disabled", true);
      $reject.prop("disabled", true);
      $extras.forEach(($b) => $b.prop("disabled", true));
      Promise.resolve(handler ? handler() : null).then(close, close);
    }
    $accept.on("click", () => decide(opts.onAccept));
    $reject.on("click", () => decide(opts.onReject));

    // Swaps the after image in place, keeping zoom/pan and slider position.
    function setAfterUrl(url) {
      $afterImg.attr("src", url);
    }

    function setBeforeUrl(url) {
      $beforeImg.attr("src", url);
    }

    // true = before/after split with the slider; false = the after image
    // alone (no slider, no badges).
    function setSplit(on) {
      split = !!on;
      $viewport.toggleClass("compare-window-single", !split);
      $toolbar.find(".compare-window-hint").text(split ? HINT_SPLIT : HINT_SINGLE);
      syncClip();
    }

    // How much of the image (normalised 0-1, per axis) `px` screen pixels
    // span at the current zoom — for zoom-independent hit tolerances.
    function pxToImage(px) {
      const w = ($viewport.width() || 1) * content.w;
      const h = ($viewport.height() || 1) * content.h;
      return { x: px / (w * zoomCtl.zoom), y: px / (h * zoomCtl.zoom) };
    }

    // Changes the fixed viewport's reference size (a different photo size
    // after the edit panel's Reload); null goes back to following the picture.
    function setViewportSize(size) {
      viewportSize = size || null;
      lock();
    }

    return {
      close, setAfterUrl, setBeforeUrl, setSplit, pxToImage, setViewportSize,
      get zoom() { return zoomCtl.zoom; },
    };
  }

  return { computeContainFit, createZoomController, showImageWindow, showCompareWindow };
})();
