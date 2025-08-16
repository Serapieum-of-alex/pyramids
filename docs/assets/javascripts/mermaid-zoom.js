/*
Mermaid Zoom & Pan Enhancer
- Adds wheel-zoom (cursor-centered), drag-pan, and clickable controls to Mermaid SVG diagrams
- Works with MkDocs Material instant navigation by subscribing to `document$`
*/
(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const stateMap = new WeakMap(); // svg -> control API

  function ensureViewBox(svg) {
    if (!svg.hasAttribute("viewBox")) {
      const w = parseFloat(svg.getAttribute("width")) || svg.clientWidth || 1000;
      const h = parseFloat(svg.getAttribute("height")) || svg.clientHeight || 600;
      if (isFinite(w) && isFinite(h)) svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    }
  }

  function getViewBox(svg) {
    const vb = (svg.getAttribute("viewBox") || "").trim().split(/\s+/).map(parseFloat);
    if (vb.length === 4 && vb.every((n) => isFinite(n))) {
      return { x: vb[0], y: vb[1], width: vb[2], height: vb[3] };
    }
    // Fallback to client size
    return { x: 0, y: 0, width: svg.clientWidth || 1000, height: svg.clientHeight || 600 };
  }

  function wrapInGroup(svg) {
    // If already wrapped, return existing group
    const existing = svg.querySelector("g[data-zoom-layer]");
    if (existing) return existing;

    const g = document.createElementNS(SVG_NS, "g");
    g.setAttribute("data-zoom-layer", "true");

    // Move all children under the group
    const children = Array.from(svg.childNodes);
    for (const node of children) {
      g.appendChild(node);
    }
    svg.appendChild(g);
    return g;
  }

  function setupPanZoom(svg) {
    if (!svg || svg.dataset.zoomPanEnhanced === "true") return stateMap.get(svg);

    ensureViewBox(svg);
    const layer = wrapInGroup(svg);

    let scale = 1;
    let tx = 0;
    let ty = 0;

    function applyTransform() {
      layer.setAttribute("transform", `translate(${tx} ${ty}) scale(${scale})`);
    }

    function svgPointFromClient(svgEl, clientX, clientY) {
      const pt = svgEl.createSVGPoint();
      pt.x = clientX;
      pt.y = clientY;
      const ctm = svgEl.getScreenCTM();
      if (!ctm) return { x: 0, y: 0 };
      const inv = ctm.inverse();
      const p = pt.matrixTransform(inv);
      return { x: p.x, y: p.y };
    }

    function clamp(val, min, max) {
      return Math.max(min, Math.min(max, val));
    }

    function getCenterSvgPoint() {
      const rect = svg.getBoundingClientRect();
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      return svgPointFromClient(svg, cx, cy);
    }

    function zoomAt(point, factor) {
      const newScale = clamp(scale * factor, 0.2, 10);
      const k = newScale / scale;
      const x = point.x, y = point.y;
      tx = x - k * (x - tx);
      ty = y - k * (y - ty);
      scale = newScale;
      applyTransform();
    }

    function panBy(dx, dy) {
      tx += dx;
      ty += dy;
      applyTransform();
    }

    // Wheel zoom (cursor-centered)
    svg.addEventListener(
      "wheel",
      (ev) => {
        if (!ev.ctrlKey) ev.preventDefault(); // allow browser zoom if Ctrl pressed
        const { x, y } = svgPointFromClient(svg, ev.clientX, ev.clientY);
        const delta = -ev.deltaY; // up = zoom in
        const factor = Math.exp(delta * 0.0015); // smooth zoom
        zoomAt({ x, y }, factor);
      },
      { passive: false }
    );

    // Drag to pan (pointer events)
    let dragging = false;
    let last = { x: 0, y: 0 };

    function onPointerDown(ev) {
      dragging = true;
      svg.setPointerCapture(ev.pointerId);
      last = { x: ev.clientX, y: ev.clientY };
      svg.classList.add("pz-grabbing");
    }

    function onPointerMove(ev) {
      if (!dragging) return;
      const dx = ev.clientX - last.x;
      const dy = ev.clientY - last.y;
      last = { x: ev.clientX, y: ev.clientY };
      // Convert pixel delta to SVG units roughly using current scale and screen CTM
      const ctm = svg.getScreenCTM();
      if (ctm) {
        tx += dx / ctm.a; // ctm.a ~ pixels per SVG unit in X at current scale
        ty += dy / ctm.d; // ctm.d ~ pixels per SVG unit in Y at current scale
      } else {
        tx += dx / scale;
        ty += dy / scale;
      }
      applyTransform();
    }

    function onPointerUp(ev) {
      if (!dragging) return;
      dragging = false;
      try { svg.releasePointerCapture(ev.pointerId); } catch (e) {}
      svg.classList.remove("pz-grabbing");
    }

    svg.addEventListener("pointerdown", onPointerDown);
    svg.addEventListener("pointermove", onPointerMove);
    svg.addEventListener("pointerup", onPointerUp);
    svg.addEventListener("pointerleave", onPointerUp);

    // Double-click to reset
    function reset() {
      scale = 1;
      tx = 0;
      ty = 0;
      applyTransform();
    }
    svg.addEventListener("dblclick", reset);

    // Improve hit area
    svg.style.touchAction = "none"; // disable browser panning/zoom gestures on the element
    svg.style.cursor = "grab";

    const api = {
      zoomIn: () => {
        const center = getCenterSvgPoint();
        zoomAt(center, 1.2);
      },
      zoomOut: () => {
        const center = getCenterSvgPoint();
        zoomAt(center, 1 / 1.2);
      },
      panLeft: () => {
        const vb = getViewBox(svg);
        panBy(vb.width * 0.1, 0);
      },
      panRight: () => {
        const vb = getViewBox(svg);
        panBy(-vb.width * 0.1, 0);
      },
      panUp: () => {
        const vb = getViewBox(svg);
        panBy(0, vb.height * 0.1);
      },
      panDown: () => {
        const vb = getViewBox(svg);
        panBy(0, -vb.height * 0.1);
      },
      reset,
    };

    svg.dataset.zoomPanEnhanced = "true";
    applyTransform();
    stateMap.set(svg, api);
    return api;
  }

  function createControls(svg) {
    const api = setupPanZoom(svg);
    if (!api) return; // if not set up

    const container = svg.closest(".mermaid") || svg.parentElement;
    if (!container) return;

    // Avoid duplicates
    if (container.querySelector(":scope > .mz-controls")) return;

    const controls = document.createElement("div");
    controls.className = "mz-controls";

    function stopEvt(e) {
      e.preventDefault();
      e.stopPropagation();
    }

    function btn(label, title, onClick, className = "") {
      const b = document.createElement("button");
      b.className = `mz-btn ${className}`.trim();
      b.type = "button";
      b.setAttribute("aria-label", title);
      b.title = title;
      b.textContent = label;
      b.addEventListener("click", (e) => { stopEvt(e); onClick(); });
      b.addEventListener("pointerdown", stopEvt);
      b.addEventListener("wheel", stopEvt, { passive: false });
      return b;
    }

    // Build controls
    const row1 = document.createElement("div");
    row1.className = "mz-row";
    row1.appendChild(btn("+", "Zoom in", api.zoomIn, "mz-zoom-in"));
    row1.appendChild(btn("-", "Zoom out", api.zoomOut, "mz-zoom-out"));
    row1.appendChild(btn("⟳", "Reset view", api.reset, "mz-reset"));

    const row2 = document.createElement("div");
    row2.className = "mz-row mz-arrows";
    row2.appendChild(btn("▲", "Pan up", api.panUp, "mz-up"));
    const lr = document.createElement("div");
    lr.className = "mz-lr";
    lr.appendChild(btn("◀", "Pan left", api.panLeft, "mz-left"));
    lr.appendChild(btn("▶", "Pan right", api.panRight, "mz-right"));
    row2.appendChild(lr);
    row2.appendChild(btn("▼", "Pan down", api.panDown, "mz-down"));

    controls.appendChild(row1);
    controls.appendChild(row2);

    container.appendChild(controls);
  }

  function enhanceAll() {
    // Mermaid renders into <div class="mermaid"> -> <svg> ...
    const svgs = document.querySelectorAll(".mermaid svg");
    svgs.forEach((svg) => {
      setupPanZoom(svg);
      createControls(svg);
    });
  }

  // Run on DOM ready and after Material's instant navigation updates
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => {
      // Mermaid may render asynchronously; give it a tick
      setTimeout(enhanceAll, 0);
    });
  } else {
    document.addEventListener("DOMContentLoaded", () => setTimeout(enhanceAll, 0));
  }
})();
