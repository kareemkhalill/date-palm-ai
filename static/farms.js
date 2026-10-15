(() => {
  const $ = (id) => document.getElementById(id);
  const fmt = new Intl.NumberFormat("en-US");
  const json = (body) => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const api = async (url, options) => {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : "Something went wrong. Please try again.");
    return payload;
  };
  const AREA_REGION = "Custom area";

  const state = {
    layers: new Map(),          // code -> Leaflet layer (farms and custom areas)
    props: new Map(),           // code -> feature properties
    regionGroups: new Map(),    // region -> L.featureGroup
    selected: new Set(),
    focus: null,                // code shown in the result panel
    captures: [],
    overlays: new Map(),        // capture name -> L.imageOverlay
    lastResult: new Map(),      // code -> full analysis result
    lightbox: { code: null, raw: false },
    job: null,
    pending: null,              // codes waiting for the user to confirm a Google Earth capture
    drawing: false,
    draft: null,                // polygon drawn but not saved yet
  };

  /* ------------------------------------------------------------------ map */
  const map = L.map("map", { preferCanvas: true, zoomSnap: 0.25, maxZoom: 22, minZoom: 5, zoomControl: true, attributionControl: false });
  // Orientation-only base maps (stored/live Google Earth captures are drawn on top of them).
  const street = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 19, attribution: "© OpenStreetMap contributors" });
  const satellite = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19, attribution: "Imagery © Esri" });
  street.addTo(map);
  L.control.layers({ "Street map": street, "Satellite (Esri)": satellite, "No base map": L.layerGroup() }, null, { position: "topright" }).addTo(map);
  L.control.attribution({ prefix: false, position: "bottomleft" }).addTo(map);
  map.createPane("imageryPane").style.zIndex = 250;   // Earth captures: above the base map, below polygons
  map.createPane("regionsPane").style.zIndex = 350;
  const regionLayer = L.layerGroup().addTo(map);
  const dots = L.layerGroup().addTo(map);
  const areaGroup = L.featureGroup().addTo(map);
  const draftGroup = L.featureGroup().addTo(map);
  const dotRenderer = L.canvas({ padding: 0.3 });

  function farmStyle(code) {
    const p = state.props.get(code);
    const isArea = p.region === AREA_REGION;
    if (state.selected.has(code)) return { color: "#ffd21f", weight: 3, fillColor: "#ffd21f", fillOpacity: 0.18, dashArray: null };
    if (p.palms != null) {
      const density = p.area_ha > 0.01 ? p.palms / p.area_ha : 0;
      const t = Math.min(1, density / 160);
      return { color: isArea ? "#8b3fd1" : "#1d7a45", weight: 2, fillColor: `hsl(140 ${40 + t * 25}% ${62 - t * 28}%)`, fillOpacity: 0.5, dashArray: null };
    }
    if (isArea) return { color: "#8b3fd1", weight: 2.5, fillColor: "#a855f7", fillOpacity: 0.12, dashArray: null };
    if (p.coverage >= 99.5) return { color: "#19a7b8", weight: 1.6, fillColor: "#19a7b8", fillOpacity: 0.1, dashArray: null };
    if (p.coverage > 0) return { color: "#e0a526", weight: 1.4, fillColor: "#e0a526", fillOpacity: 0.08, dashArray: "4 3" };
    return { color: "#e8ebe7", weight: 1, fillColor: "#8c968f", fillOpacity: 0.12, dashArray: "3 3" };
  }
  const restyle = (code) => state.layers.get(code)?.setStyle(farmStyle(code));

  function tooltipHtml(p) {
    const count = p.palms != null ? `<br><b>${fmt.format(p.palms)} palms</b>` : "";
    return `${p.code}${p.name ? " · " + p.name : ""}<br><span>${p.status || p.type || ""} · ${p.area_ha} ha</span>${count}`;
  }

  function registerLayer(feature, group) {
    const p = feature.properties;
    state.props.set(p.code, p);
    const layer = L.geoJSON(feature, { style: () => farmStyle(p.code) }).getLayers()[0];
    layer.on("click", (e) => { if (state.drawing) return; L.DomEvent.stopPropagation(e); toggle(p.code); });
    layer.bindTooltip(() => tooltipHtml(state.props.get(p.code)), { sticky: true, className: "farm-tip", direction: "top" });
    state.layers.set(p.code, layer);
    group.addLayer(layer);
    if (p.palms != null) setCountLabel(p.code);
    return layer;
  }

  function addFarms(collection) {
    const byRegion = new Map();
    for (const feature of collection.features) {
      const region = feature.properties.region;
      if (!byRegion.has(region)) byRegion.set(region, L.featureGroup());
      registerLayer(feature, byRegion.get(region));
    }
    for (const [region, group] of byRegion) { state.regionGroups.set(region, group); group.addTo(map); }
  }

  function addArea(feature) {
    const p = feature.properties;
    Object.assign(p, { region: AREA_REGION, status: p.name, type: "Custom area", coverage: 100 });
    return registerLayer(feature, areaGroup);
  }

  function setCountLabel(code) {
    const layer = state.layers.get(code);
    const p = state.props.get(code);
    if (!layer._countLabel) layer._countLabel = L.tooltip({ permanent: true, direction: "center", className: "count-label" });
    layer._countLabel.setContent(fmt.format(p.palms));
    layer._countLabel.setLatLng(layer.getBounds().getCenter());
    if (map.getZoom() >= 16) layer._countLabel.addTo(map); else layer._countLabel.remove();
  }
  const refreshLabels = () => {
    for (const layer of state.layers.values()) if (layer._countLabel) (map.getZoom() >= 16 ? layer._countLabel.addTo(map) : layer._countLabel.remove());
  };

  /* ------------------------------------------------- capture previews (lazy) */
  function syncOverlays() {
    const bounds = map.getBounds();
    const wanted = map.getZoom() >= 12
      ? state.captures.filter((c) => c.available && bounds.intersects(L.latLngBounds([c.bounds[1], c.bounds[0]], [c.bounds[3], c.bounds[2]]))).slice(0, 36)
      : [];
    const names = new Set(wanted.map((c) => c.name));
    for (const [name, overlay] of state.overlays) if (!names.has(name)) { overlay.remove(); state.overlays.delete(name); }
    for (const c of wanted) {
      if (state.overlays.has(c.name)) continue;
      const overlay = L.imageOverlay(`/api/farms/preview/${c.name}.jpg?v=${encodeURIComponent(c.captured_at || "")}`,
        [[c.bounds[1], c.bounds[0]], [c.bounds[3], c.bounds[2]]], { interactive: false, pane: "imageryPane" });
      overlay.addTo(map);
      state.overlays.set(c.name, overlay);
    }
  }
  async function reloadCaptures() {
    state.captures = await api("/api/farms/captures");
    for (const overlay of state.overlays.values()) overlay.remove();
    state.overlays.clear();
    syncOverlays();
  }

  /* ------------------------------------------------------------ selection */
  function toggle(code) {
    if (state.drawing) return;
    state.selected.has(code) ? state.selected.delete(code) : state.selected.add(code);
    restyle(code);
    renderSelection();
    showFarm(code);
  }
  function setSelection(codes) {
    const old = [...state.selected];
    state.selected = new Set(codes);
    old.forEach(restyle);
    codes.forEach(restyle);
    renderSelection();
  }
  function renderSelection() {
    const n = state.selected.size;
    $("sel-count").textContent = fmt.format(n);
    $("clear-btn").hidden = n === 0;
    $("sel-hint").hidden = n > 0;
    $("count-btn").disabled = n === 0 || !!state.job;
    $("count-btn").textContent = n > 1 ? `Count palms in ${fmt.format(n)} polygons` : "Count palms in selection";
    const chips = $("chips");
    chips.replaceChildren(...[...state.selected].slice(0, 60).map((code) => {
      const li = document.createElement("li");
      li.textContent = code;
      const x = document.createElement("button");
      x.textContent = "×"; x.setAttribute("aria-label", `Deselect ${code}`);
      x.addEventListener("click", (e) => { e.stopPropagation(); toggle(code); });
      li.append(x);
      li.addEventListener("click", () => { showFarm(code); zoomTo(code); });
      return li;
    }));
    if (n > 60) { const li = document.createElement("li"); li.textContent = `+${n - 60} more`; chips.append(li); }
    refreshSavedPicker();
  }
  function zoomTo(code) {
    const layer = state.layers.get(code);
    if (layer) map.fitBounds(layer.getBounds(), { maxZoom: 19, padding: [60, 60] });
  }

  /* ------------------------------------------------------------- result panel */
  function resultHtml(r) {
    const counted = r.count != null && r.analyzed !== false;
    const p = state.props.get(r.code) || {};
    const isArea = (r.kind || (p.region === AREA_REGION ? "area" : "farm")) === "area";
    const coverage = r.coverage_pct ?? p.coverage;
    const rows = [[isArea ? "Type" : "Region", isArea ? "Custom area" : (r.region || p.region)],
      [isArea ? "Name" : "Status", isArea ? (r.name || p.name || "") : (r.status || p.status || "—")],
      ["Area", `${r.area_ha ?? p.area_ha} ha`]];
    if (r.saved_captures != null) rows.push(["Saved captures", r.saved_captures]);
    if (counted) rows.push(["Palms / ha", r.palms_per_ha ?? "—"], ["Avg confidence", r.mean_confidence != null ? `${(r.mean_confidence * 100).toFixed(1)}%` : "—"],
      ["Imagery", r.source === "saved" ? "Saved capture" : "New Google Earth capture (saved)"],
      ["Captured", r.captured_at ? r.captured_at.replace("T", " ") : "—"],
      ["Model", String(r.model || "").replace(/\.pt$/, "")]);
    const meta = rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
    const strip = counted && r.imagery_status_images?.[0]
      ? `<p class="muted" style="margin:8px 0 0">Earth's status bar when captured (shows the imagery date):</p><img src="${r.imagery_status_images[0]}" style="max-width:100%;border-radius:6px;margin-top:4px" alt="">` : "";
    const partial = counted && coverage < 99.5
      ? `<p class="warn">The capture covers only ${coverage}% of this polygon, so the count is partial.</p>` : "";
    const quality = counted && r.warning ? `<p class="warn">${r.warning}</p>` : "";
    const title = r.name ? `${r.code} · ${r.name}` : r.code;
    return `
      <p class="result-code">${title}</p>
      ${counted ? `<p class="result-count">${fmt.format(r.count)}</p><p class="muted">date palms inside this ${isArea ? "area" : "farm"}</p>` : `<p class="muted">Not counted yet — select it and press “Count palms”.</p>`}
      <dl class="result-meta">${meta}</dl>
      ${counted ? `<img class="result-img" id="result-img" src="${r.image}?t=${Date.now()}" alt="${r.code} with detected palms">${strip}` : ""}
      <div class="result-actions">
        ${counted ? `<button class="btn btn-secondary" id="open-img">Open full size</button>` : ""}
        <button class="btn btn-secondary" id="zoom-farm">Zoom to ${isArea ? "area" : "farm"}</button>
        <button class="btn btn-secondary" id="earth-farm">Open in Google Earth</button>
        ${isArea ? `<button class="btn btn-secondary" id="delete-area">Delete area</button>` : ""}
      </div>${quality}${partial}`;
  }

  async function showFarm(code) {
    state.focus = code;
    $("result-panel").hidden = false;
    let r = state.lastResult.get(code);
    if (!r) {
      try { r = await api(`/api/farms/${encodeURIComponent(code)}`); if (r.analyzed) state.lastResult.set(code, r); }
      catch (err) { $("result-body").innerHTML = `<p class="warn">${err.message}</p>`; return; }
    }
    if (state.focus !== code) return;
    $("result-body").innerHTML = resultHtml(r);
    $("zoom-farm")?.addEventListener("click", () => zoomTo(code));
    $("earth-farm")?.addEventListener("click", () => earthCall(`/earth/fly/${encodeURIComponent(code)}`, `Flying Google Earth to ${code}…`));
    $("open-img")?.addEventListener("click", () => openLightbox(code));
    $("result-img")?.addEventListener("click", () => openLightbox(code));
    $("delete-area")?.addEventListener("click", () => deleteArea(code));
    drawDots();
  }

  function drawDots() {
    dots.clearLayers();
    if (map.getZoom() < 16) return;
    for (const code of state.selected) {
      const r = state.lastResult.get(code);
      if (!r?.points?.length) continue;
      for (const [lon, lat] of r.points.slice(0, 20000)) {
        L.circleMarker([lat, lon], { renderer: dotRenderer, radius: 3, color: "#fff", weight: 1, fillColor: "#eb2828", fillOpacity: 1, interactive: false }).addTo(dots);
      }
    }
  }

  /* ---------------------------------------------------------------- counting */
  const settings = () => ({ model: $("model-select").value, confidence: Number($("conf-range").value), source: $("source-select").value,
    refresh: true, imagery_date: $("date-mode").value, capture_id: $("source-select").value === "saved" && state.selected.size === 1 ? $("capture-select").value : "" });

  /* saved captures: pick which one to count when a single polygon is selected */
  async function refreshSavedPicker() {
    const picker = $("saved-picker");
    const saved = $("source-select").value === "saved";
    picker.hidden = !saved;
    $("refresh-wrap").hidden = saved;
    if (!saved) return;
    const note = $("saved-note"), select = $("capture-select");
    select.hidden = true; $("delete-capture").hidden = true;
    if (state.selected.size !== 1) {
      note.textContent = state.selected.size ? "Each selected polygon uses its newest saved capture; polygons without one are skipped." : "Select a farm or area to see its saved captures.";
      return;
    }
    const code = [...state.selected][0];
    try {
      const list = await api(`/api/farms/${encodeURIComponent(code)}/captures`);
      if (!list.length) { note.textContent = `${code} has no saved capture yet. Choose “Capture new from Google Earth” once and it will be saved.`; return; }
      select.replaceChildren(...list.map((m, i) => new Option(`${m.captured_at.replace("T", " ")}${i === 0 ? " (newest)" : ""} · ${m.size_mb} MB`, m.id)));
      select.hidden = false; $("delete-capture").hidden = false;
      note.textContent = `${list.length} saved capture${list.length > 1 ? "s" : ""} for ${code}.`;
    } catch (err) { note.textContent = err.message; }
  }
  $("delete-capture").addEventListener("click", async () => {
    const code = [...state.selected][0], id = $("capture-select").value;
    if (!code || !id || !confirm(`Delete the saved capture ${id} of ${code}?`)) return;
    await api(`/api/farms/${encodeURIComponent(code)}/captures/${id}`, { method: "DELETE" });
    await reloadCaptures(); refreshSavedPicker();
  });
  const countError = (message) => { $("count-error").textContent = message; $("count-error").hidden = false; };
  function setBusy(busy, text) {
    $("progress").hidden = !busy; $("progress-text").hidden = !busy; $("cancel-job").hidden = !busy || !state.job;
    if (busy) { $("progress-bar").style.width = "4%"; $("progress-text").textContent = text || ""; $("progress-text").classList.remove("phase-capture"); }
    $("count-error").hidden = true;
    renderSelection();
  }

  function applyResult(r) {
    const p = state.props.get(r.code);
    if (!p) return;
    p.palms = r.count;
    state.lastResult.set(r.code, r);
    restyle(r.code);
    setCountLabel(r.code);
  }

  async function countSelected() {
    const codes = [...state.selected];
    const s = settings();
    $("count-error").hidden = true; $("confirm").hidden = true;
    try {
      const est = await api("/api/farms/estimate", json({ codes, source: s.source, refresh: s.refresh }));
      if (est.too_many) return countError(`This needs ${est.captures_needed} Google Earth captures (the limit is ${est.limit}). Select fewer farms or draw a smaller area.`);
      if (s.source === "saved" && est.farms_without_imagery?.length === codes.length) {
        return countError(codes.length === 1
          ? `${codes[0]} has no saved capture yet. Choose “Capture new from Google Earth” once; the capture is saved for later.`
          : "None of the selected polygons has a saved capture yet. Choose “Capture new from Google Earth” first.");
      }
      if (s.source === "live" && est.captures_needed > 0) {
        const n = est.captures_needed;
        $("confirm-text").textContent = `This needs ${n} capture${n > 1 ? "s" : ""} from Google Earth (about ${est.minutes} min). Google Earth must stay open and linked, and please don't use the mouse or keyboard while it runs.`;
        state.pending = codes;
        $("confirm").hidden = false;
        return;
      }
    } catch (err) { return countError(err.message); }
    await runCount(codes);
  }

  async function runCount(codes) {
    $("confirm").hidden = true;
    setBusy(true, codes.length === 1 ? `Starting ${codes[0]}…` : `Starting ${codes.length} polygons…`);
    try {
      const start = await api("/api/farms/analyze", json({ codes, ...settings() }));
      if (start.results) {
        applyResult(start.results[0]);
        await showFarm(codes[0]);
      } else {
        state.job = start.job;
        $("cancel-job").hidden = false;
        await pollJob(start.job, codes);
      }
      await refreshTotals();
      await reloadCaptures();
      refreshSavedPicker();
    } catch (err) {
      countError(err.message);
    } finally {
      state.job = null;
      setBusy(false);
    }
  }

  async function pollJob(jobId, codes) {
    for (;;) {
      const job = await api(`/api/farms/jobs/${jobId}`);
      $("progress-bar").style.width = `${Math.max(4, ((job.done + (job.phase === "capture" ? 0.3 : 0.6)) / job.total) * 100)}%`;
      $("progress-text").textContent = `${job.message} · ${fmt.format(Math.min(job.done + 1, job.total))} of ${fmt.format(job.total)}`;
      $("progress-text").classList.toggle("phase-capture", job.phase === "capture");
      for (const r of job.results) {
        const p = state.props.get(r.code);
        state.lastResult.delete(r.code);
        if (p) { p.palms = r.count; restyle(r.code); setCountLabel(r.code); }
      }
      if (job.status === "error") throw new Error(job.error);
      if (job.status === "done" || job.status === "cancelled") {
        const total = job.results.reduce((s, r) => s + r.count, 0);
        $("progress-text").hidden = false;
        $("progress-text").classList.remove("phase-capture");
        $("progress-text").textContent = `${job.status === "cancelled" ? "Cancelled. " : ""}Counted ${fmt.format(job.results.length)} polygon${job.results.length === 1 ? "" : "s"}: ${fmt.format(total)} palms.`;
        if (job.skipped.length) countError(job.skipped.slice(0, 2).map((s) => `${s.code}: ${s.reason}`).join("  ") + (job.skipped.length > 2 ? `  (+${job.skipped.length - 2} more)` : ""));
        if (state.focus) showFarm(state.focus);
        return;
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
  }

  async function refreshTotals() {
    const meta = await api("/api/farms/meta");
    $("total-palms").textContent = fmt.format(meta.total_palms);
    $("total-line").textContent = `palms counted in ${fmt.format(meta.analyzed)} of ${fmt.format(meta.farms)} farms`;
    renderRegions(meta);
  }

  /* ----------------------------------------------------------------- regions */
  function renderRegions(meta) {
    $("meta-line").textContent = `${fmt.format(meta.farms)} farms · ${meta.live_captures + meta.captures_ready} captures`;
    const list = $("regions");
    list.replaceChildren(...meta.regions.map((r) => {
      const li = document.createElement("li");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.checked = state.regionGroups.get(r.name) ? map.hasLayer(state.regionGroups.get(r.name)) : true;
      cb.setAttribute("aria-label", `Show ${r.name}`);
      cb.addEventListener("change", () => { const g = state.regionGroups.get(r.name); if (!g) return; cb.checked ? g.addTo(map) : g.remove(); });
      const name = document.createElement("span");
      name.className = "rname"; name.textContent = r.name; name.title = "Zoom to region";
      name.addEventListener("click", () => { const g = state.regionGroups.get(r.name); if (g) map.fitBounds(g.getBounds(), { padding: [30, 30] }); });
      const stat = document.createElement("span");
      stat.className = "rstat";
      stat.innerHTML = `${fmt.format(r.farms)} farms${r.analyzed ? ` · <b>${fmt.format(r.palms)}</b> palms` : ""}`;
      const btns = document.createElement("span");
      btns.className = "rbtn";
      const selAll = document.createElement("button");
      selAll.className = "link"; selAll.textContent = "Select all farms";
      selAll.addEventListener("click", () => {
        setSelection([...state.props.values()].filter((p) => p.region === r.name).map((p) => p.code));
      });
      btns.append(selAll);
      li.append(cb, name, stat, btns);
      return li;
    }));
  }

  /* ------------------------------------------------------------ custom areas */
  let drawer = null;
  function stopDrawing() { state.drawing = false; drawer?.disable(); drawer = null; $("draw-btn").textContent = "Draw an area"; }

  $("draw-btn").addEventListener("click", () => {
    if (drawer) return stopDrawing();
    $("area-error").hidden = true;
    draftGroup.clearLayers();
    $("area-form").hidden = true;
    state.drawing = true;
    drawer = new L.Draw.Polygon(map, { allowIntersection: false, showArea: true, shapeOptions: { color: "#8b3fd1", weight: 3, fillOpacity: 0.15 } });
    drawer.enable();
    $("draw-btn").textContent = "Cancel drawing";
    $("draw-hint").textContent = "Click the corners of the area on the map; click the first point (or double-click) to finish.";
  });
  map.on(L.Draw.Event.CREATED, (e) => {
    stopDrawing();
    state.draft = e.layer;
    draftGroup.addLayer(e.layer);
    const km2 = L.GeometryUtil.geodesicArea(e.layer.getLatLngs()[0]) / 1e6;
    $("area-size").textContent = `${(km2 * 100).toFixed(2)} ha (${km2.toFixed(3)} km²)`;
    $("area-name").value = "";
    $("area-form").hidden = false;
    $("draw-hint").textContent = "Name it (optional), then save it to count the palms inside.";
  });
  map.on(L.Draw.Event.DRAWSTOP, () => { state.drawing = false; });
  $("area-discard").addEventListener("click", () => { draftGroup.clearLayers(); state.draft = null; $("area-form").hidden = true; });
  $("area-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!state.draft) return;
    try {
      const saved = await api("/api/farms/areas", json({ name: $("area-name").value, geometry: state.draft.toGeoJSON().geometry }));
      draftGroup.clearLayers(); state.draft = null; $("area-form").hidden = true;
      addArea({ type: "Feature", geometry: saved.geometry, properties: { code: saved.code, name: saved.name, area_ha: saved.area_ha, palms: null } });
      setSelection([saved.code]); showFarm(saved.code);
      $("draw-hint").textContent = "Area saved and selected. Press “Count palms” to capture it from Google Earth and count.";
    } catch (err) { $("area-error").textContent = err.message; $("area-error").hidden = false; }
  });
  async function deleteArea(code) {
    if (!confirm(`Delete ${code}?`)) return;
    await api(`/api/farms/areas/${encodeURIComponent(code)}`, { method: "DELETE" });
    state.layers.get(code)?._countLabel?.remove();
    areaGroup.removeLayer(state.layers.get(code));
    state.layers.delete(code); state.props.delete(code); state.lastResult.delete(code); state.selected.delete(code);
    $("result-panel").hidden = true;
    renderSelection(); refreshTotals();
  }

  /* ------------------------------------------------------------- Google Earth */
  async function earthCall(url, message) {
    const out = $("earth-msg");
    out.hidden = false;
    out.textContent = message;
    try {
      await api(url, { method: "POST" });
      out.textContent = `${message} Done.`;
    } catch (err) { out.textContent = err.message; }
    pollEarth();
  }
  async function pollEarth() {
    try {
      const s = await api("/earth/status");
      $("earth-status").dataset.linked = String(s.linked);
      $("earth-status-text").textContent = s.linked ? "Google Earth is linked — ready to capture" : "Google Earth is not linked";
    } catch { $("earth-status-text").textContent = "Unknown"; }
  }
  $("earth-link-btn").addEventListener("click", () => earthCall("/earth/open-link", "Linking Google Earth (it may take ~10 s; Earth's prompts are answered for you)…"));
  setInterval(pollEarth, 5000);

  /* ---------------------------------------------------------------- lightbox */
  const box = $("lightbox"), scroll = $("lightbox-scroll"), boxImg = $("lightbox-img");
  function openLightbox(code, raw = false) {
    state.lightbox = { code, raw };
    const r = state.lastResult.get(code);
    boxImg.src = `${raw ? r.image_raw : r.image}?t=${Date.now()}`;
    $("lightbox-title").textContent = `${code} · ${fmt.format(r.count)} palms`;
    $("lightbox-toggle").textContent = raw ? "Show palm dots" : "Show clean image";
    scroll.classList.remove("actual"); $("lightbox-fit").textContent = "100%";
    box.hidden = false; document.body.style.overflow = "hidden";
  }
  const closeBox = () => { box.hidden = true; document.body.style.overflow = ""; boxImg.removeAttribute("src"); };
  $("lightbox-close").addEventListener("click", closeBox);
  $("lightbox-toggle").addEventListener("click", () => openLightbox(state.lightbox.code, !state.lightbox.raw));
  $("lightbox-fit").addEventListener("click", () => { const a = scroll.classList.toggle("actual"); $("lightbox-fit").textContent = a ? "Fit" : "100%"; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !box.hidden) closeBox(); });

  /* ------------------------------------------------------------------- wiring */
  $("clear-btn").addEventListener("click", () => { setSelection([]); dots.clearLayers(); });
  $("count-btn").addEventListener("click", countSelected);
  $("confirm-go").addEventListener("click", () => runCount(state.pending || [...state.selected]));
  $("confirm-no").addEventListener("click", () => { $("confirm").hidden = true; });
  $("cancel-job").addEventListener("click", async () => {
    if (state.job) { $("cancel-job").hidden = true; $("progress-text").textContent = "Cancelling after the current step…"; try { await api(`/api/farms/jobs/${state.job}/cancel`, { method: "POST" }); } catch { /* finished already */ } }
  });
  $("source-select").addEventListener("change", refreshSavedPicker);
  $("conf-range").addEventListener("input", () => ($("conf-out").textContent = Number($("conf-range").value).toFixed(2)));
  $("search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const code = $("search").value.trim().toUpperCase();
    const err = $("search-error");
    err.hidden = true;
    if (!code) return;
    if (!state.layers.has(code)) { err.textContent = `No farm or area with code ${code}.`; err.hidden = false; return; }
    setSelection([code]); zoomTo(code); showFarm(code);
  });
  map.on("moveend", syncOverlays);
  map.on("zoomend", () => { refreshLabels(); drawDots(); });

  /* ---------------------------------------------------------------------- init */
  async function init() {
    try {
      const health = await api("/health");
      $("status").dataset.state = health.model_ready ? "ready" : "loading";
      $("status-text").textContent = health.model_ready ? "AI Model Ready" : "Model loading…";
      $("model-select").replaceChildren(...health.models.map((m) => new Option(m.replace(/\.pt$/, ""), m)));
      $("model-select").value = health.default_model;
      $("conf-range").value = health.default_confidence;
      $("conf-out").textContent = Number(health.default_confidence).toFixed(2);
    } catch { $("status").dataset.state = "error"; $("status-text").textContent = "AI Model Unavailable"; }

    try {
      const [meta, regions, farms, captures, areas] = await Promise.all([
        api("/api/farms/meta"), api("/api/farms/regions"), api("/api/farms/geojson"), api("/api/farms/captures"), api("/api/farms/areas")]);
      state.captures = captures;
      L.geoJSON(regions, { pane: "regionsPane", interactive: false, style: { color: "#c8d0ca", weight: 1.5, fill: false, dashArray: "6 4" } }).addTo(regionLayer);
      addFarms(farms);
      areas.features.forEach(addArea);
      $("total-palms").textContent = fmt.format(meta.total_palms);
      $("total-line").textContent = `palms counted in ${fmt.format(meta.analyzed)} of ${fmt.format(meta.farms)} farms`;
      renderRegions(meta);
      const start = meta.regions.slice().sort((a, b) => b.analyzed - a.analyzed || (a.name === "Tayma" ? -1 : 1))[0];
      const group = start && state.regionGroups.get(start.name);
      map.fitBounds((group || L.featureGroup([...state.regionGroups.values()])).getBounds(), { padding: [20, 20] });
      syncOverlays();
    } catch (err) {
      $("meta-line").textContent = err.message;
      map.setView([26.6, 38.0], 8);
    }
    renderSelection();
    pollEarth();
    const wanted = new URLSearchParams(location.search).get("code");   // e.g. linked from a Google Earth balloon
    if (wanted && state.layers.has(wanted.toUpperCase())) {
      const code = wanted.toUpperCase();
      setSelection([code]); zoomTo(code); showFarm(code);
    }
  }
  init();
})();
