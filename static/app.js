(() => {
  const $ = (id) => document.getElementById(id);
  const ALLOWED = /\.(jpe?g|png)$/i;
  const MESSAGES = {
    format: "Unsupported image format. Please upload a JPG, JPEG, or PNG image.",
    failed: "We couldn't analyze this image. Please try another image.",
  };

  const views = { upload: $("view-upload"), processing: $("view-processing"), results: $("view-results") };
  const el = {
    hero: $("hero"), dropzone: $("dropzone"), input: $("file-input"),
    dropEmpty: $("drop-empty"), dropSelected: $("drop-selected"),
    fileName: $("file-name"), fileMeta: $("file-meta"), error: $("upload-error"),
    analyze: $("analyze-btn"), another: $("another-btn"), rerun: $("rerun-btn"),
    download: $("download-btn"), report: $("report-btn"),
    confRange: $("conf-range"), confOut: $("conf-out"), mode: $("mode-select"),
    model: $("model-select"), status: $("status"), statusText: $("status-text"), elapsed: $("elapsed"),
  };

  let file = null;
  let lastResult = null;
  let maxUploadMb = 100;
  let timer = null;

  const fmt = new Intl.NumberFormat("en-US");
  const formatBytes = (n) => (n >= 1048576 ? `${(n / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} KB`);

  function show(name) {
    Object.entries(views).forEach(([key, node]) => (node.hidden = key !== name));
    el.hero.hidden = name !== "upload";
    el.rerun.hidden = name !== "results";
  }

  function showError(message) {
    el.error.textContent = message;
    el.error.hidden = !message;
  }

  /* ---------- model status ---------- */
  async function loadHealth() {
    try {
      const info = await (await fetch("/health")).json();
      if (!info.model_ready) throw new Error("not ready");
      el.status.dataset.state = "ready";
      el.statusText.textContent = "AI Model Ready";
      el.confRange.value = info.default_confidence;
      el.confOut.textContent = Number(info.default_confidence).toFixed(2);
      maxUploadMb = info.max_upload_mb;
      $("tech-device").textContent = info.device;
      el.model.replaceChildren(...info.models.map((name) => new Option(name.replace(/\.pt$/, ""), name)));
      el.model.value = info.default_model;
      el.model.disabled = false;
      $("tech-model").textContent = el.model.value;
    } catch {
      el.status.dataset.state = "error";
      el.statusText.textContent = "AI Model Unavailable";
    }
  }

  /* ---------- file selection ---------- */
  function selectFile(candidate) {
    showError("");
    if (!candidate) return;
    if (!ALLOWED.test(candidate.name) || (candidate.type && !/^image\/(jpeg|png)$/.test(candidate.type))) {
      return showError(MESSAGES.format);
    }
    if (candidate.size > maxUploadMb * 1048576) {
      return showError(`This image is larger than the ${maxUploadMb} MB limit.`);
    }
    file = candidate;
    el.fileName.textContent = file.name;
    el.fileMeta.textContent = formatBytes(file.size);
    el.dropEmpty.hidden = true;
    el.dropSelected.hidden = false;
    el.dropzone.classList.add("has-file");
    el.analyze.disabled = false;
  }

  function resetSelection() {
    file = null;
    lastResult = null;
    el.input.value = "";
    el.dropEmpty.hidden = false;
    el.dropSelected.hidden = true;
    el.dropzone.classList.remove("has-file");
    el.analyze.disabled = true;
    showError("");
  }

  el.dropzone.addEventListener("click", () => el.input.click());
  el.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); el.input.click(); }
  });
  el.input.addEventListener("change", () => selectFile(el.input.files[0]));
  ["dragenter", "dragover"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((type) =>
    el.dropzone.addEventListener(type, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); }));
  el.dropzone.addEventListener("drop", (e) => selectFile(e.dataTransfer.files[0]));
  // Dropping outside the zone should not make the browser navigate to the image.
  ["dragover", "drop"].forEach((type) => window.addEventListener(type, (e) => e.preventDefault()));

  /* ---------- inference ---------- */
  async function analyze() {
    if (!file) return;
    showError("");
    show("processing");
    const started = Date.now();
    el.elapsed.textContent = "· 0s";
    timer = setInterval(() => (el.elapsed.textContent = `· ${Math.round((Date.now() - started) / 1000)}s`), 1000);

    const body = new FormData();
    body.append("image", file);
    body.append("confidence", el.confRange.value);
    body.append("mode", el.mode.value);
    body.append("model", el.model.value);

    try {
      const response = await fetch("/predict", { method: "POST", body });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : MESSAGES.failed);
      lastResult = payload;
      renderResult(payload);
      show("results");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      show("upload");
      showError(err instanceof TypeError ? "We couldn't reach the server. Please check that the app is running." : err.message);
    } finally {
      clearInterval(timer);
    }
  }

  function describeRun(r) {
    if (!r.tile_window) return "single pass";
    return `${r.mode === "auto" ? "auto-scaled" : "tiled"} (${r.tiles} tiles, ${fmt.format(r.tile_window)} px window)`;
  }

  function renderResult(r) {
    const count = fmt.format(r.count);
    $("res-count").textContent = count;
    $("stat-count").textContent = count;
    $("res-filename").textContent = r.filename;
    $("stat-conf").textContent = r.average_confidence == null ? "—" : `${(r.average_confidence * 100).toFixed(1)}%`;
    $("stat-size").textContent = `${fmt.format(r.image_width)} × ${fmt.format(r.image_height)}`;
    $("note-original").textContent = r.filename;
    $("note-annotated").textContent = `${count} ${r.count === 1 ? "palm" : "palms"}`;

    const bust = `?t=${Date.now()}`;
    $("img-original").src = r.original_url + bust;
    $("img-annotated").src = r.annotated_url + bust;

    const stem = r.filename.replace(/\.[^.]+$/, "");
    el.download.href = r.annotated_url;
    el.download.setAttribute("download", `${stem}_date-palms.jpg`);

    $("tech-model").textContent = r.model;
    $("tech-run").textContent =
      `${(r.inference_ms / 1000).toFixed(1)}s · ${describeRun(r)} · conf ${r.confidence_threshold.toFixed(2)}`;
    document.querySelector(".tech-run").hidden = false;
  }

  /* ---------- report ---------- */
  function downloadReport() {
    if (!lastResult) return;
    const r = lastResult;
    const lines = [
      "DATE PALM AI — DETECTION REPORT",
      "================================",
      `Image:                  ${r.filename}`,
      `Analyzed:               ${new Date().toLocaleString()}`,
      `Date palms detected:    ${r.count}`,
      `Average confidence:     ${r.average_confidence == null ? "n/a" : (r.average_confidence * 100).toFixed(1) + "%"}`,
      `Image dimensions:       ${r.image_width} x ${r.image_height} px`,
      `Confidence threshold:   ${r.confidence_threshold.toFixed(2)}`,
      `Processing:             ${describeRun(r)}`,
      `Model:                  ${r.model}`,
      `Device:                 ${r.device}`,
      "",
      "Counts are model estimates and may miss or mis-detect individual trees.",
    ];
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/plain" }));
    link.download = `${r.filename.replace(/\.[^.]+$/, "")}_report.txt`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  /* ---------- lightbox ---------- */
  const box = $("lightbox");
  const boxScroll = $("lightbox-scroll");
  const boxImg = $("lightbox-img");
  document.querySelectorAll("[data-zoom]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const src = btn.querySelector("img").src;
      boxImg.src = src;
      $("lightbox-title").textContent = btn.dataset.zoom === "original" ? "Original" : "AI Detection";
      boxScroll.classList.remove("actual");
      $("lightbox-fit").textContent = "100%";
      box.hidden = false;
      document.body.style.overflow = "hidden";
    }));
  const closeBox = () => { box.hidden = true; document.body.style.overflow = ""; boxImg.removeAttribute("src"); };
  $("lightbox-close").addEventListener("click", closeBox);
  $("lightbox-fit").addEventListener("click", () => {
    const actual = boxScroll.classList.toggle("actual");
    $("lightbox-fit").textContent = actual ? "Fit" : "100%";
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !box.hidden) closeBox(); });

  /* ---------- wiring ---------- */
  el.analyze.addEventListener("click", analyze);
  el.rerun.addEventListener("click", analyze);
  el.report.addEventListener("click", downloadReport);
  el.model.addEventListener("change", () => {
    $("tech-model").textContent = el.model.value;
    if (file && !views.results.hidden) analyze(); // switching models on a result re-runs the same image
  });
  el.another.addEventListener("click", () => {
    resetSelection();
    show("upload");
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
  el.confRange.addEventListener("input", () => (el.confOut.textContent = Number(el.confRange.value).toFixed(2)));

  show("upload");
  loadHealth();
})();
