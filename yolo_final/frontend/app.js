const state = {
  apiBase: "",
  imageFile: null,
  imageId: "",
  imageWidth: 0,
  imageHeight: 0,
  boxes: [],
  selectedId: null,
  nextBoxId: 1,
  interaction: null,
};

const els = {
  imageInput: document.getElementById("imageInput"),
  predictButton: document.getElementById("predictButton"),
  addBoxButton: document.getElementById("addBoxButton"),
  fitButton: document.getElementById("fitButton"),
  canvasShell: document.getElementById("canvasShell"),
  emptyState: document.getElementById("emptyState"),
  imageViewport: document.getElementById("imageViewport"),
  previewImage: document.getElementById("previewImage"),
  boxLayer: document.getElementById("boxLayer"),
  apiBaseInput: document.getElementById("apiBaseInput"),
  healthButton: document.getElementById("healthButton"),
  apiStatus: document.getElementById("apiStatus"),
  apiStatusText: document.getElementById("apiStatusText"),
  scoreInput: document.getElementById("scoreInput"),
  topKInput: document.getElementById("topKInput"),
  nmsInput: document.getElementById("nmsInput"),
  imageName: document.getElementById("imageName"),
  imageSize: document.getElementById("imageSize"),
  boxCount: document.getElementById("boxCount"),
  boxList: document.getElementById("boxList"),
  deleteButton: document.getElementById("deleteButton"),
  editorEmpty: document.getElementById("editorEmpty"),
  boxEditor: document.getElementById("boxEditor"),
  classIdInput: document.getElementById("classIdInput"),
  scoreReadout: document.getElementById("scoreReadout"),
  x1Input: document.getElementById("x1Input"),
  y1Input: document.getElementById("y1Input"),
  x2Input: document.getElementById("x2Input"),
  y2Input: document.getElementById("y2Input"),
  saveButton: document.getElementById("saveButton"),
  downloadLink: document.getElementById("downloadLink"),
  saveMessage: document.getElementById("saveMessage"),
};

function endpoint(path) {
  const base = state.apiBase.replace(/\/$/, "");
  return `${base}${path}`;
}

function setStatus(type, text) {
  els.apiStatus.className = `status-dot ${type}`;
  els.apiStatusText.textContent = text;
}

function setMessage(text, type = "") {
  els.saveMessage.textContent = text;
  els.saveMessage.className = `save-message ${type}`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(value, max));
}

function selectedBox() {
  return state.boxes.find((box) => box.id === state.selectedId) || null;
}

function scaleInfo() {
  const rect = els.previewImage.getBoundingClientRect();
  return {
    width: rect.width,
    height: rect.height,
    scaleX: rect.width / Math.max(state.imageWidth, 1),
    scaleY: rect.height / Math.max(state.imageHeight, 1),
  };
}

function clientToImagePoint(clientX, clientY) {
  const rect = els.previewImage.getBoundingClientRect();
  return {
    x: clamp((clientX - rect.left) * state.imageWidth / Math.max(rect.width, 1), 0, state.imageWidth),
    y: clamp((clientY - rect.top) * state.imageHeight / Math.max(rect.height, 1), 0, state.imageHeight),
  };
}

function normalizeBox(box) {
  const x1 = clamp(Math.min(Number(box.x1), Number(box.x2)), 0, state.imageWidth);
  const y1 = clamp(Math.min(Number(box.y1), Number(box.y2)), 0, state.imageHeight);
  const x2 = clamp(Math.max(Number(box.x1), Number(box.x2)), 0, state.imageWidth);
  const y2 = clamp(Math.max(Number(box.y1), Number(box.y2)), 0, state.imageHeight);
  return {
    ...box,
    class_id: Math.max(0, Number.parseInt(box.class_id || 0, 10)),
    x1,
    y1,
    x2: Math.max(x2, x1 + 1),
    y2: Math.max(y2, y1 + 1),
  };
}

function yoloLine(box) {
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;
  const cx = box.x1 + width * 0.5;
  const cy = box.y1 + height * 0.5;
  return [
    box.class_id,
    cx / state.imageWidth,
    cy / state.imageHeight,
    width / state.imageWidth,
    height / state.imageHeight,
  ]
    .map((value, index) => (index === 0 ? String(value) : Number(value).toFixed(6)))
    .join(" ");
}

function updateDownload(txt) {
  if (els.downloadLink.href) {
    URL.revokeObjectURL(els.downloadLink.href);
  }
  const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
  els.downloadLink.href = URL.createObjectURL(blob);
  els.downloadLink.download = `${state.imageId || "annotation"}.txt`;
  els.downloadLink.classList.remove("hidden");
}

function render() {
  renderBoxes();
  renderList();
  renderEditor();
  els.boxCount.textContent = String(state.boxes.length);
  els.deleteButton.disabled = !selectedBox();
  els.saveButton.disabled = !state.imageFile;
}

function renderBoxes() {
  els.boxLayer.innerHTML = "";
  const { scaleX, scaleY } = scaleInfo();
  for (const box of state.boxes) {
    const node = document.createElement("div");
    node.className = `bbox${box.id === state.selectedId ? " selected" : ""}`;
    node.style.left = `${box.x1 * scaleX}px`;
    node.style.top = `${box.y1 * scaleY}px`;
    node.style.width = `${Math.max((box.x2 - box.x1) * scaleX, 6)}px`;
    node.style.height = `${Math.max((box.y2 - box.y1) * scaleY, 6)}px`;
    node.dataset.id = box.id;

    const label = document.createElement("div");
    label.className = "bbox-label";
    label.textContent = `#${box.class_id}${box.score != null ? ` ${box.score.toFixed(2)}` : ""}`;
    node.appendChild(label);

    for (const corner of ["nw", "ne", "sw", "se"]) {
      const handle = document.createElement("div");
      handle.className = `resize-handle ${corner}`;
      handle.dataset.handle = corner;
      node.appendChild(handle);
    }

    node.addEventListener("pointerdown", onBoxPointerDown);
    els.boxLayer.appendChild(node);
  }
}

function renderList() {
  els.boxList.innerHTML = "";
  if (state.boxes.length === 0) {
    const empty = document.createElement("div");
    empty.className = "editor-empty";
    empty.textContent = "暂无 bbox。可以点击新增框，或调用模型预标注。";
    els.boxList.appendChild(empty);
    return;
  }

  for (const box of state.boxes) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `box-row${box.id === state.selectedId ? " selected" : ""}`;
    row.addEventListener("click", () => {
      state.selectedId = box.id;
      render();
    });

    const text = document.createElement("div");
    const title = document.createElement("div");
    title.className = "box-row-title";
    title.textContent = `Class ${box.class_id}`;
    const meta = document.createElement("div");
    meta.className = "box-row-meta";
    meta.textContent = `${Math.round(box.x1)}, ${Math.round(box.y1)} - ${Math.round(box.x2)}, ${Math.round(box.y2)}`;
    text.append(title, meta);

    const score = document.createElement("span");
    score.className = "score-pill";
    score.textContent = box.score == null ? "manual" : box.score.toFixed(2);
    row.append(text, score);
    els.boxList.appendChild(row);
  }
}

function renderEditor() {
  const box = selectedBox();
  if (!box) {
    els.editorEmpty.classList.remove("hidden");
    els.boxEditor.classList.add("hidden");
    return;
  }
  els.editorEmpty.classList.add("hidden");
  els.boxEditor.classList.remove("hidden");
  els.classIdInput.value = box.class_id;
  els.scoreReadout.value = box.score == null ? "manual" : box.score.toFixed(4);
  els.x1Input.value = Math.round(box.x1);
  els.y1Input.value = Math.round(box.y1);
  els.x2Input.value = Math.round(box.x2);
  els.y2Input.value = Math.round(box.y2);
}

function upsertSelectedFromInputs() {
  const box = selectedBox();
  if (!box) return;
  Object.assign(box, normalizeBox({
    ...box,
    class_id: els.classIdInput.value,
    x1: els.x1Input.value,
    y1: els.y1Input.value,
    x2: els.x2Input.value,
    y2: els.y2Input.value,
  }));
  setMessage("");
  render();
}

function addBoxAt(centerX = state.imageWidth * 0.5, centerY = state.imageHeight * 0.5) {
  const size = Math.max(32, Math.min(state.imageWidth, state.imageHeight) * 0.18);
  const box = normalizeBox({
    id: state.nextBoxId++,
    class_id: 0,
    class_name: "manual",
    score: null,
    x1: centerX - size * 0.5,
    y1: centerY - size * 0.5,
    x2: centerX + size * 0.5,
    y2: centerY + size * 0.5,
  });
  state.boxes.push(box);
  state.selectedId = box.id;
  setMessage("");
  render();
}

function onBoxPointerDown(event) {
  event.preventDefault();
  const boxNode = event.currentTarget;
  const id = Number(boxNode.dataset.id);
  const box = state.boxes.find((item) => item.id === id);
  if (!box) return;
  state.selectedId = id;
  const point = clientToImagePoint(event.clientX, event.clientY);
  state.interaction = {
    id,
    type: event.target.dataset.handle ? "resize" : "move",
    handle: event.target.dataset.handle || "",
    startPoint: point,
    startBox: { x1: box.x1, y1: box.y1, x2: box.x2, y2: box.y2 },
  };
  boxNode.setPointerCapture(event.pointerId);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp, { once: true });
  render();
}

function onPointerMove(event) {
  const interaction = state.interaction;
  if (!interaction) return;
  const box = state.boxes.find((item) => item.id === interaction.id);
  if (!box) return;

  const point = clientToImagePoint(event.clientX, event.clientY);
  const dx = point.x - interaction.startPoint.x;
  const dy = point.y - interaction.startPoint.y;
  const start = interaction.startBox;

  if (interaction.type === "move") {
    const width = start.x2 - start.x1;
    const height = start.y2 - start.y1;
    const x1 = clamp(start.x1 + dx, 0, state.imageWidth - width);
    const y1 = clamp(start.y1 + dy, 0, state.imageHeight - height);
    Object.assign(box, { x1, y1, x2: x1 + width, y2: y1 + height });
  } else {
    const next = { ...box };
    if (interaction.handle.includes("w")) next.x1 = start.x1 + dx;
    if (interaction.handle.includes("e")) next.x2 = start.x2 + dx;
    if (interaction.handle.includes("n")) next.y1 = start.y1 + dy;
    if (interaction.handle.includes("s")) next.y2 = start.y2 + dy;
    Object.assign(box, normalizeBox(next));
  }
  render();
}

function onPointerUp() {
  state.interaction = null;
  window.removeEventListener("pointermove", onPointerMove);
}

async function checkHealth() {
  setStatus("busy", "检测中");
  try {
    const response = await fetch(endpoint("/health"));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    setStatus(payload.success ? "ok" : "bad", payload.success ? "后端可用" : "后端异常");
  } catch (error) {
    setStatus("bad", `后端不可用: ${error.message}`);
  }
}

async function runPredict() {
  if (!state.imageFile) return;
  els.predictButton.disabled = true;
  els.predictButton.textContent = "推理中";
  setMessage("");
  setStatus("busy", "模型推理中");
  try {
    const form = new FormData();
    form.append("image", state.imageFile);
    form.append("score_threshold", els.scoreInput.value || "0.05");
    form.append("top_k", els.topKInput.value || "100");
    form.append("nms_iou_threshold", els.nmsInput.value || "0.5");
    const response = await fetch(endpoint("/model_predict"), { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok || !payload.success) {
      throw new Error(payload.error || `HTTP ${response.status}`);
    }
    state.boxes = payload.bboxes.map((box) => normalizeBox({
      id: state.nextBoxId++,
      class_id: box.class_id,
      class_name: box.class_name,
      score: box.score,
      x1: box.x1,
      y1: box.y1,
      x2: box.x2,
      y2: box.y2,
    }));
    state.selectedId = state.boxes[0]?.id || null;
    setStatus("ok", `推理完成，${state.boxes.length} 个框`);
    render();
  } catch (error) {
    setStatus("bad", "推理失败");
    setMessage(error.message, "bad");
  } finally {
    els.predictButton.disabled = !state.imageFile;
    els.predictButton.textContent = "模型预标注";
  }
}

async function saveAnnotation() {
  if (!state.imageFile) return;
  const cleanBoxes = state.boxes.map((box) => normalizeBox(box));
  const payload = {
    image_id: state.imageId,
    image_width: state.imageWidth,
    image_height: state.imageHeight,
    bboxes: cleanBoxes.map(({ class_id, x1, y1, x2, y2 }) => ({ class_id, x1, y1, x2, y2 })),
  };

  els.saveButton.disabled = true;
  els.saveButton.textContent = "保存中";
  try {
    const response = await fetch(endpoint("/human_annotate"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      throw new Error(result.error || `HTTP ${response.status}`);
    }
    const txt = `${cleanBoxes.map(yoloLine).join("\n")}${cleanBoxes.length ? "\n" : ""}`;
    updateDownload(txt);
    setMessage(`已保存 ${result.num_boxes} 个框: ${result.saved_path}`, "ok");
  } catch (error) {
    setMessage(error.message, "bad");
  } finally {
    els.saveButton.disabled = !state.imageFile;
    els.saveButton.textContent = "确认标注并保存";
  }
}

function loadImage(file) {
  if (!file) return;
  state.imageFile = file;
  state.imageId = file.name.replace(/\.[^.]+$/, "") || "uploaded_image";
  state.boxes = [];
  state.selectedId = null;
  state.nextBoxId = 1;
  setMessage("");
  els.downloadLink.classList.add("hidden");

  const objectUrl = URL.createObjectURL(file);
  els.previewImage.onload = () => {
    state.imageWidth = els.previewImage.naturalWidth;
    state.imageHeight = els.previewImage.naturalHeight;
    URL.revokeObjectURL(objectUrl);
    els.canvasShell.classList.remove("empty");
    els.emptyState.style.display = "none";
    els.predictButton.disabled = false;
    els.addBoxButton.disabled = false;
    els.fitButton.disabled = false;
    els.imageName.textContent = file.name;
    els.imageSize.textContent = `${state.imageWidth} x ${state.imageHeight}`;
    render();
  };
  els.previewImage.src = objectUrl;
}

function deleteSelected() {
  if (!selectedBox()) return;
  state.boxes = state.boxes.filter((box) => box.id !== state.selectedId);
  state.selectedId = state.boxes[0]?.id || null;
  setMessage("");
  render();
}

function bindEvents() {
  els.apiBaseInput.value = "";
  els.apiBaseInput.addEventListener("change", () => {
    state.apiBase = els.apiBaseInput.value.trim();
  });
  els.healthButton.addEventListener("click", checkHealth);
  els.imageInput.addEventListener("change", (event) => loadImage(event.target.files[0]));
  els.predictButton.addEventListener("click", runPredict);
  els.addBoxButton.addEventListener("click", () => addBoxAt());
  els.deleteButton.addEventListener("click", deleteSelected);
  els.saveButton.addEventListener("click", saveAnnotation);
  els.fitButton.addEventListener("click", () => els.previewImage.scrollIntoView({ block: "center", inline: "center" }));

  for (const input of [els.classIdInput, els.x1Input, els.y1Input, els.x2Input, els.y2Input]) {
    input.addEventListener("change", upsertSelectedFromInputs);
  }

  els.imageViewport.addEventListener("dblclick", (event) => {
    if (!state.imageFile || event.target !== els.boxLayer) return;
    const point = clientToImagePoint(event.clientX, event.clientY);
    addBoxAt(point.x, point.y);
  });

  window.addEventListener("resize", render);
  window.addEventListener("keydown", (event) => {
    if ((event.key === "Delete" || event.key === "Backspace") && selectedBox()) {
      const activeTag = document.activeElement?.tagName?.toLowerCase();
      if (activeTag !== "input") deleteSelected();
    }
  });
}

bindEvents();
checkHealth();
