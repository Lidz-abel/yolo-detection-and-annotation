import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  Bot,
  Check,
  Download,
  FileImage,
  ImagePlus,
  Plus,
  Redo2,
  Save,
  Trash2,
  Undo2
} from "lucide-react";
import { health, predictImage, saveAnnotation } from "./api";
import { createManualBox, downloadTxt, normalizeBox, toYoloTxt } from "./boxUtils";
import { useBoxHistory } from "./useBoxHistory";
import type { Box, Interaction } from "./types";
import "./styles.css";

type ApiState = "idle" | "ok" | "bad" | "busy";

function App() {
  const [apiBase, setApiBase] = useState("");
  const [apiState, setApiState] = useState<ApiState>("idle");
  const [apiText, setApiText] = useState("后端未检测");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState("");
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [scoreThreshold, setScoreThreshold] = useState(0.5);
  const [topK, setTopK] = useState(100);
  const [nmsIou, setNmsIou] = useState(0.3);
  const [message, setMessage] = useState("");
  const [messageType, setMessageType] = useState<"ok" | "bad" | "">("");
  const [isPredicting, setIsPredicting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const interactionRef = useRef<Interaction | null>(null);
  const draftRef = useRef<Box[] | null>(null);
  const history = useBoxHistory();

  const imageId = useMemo(() => imageFile?.name.replace(/\.[^.]+$/, "") || "uploaded_image", [imageFile]);
  const selectedBox = useMemo(() => history.boxes.find((box) => box.id === selectedId) || null, [history.boxes, selectedId]);

  const endpointBase = apiBase.trim();
  const viewportSize = useMemo(() => {
    if (!imageSize.width || !imageSize.height || !canvasSize.width || !canvasSize.height) {
      return { width: 0, height: 0 };
    }
    const maxWidth = Math.max(260, canvasSize.width - 34);
    const maxHeight = Math.max(220, canvasSize.height - 34);
    const fitScale = Math.min(maxWidth / imageSize.width, maxHeight / imageSize.height);
    return {
      width: Math.max(1, imageSize.width * fitScale),
      height: Math.max(1, imageSize.height * fitScale)
    };
  }, [canvasSize.height, canvasSize.width, imageSize.height, imageSize.width]);
  const displayScale = useMemo(
    () => ({
      x: viewportSize.width / Math.max(imageSize.width, 1),
      y: viewportSize.height / Math.max(imageSize.height, 1)
    }),
    [imageSize.height, imageSize.width, viewportSize.height, viewportSize.width]
  );

  const checkHealth = useCallback(async () => {
    setApiState("busy");
    setApiText("检测中");
    try {
      const ok = await health(endpointBase);
      setApiState(ok ? "ok" : "bad");
      setApiText(ok ? "后端可用" : "后端异常");
    } catch (error) {
      setApiState("bad");
      setApiText(error instanceof Error ? error.message : "后端不可用");
    }
  }, [endpointBase]);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  useEffect(() => {
    const target = canvasRef.current;
    if (!target) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0]?.contentRect;
      if (!rect) return;
      setCanvasSize({ width: rect.width, height: rect.height });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const tag = document.activeElement?.tagName.toLowerCase();
      const typing = tag === "input" || tag === "textarea";
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) history.redo();
        else history.undo();
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        history.redo();
      }
      if (!typing && (event.key === "Delete" || event.key === "Backspace") && selectedId) {
        event.preventDefault();
        deleteSelected();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  function imagePoint(clientX: number, clientY: number) {
    const rect = imageRef.current?.getBoundingClientRect();
    if (!rect) return { x: 0, y: 0 };
    return {
      x: Math.max(0, Math.min(((clientX - rect.left) * imageSize.width) / Math.max(rect.width, 1), imageSize.width)),
      y: Math.max(0, Math.min(((clientY - rect.top) * imageSize.height) / Math.max(rect.height, 1), imageSize.height))
    };
  }

  function onFileChange(file: File | undefined) {
    if (!file) return;
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    const nextUrl = URL.createObjectURL(file);
    setImageFile(file);
    setImageUrl(nextUrl);
    setImageSize({ width: 0, height: 0 });
    setSelectedId(null);
    setMessage("");
    setMessageType("");
    history.reset([]);
  }

  async function runPredict() {
    if (!imageFile) return;
    setIsPredicting(true);
    setApiState("busy");
    setApiText("模型推理中");
    setMessage("");
    try {
      const payload = await predictImage({
        baseUrl: endpointBase,
        file: imageFile,
        scoreThreshold,
        topK,
        nmsIouThreshold: nmsIou
      });
      const boxes = payload.bboxes.map((box) =>
        normalizeBox({ ...box, id: crypto.randomUUID() }, payload.image_width, payload.image_height)
      );
      history.replace(boxes);
      setSelectedId(boxes[0]?.id || null);
      setApiState("ok");
      setApiText(`推理完成，${boxes.length} 个框`);
    } catch (error) {
      setApiState("bad");
      setApiText("推理失败");
      setMessage(error instanceof Error ? error.message : "推理失败");
      setMessageType("bad");
    } finally {
      setIsPredicting(false);
    }
  }

  function addBox() {
    if (!imageSize.width || !imageSize.height) return;
    const box = createManualBox(imageSize.width, imageSize.height);
    history.replace([...history.boxes, box]);
    setSelectedId(box.id);
    setMessage("");
  }

  function deleteSelected() {
    if (!selectedId) return;
    const boxes = history.boxes.filter((box) => box.id !== selectedId);
    history.replace(boxes);
    setSelectedId(boxes[0]?.id || null);
  }

  function updateSelected(patch: Partial<Box>) {
    if (!selectedBox) return;
    const boxes = history.boxes.map((box) =>
      box.id === selectedBox.id ? normalizeBox({ ...box, ...patch }, imageSize.width, imageSize.height) : box
    );
    history.replace(boxes);
  }

  function beginInteraction(event: React.PointerEvent<HTMLElement>, box: Box, handle: Interaction["handle"] = "") {
    event.preventDefault();
    setSelectedId(box.id);
    interactionRef.current = {
      id: box.id,
      mode: handle ? "resize" : "move",
      handle,
      startPoint: imagePoint(event.clientX, event.clientY),
      startBox: { ...box }
    };
    draftRef.current = history.boxes.map((item) => ({ ...item }));
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function moveInteraction(event: React.PointerEvent<HTMLDivElement>) {
    const interaction = interactionRef.current;
    const draft = draftRef.current;
    if (!interaction || !draft) return;
    const point = imagePoint(event.clientX, event.clientY);
    const dx = point.x - interaction.startPoint.x;
    const dy = point.y - interaction.startPoint.y;
    const start = interaction.startBox;

    const boxes = draft.map((box) => {
      if (box.id !== interaction.id) return box;
      if (interaction.mode === "move") {
        const width = start.x2 - start.x1;
        const height = start.y2 - start.y1;
        const x1 = Math.max(0, Math.min(start.x1 + dx, imageSize.width - width));
        const y1 = Math.max(0, Math.min(start.y1 + dy, imageSize.height - height));
        return { ...box, x1, y1, x2: x1 + width, y2: y1 + height };
      }
      const next = { ...box };
      if (interaction.handle.includes("w")) next.x1 = start.x1 + dx;
      if (interaction.handle.includes("e")) next.x2 = start.x2 + dx;
      if (interaction.handle.includes("n")) next.y1 = start.y1 + dy;
      if (interaction.handle.includes("s")) next.y2 = start.y2 + dy;
      return normalizeBox(next, imageSize.width, imageSize.height);
    });
    history.replace(boxes, false);
  }

  function endInteraction() {
    if (interactionRef.current) {
      history.replace(history.boxes, true);
    }
    interactionRef.current = null;
    draftRef.current = null;
  }

  async function confirmAnnotation() {
    if (!imageFile) return;
    setIsSaving(true);
    setMessage("");
    try {
      const result = await saveAnnotation({
        baseUrl: endpointBase,
        file: imageFile,
        imageId,
        imageWidth: imageSize.width,
        imageHeight: imageSize.height,
        boxes: history.boxes
      });
      downloadTxt(`${imageId}.txt`, toYoloTxt(history.boxes, imageSize.width, imageSize.height));
      const imageNote = result.saved_image_path ? `，原图: ${result.saved_image_path}` : "";
      setMessage(`已保存 ${result.num_boxes || 0} 个框: ${result.saved_path}${imageNote}`);
      setMessageType("ok");
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "保存失败");
      setMessageType("bad");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brandIcon"><Bot size={22} /></div>
          <div>
            <h1>YOLO React 标注工作台</h1>
            <p>模型预标注、人工修正、状态回退、YOLO txt 导出</p>
          </div>
        </div>
        <div className="health">
          <span className={`healthDot ${apiState}`} />
          <span>{apiText}</span>
        </div>
      </header>

      <main className="layout">
        <section className="stage">
          <div className="toolbar">
            <label className="uploadButton">
              <ImagePlus size={17} />
              上传图片
              <input type="file" accept="image/*" onChange={(event) => onFileChange(event.target.files?.[0])} />
            </label>
            <button className="primary" disabled={!imageFile || isPredicting} onClick={runPredict}>
              <Activity size={17} />
              {isPredicting ? "推理中" : "模型预标注"}
            </button>
            <button disabled={!imageFile} onClick={addBox}><Plus size={17} />新增框</button>
            <button disabled={!history.canUndo} onClick={history.undo}><Undo2 size={17} />撤销</button>
            <button disabled={!history.canRedo} onClick={history.redo}><Redo2 size={17} />重做</button>
          </div>

          <div ref={canvasRef} className={`canvas ${imageFile ? "" : "empty"}`}>
            {!imageFile && (
              <div className="emptyState">
                <FileImage size={44} />
                <h2>上传图片开始标注</h2>
                <p>React 版本支持拖拽、缩放、表单精修和撤销重做。</p>
              </div>
            )}
            {imageUrl && (
              <div
                className="viewport"
                style={{
                  width: viewportSize.width || undefined,
                  height: viewportSize.height || undefined
                }}
              >
                <img
                  ref={imageRef}
                  src={imageUrl}
                  alt="待标注图片"
                  onLoad={(event) => setImageSize({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight
                  })}
                />
                <div className="overlay" onPointerMove={moveInteraction} onPointerUp={endInteraction}>
                  {history.boxes.map((box) => (
                    <div
                      key={box.id}
                      className={`bbox ${box.id === selectedId ? "selected" : ""}`}
                      style={{
                        left: box.x1 * displayScale.x,
                        top: box.y1 * displayScale.y,
                        width: Math.max((box.x2 - box.x1) * displayScale.x, 8),
                        height: Math.max((box.y2 - box.y1) * displayScale.y, 8)
                      }}
                      onPointerDown={(event) => beginInteraction(event, box)}
                    >
                      <span>{box.class_name || `class ${box.class_id}`} {box.score == null ? "manual" : box.score.toFixed(2)}</span>
                      {(["nw", "ne", "sw", "se"] as const).map((handle) => (
                        <i
                          key={handle}
                          className={`handle ${handle}`}
                          onPointerDown={(event) => {
                            event.stopPropagation();
                            beginInteraction(event, box, handle);
                          }}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        <aside className="sidebar">
          <section>
            <div className="sectionHead">
              <h2>推理设置</h2>
              <button className="small" onClick={checkHealth}>检测</button>
            </div>
            <label className="field">
              <span>后端地址</span>
              <input value={apiBase} onChange={(event) => setApiBase(event.target.value)} placeholder="同源或 http://127.0.0.1:5000" />
            </label>
            <div className="grid3">
              <label className="field"><span>置信度</span><input type="number" step="0.01" min="0" max="1" value={scoreThreshold} onChange={(event) => setScoreThreshold(Number(event.target.value))} /></label>
              <label className="field"><span>Top K</span><input type="number" min="1" value={topK} onChange={(event) => setTopK(Number(event.target.value))} /></label>
              <label className="field"><span>NMS</span><input type="number" step="0.01" min="0" max="1" value={nmsIou} onChange={(event) => setNmsIou(Number(event.target.value))} /></label>
            </div>
          </section>

          <section>
            <h2>图片信息</h2>
            <dl className="info">
              <div><dt>文件</dt><dd>{imageFile?.name || "未上传"}</dd></div>
              <div><dt>尺寸</dt><dd>{imageSize.width ? `${imageSize.width} x ${imageSize.height}` : "-"}</dd></div>
              <div><dt>框数量</dt><dd>{history.boxes.length}</dd></div>
            </dl>
          </section>

          <section className="listSection">
            <div className="sectionHead">
              <h2>标注列表</h2>
              <button className="danger small" disabled={!selectedBox} onClick={deleteSelected}><Trash2 size={15} />删除</button>
            </div>
            <div className="boxList">
              {history.boxes.length === 0 && <p className="muted">暂无 bbox。</p>}
              {history.boxes.map((box, index) => (
                <button
                  key={box.id}
                  className={`boxRow ${box.id === selectedId ? "active" : ""}`}
                  onClick={() => setSelectedId(box.id)}
                >
                  <strong>{index + 1}. Class {box.class_id}</strong>
                  <span>{Math.round(box.x1)}, {Math.round(box.y1)} - {Math.round(box.x2)}, {Math.round(box.y2)}</span>
                  <em>{box.score == null ? "manual" : box.score.toFixed(2)}</em>
                </button>
              ))}
            </div>
          </section>

          <section>
            <h2>选中框精修</h2>
            {!selectedBox && <p className="muted">选择一个 bbox 后编辑坐标。</p>}
            {selectedBox && (
              <div className="editor">
                <label className="field"><span>Class ID</span><input type="number" min="0" value={selectedBox.class_id} onChange={(event) => updateSelected({ class_id: Number(event.target.value) })} /></label>
                <label className="field"><span>Score</span><input disabled value={selectedBox.score == null ? "manual" : selectedBox.score.toFixed(4)} /></label>
                <label className="field"><span>X1</span><input type="number" value={Math.round(selectedBox.x1)} onChange={(event) => updateSelected({ x1: Number(event.target.value) })} /></label>
                <label className="field"><span>Y1</span><input type="number" value={Math.round(selectedBox.y1)} onChange={(event) => updateSelected({ y1: Number(event.target.value) })} /></label>
                <label className="field"><span>X2</span><input type="number" value={Math.round(selectedBox.x2)} onChange={(event) => updateSelected({ x2: Number(event.target.value) })} /></label>
                <label className="field"><span>Y2</span><input type="number" value={Math.round(selectedBox.y2)} onChange={(event) => updateSelected({ y2: Number(event.target.value) })} /></label>
              </div>
            )}
          </section>

          <section>
            <button className="confirm" disabled={!imageFile || isSaving} onClick={confirmAnnotation}>
              {isSaving ? <Activity size={17} /> : <Save size={17} />}
              {isSaving ? "保存中" : "确认标注并保存"}
            </button>
            {imageFile && (
              <button className="download" onClick={() => downloadTxt(`${imageId}.txt`, toYoloTxt(history.boxes, imageSize.width, imageSize.height))}>
                <Download size={17} />下载 txt 副本
              </button>
            )}
            {message && <p className={`message ${messageType}`}>{messageType === "ok" && <Check size={15} />}{message}</p>}
          </section>
        </aside>
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
