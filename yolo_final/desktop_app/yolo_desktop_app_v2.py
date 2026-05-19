"""Offline YOLO desktop app with prediction, annotation editing, and loop check."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageFont, ImageTk


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import load_settings  # noqa: E402
from backend.torchscript_predictor import TorchScriptPredictor  # noqa: E402


DEFAULT_SCORE_THRESHOLD = 0.45
DEFAULT_NMS_IOU_THRESHOLD = 0.3
DEFAULT_TOP_K = 100
ANNOTATION_ROOT = PROJECT_ROOT / "desktop_app_annotations"
BOX_COLORS = ["#f97316", "#22c55e", "#3b82f6", "#eab308", "#ec4899", "#14b8a6", "#a855f7", "#ef4444"]


class YoloDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO 离线检测与标注工具")
        self.root.geometry("1320x820")
        self.root.minsize(1100, 700)

        self.predictor: TorchScriptPredictor | None = None
        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.annotations: list[dict] = []
        self.selected_index: int | None = None
        self.last_result: dict | None = None
        self.class_names: dict[int, str] = {}

        self.canvas_scale = 1.0
        self.canvas_offset = (0, 0)
        self.drag_start: tuple[float, float] | None = None
        self.preview_rect_id: int | None = None

        self.draw_mode = tk.BooleanVar(value=False)
        self.show_labels = tk.BooleanVar(value=True)
        self.score_var = tk.DoubleVar(value=DEFAULT_SCORE_THRESHOLD)
        self.nms_var = tk.DoubleVar(value=DEFAULT_NMS_IOU_THRESHOLD)
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.class_id_var = tk.IntVar(value=0)
        self.class_name_var = tk.StringVar(value="")
        self.x1_var = tk.DoubleVar(value=0.0)
        self.y1_var = tk.DoubleVar(value=0.0)
        self.x2_var = tk.DoubleVar(value=0.0)
        self.y2_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="请选择一张图片")
        self.summary_var = tk.StringVar(value="未检测")

        self._configure_style()
        self._build_ui()
        self._schedule_predictor_load()

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(bg="#0f172a")
        style.configure("Root.TFrame", background="#0f172a")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#0f172a", foreground="#f8fafc", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#0f172a", font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#1e293b", font=("Microsoft YaHei UI", 9))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = ttk.Frame(self.root, padding=(16, 12), style="Root.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(9, weight=1)
        ttk.Label(header, text="YOLO 离线检测与标注工具", style="Header.TLabel").grid(row=0, column=0, padx=(0, 18))
        self.open_button = ttk.Button(header, text="打开图片", command=self.open_image, style="Accent.TButton")
        self.open_button.grid(row=0, column=1, padx=(0, 8))
        self.detect_button = ttk.Button(header, text="模型预标注", command=self.run_detection, state="disabled", style="Accent.TButton")
        self.detect_button.grid(row=0, column=2, padx=(0, 8))
        self.save_image_button = ttk.Button(header, text="保存结果图", command=self.save_result_image, state="disabled")
        self.save_image_button.grid(row=0, column=3, padx=(0, 8))
        self.save_annotation_button = ttk.Button(header, text="保存标注", command=self.save_annotation, state="disabled")
        self.save_annotation_button.grid(row=0, column=4, padx=(0, 8))
        self.loop_button = ttk.Button(header, text="闭环验证 1 epoch", command=self.run_closed_loop_check, state="disabled")
        self.loop_button.grid(row=0, column=5, padx=(0, 18))
        ttk.Label(header, text="置信度", background="#0f172a", foreground="#dbeafe").grid(row=0, column=6, padx=(0, 4))
        ttk.Spinbox(header, from_=0.0, to=1.0, increment=0.05, textvariable=self.score_var, width=6).grid(row=0, column=7, padx=(0, 10))
        ttk.Label(header, text="NMS", background="#0f172a", foreground="#dbeafe").grid(row=0, column=8, padx=(0, 4))
        ttk.Spinbox(header, from_=0.0, to=1.0, increment=0.05, textvariable=self.nms_var, width=6).grid(row=0, column=9, sticky="w")
        ttk.Label(header, textvariable=self.status_var, background="#0f172a", foreground="#cbd5e1").grid(row=0, column=10, sticky="e")

        content = ttk.Frame(self.root, padding=(16, 0, 16, 16), style="Root.TFrame")
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)

        image_panel = ttk.Frame(content, style="Panel.TFrame")
        image_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(1, weight=1)
        ttk.Label(image_panel, text="图像预览与框修正", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))
        self.canvas = tk.Canvas(image_panel, bg="#111827", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", lambda _event: self._render_canvas())
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        side = ttk.Frame(content, width=360, style="Panel.TFrame")
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        ttk.Label(side, text="检测概览", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        ttk.Label(side, textvariable=self.summary_var, style="Body.TLabel", wraplength=328, justify="left").grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 10))

        options = ttk.Frame(side, style="Panel.TFrame")
        options.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 8))
        ttk.Checkbutton(options, text="拖拽新增框", variable=self.draw_mode).pack(side="left")
        ttk.Checkbutton(options, text="显示标签", variable=self.show_labels, command=self._render_canvas).pack(side="left", padx=(10, 0))

        list_frame = ttk.Frame(side, style="Panel.TFrame")
        list_frame.grid(row=3, column=0, sticky="nsew", padx=12)
        side.rowconfigure(3, weight=1)
        self.result_list = tk.Listbox(list_frame, height=14, bg="#f8fafc", fg="#0f172a", selectbackground="#2563eb", activestyle="none")
        self.result_list.pack(side="left", fill="both", expand=True)
        self.result_list.bind("<<ListboxSelect>>", self._on_list_select)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.result_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_list.configure(yscrollcommand=scrollbar.set)

        edit = ttk.Frame(side, style="Panel.TFrame")
        edit.grid(row=4, column=0, sticky="ew", padx=12, pady=(10, 8))
        for col in range(4):
            edit.columnconfigure(col, weight=1)
        ttk.Label(edit, text="人工修正", style="PanelTitle.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 6))
        ttk.Label(edit, text="类别ID", style="Body.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Spinbox(edit, from_=0, to=79, increment=1, textvariable=self.class_id_var, width=8).grid(row=1, column=1, sticky="ew", padx=(4, 8))
        ttk.Label(edit, text="类别名", style="Body.TLabel").grid(row=1, column=2, sticky="w")
        ttk.Entry(edit, textvariable=self.class_name_var, width=12).grid(row=1, column=3, sticky="ew")
        for row, (label, var) in enumerate([("x1", self.x1_var), ("y1", self.y1_var), ("x2", self.x2_var), ("y2", self.y2_var)], start=2):
            ttk.Label(edit, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(edit, textvariable=var, width=10).grid(row=row, column=1, columnspan=3, sticky="ew", pady=2)
        ttk.Button(edit, text="应用修改", command=self.apply_selected_box).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0), padx=(0, 4))
        ttk.Button(edit, text="删除选中框", command=self.delete_selected_box).grid(row=6, column=2, columnspan=2, sticky="ew", pady=(8, 0), padx=(4, 0))
        ttk.Label(side, text="闭环：新图片 → 模型预标注 → 人工修正 → 保存标注 → 1 epoch loss 验证", style="Hint.TLabel", wraplength=328).grid(row=5, column=0, sticky="ew", padx=12, pady=(0, 12))

    def _schedule_predictor_load(self) -> None:
        self.status_var.set("正在加载模型...")
        threading.Thread(target=self._load_predictor_worker, daemon=True).start()

    def _load_predictor_worker(self) -> None:
        try:
            settings = load_settings()
            predictor = TorchScriptPredictor(settings.torchscript_model_path, settings.config_path, settings.device, settings.metadata_path, settings.use_fp16)
            self.root.after(0, self._on_predictor_loaded, predictor)
        except Exception as exc:
            self.root.after(0, self._on_predictor_error, exc)

    def _on_predictor_loaded(self, predictor: TorchScriptPredictor) -> None:
        self.predictor = predictor
        self.class_names = dict(getattr(predictor, "class_names", {}) or {})
        self.status_var.set(f"模型已加载: {getattr(predictor, 'device', 'unknown')}")
        if self.original_image is not None:
            self.detect_button.configure(state="normal")

    def _on_predictor_error(self, exc: Exception) -> None:
        self.status_var.set("模型加载失败")
        messagebox.showerror("模型加载失败", str(exc))

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")])
        if not file_path:
            return
        try:
            image = Image.open(file_path).convert("RGB")
            image.load()
        except Exception as exc:
            messagebox.showerror("无法打开图片", str(exc))
            return
        self.image_path = Path(file_path)
        self.original_image = image
        self.annotations = []
        self.selected_index = None
        self.last_result = None
        self._refresh_annotation_list()
        self.status_var.set(f"已打开: {self.image_path.name}")
        self.save_image_button.configure(state="disabled")
        self.save_annotation_button.configure(state="disabled")
        self.loop_button.configure(state="normal")
        if self.predictor is not None:
            self.detect_button.configure(state="normal")
        self._render_canvas()

    def run_detection(self) -> None:
        if self.predictor is None:
            messagebox.showinfo("模型尚未加载", "请等待模型加载完成。")
            return
        if self.original_image is None:
            messagebox.showinfo("未选择图片", "请先打开一张图片。")
            return
        self._set_busy(True, "检测中...")
        threading.Thread(
            target=self._detect_worker,
            args=(self.original_image.copy(), float(self.score_var.get()), float(self.nms_var.get()), int(self.topk_var.get())),
            daemon=True,
        ).start()

    def _detect_worker(self, image: Image.Image, score: float, nms_iou: float, top_k: int) -> None:
        try:
            assert self.predictor is not None
            result = self.predictor.predict(image=image, score_threshold=score, top_k=top_k, nms_iou_threshold=nms_iou)
            self.root.after(0, self._on_detection_done, result)
        except Exception as exc:
            self.root.after(0, self._on_detection_error, exc)

    def _on_detection_done(self, result: dict) -> None:
        self.last_result = result
        self.annotations = [self._normalize_prediction_box(item) for item in result.get("bboxes", [])]
        self.annotations = [item for item in self.annotations if item is not None]
        self.selected_index = 0 if self.annotations else None
        self._refresh_annotation_list()
        self._populate_editor()
        self._render_canvas()
        self._set_busy(False, f"预标注完成: {float(result.get('latency_ms', {}).get('total', 0.0)):.1f} ms")
        self.save_image_button.configure(state="normal")
        self.save_annotation_button.configure(state="normal")
        self.loop_button.configure(state="normal")

    def _on_detection_error(self, exc: Exception) -> None:
        self._set_busy(False, "检测失败")
        messagebox.showerror("检测失败", str(exc))

    def _set_busy(self, busy: bool, status: str) -> None:
        self.status_var.set(status)
        state = "disabled" if busy else "normal"
        self.open_button.configure(state=state)
        if self.original_image is not None and self.predictor is not None:
            self.detect_button.configure(state=state)

    def _normalize_prediction_box(self, item: dict) -> dict | None:
        return self._sanitize_box(
            {
                "class_id": int(item.get("class_id", 0)),
                "class_name": str(item.get("class_name") or self.class_names.get(int(item.get("class_id", 0)), "")),
                "score": float(item.get("score", 1.0)),
                "x1": float(item.get("x1", 0.0)),
                "y1": float(item.get("y1", 0.0)),
                "x2": float(item.get("x2", 0.0)),
                "y2": float(item.get("y2", 0.0)),
            }
        )

    def _refresh_annotation_list(self) -> None:
        self.result_list.delete(0, tk.END)
        for index, box in enumerate(self.annotations, start=1):
            name = box.get("class_name") or self.class_names.get(int(box.get("class_id", 0)), str(box.get("class_id", "")))
            self.result_list.insert(tk.END, f"{index:02d}. {name}  {float(box.get('score', 1.0)):.3f}")
        if not self.annotations:
            self.result_list.insert(tk.END, "暂无标注框")
        if self.selected_index is not None and self.annotations:
            self.result_list.selection_set(self.selected_index)
        self._update_summary()

    def _update_summary(self) -> None:
        if self.original_image is None:
            self.summary_var.set("未检测")
            return
        self.summary_var.set(
            f"当前标注框: {len(self.annotations)}\n图片尺寸: {self.original_image.width} x {self.original_image.height}\n"
            f"置信度阈值: {float(self.score_var.get()):.2f}\nNMS IoU: {float(self.nms_var.get()):.2f}"
        )

    def _on_list_select(self, _event) -> None:
        selection = self.result_list.curselection()
        if not selection or not self.annotations:
            return
        index = int(selection[0])
        if index >= len(self.annotations):
            return
        self.selected_index = index
        self._populate_editor()
        self._render_canvas()

    def _populate_editor(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.annotations):
            return
        box = self.annotations[self.selected_index]
        class_id = int(box.get("class_id", 0))
        self.class_id_var.set(class_id)
        self.class_name_var.set(str(box.get("class_name") or self.class_names.get(class_id, "")))
        self.x1_var.set(round(float(box["x1"]), 1))
        self.y1_var.set(round(float(box["y1"]), 1))
        self.x2_var.set(round(float(box["x2"]), 1))
        self.y2_var.set(round(float(box["y2"]), 1))

    def apply_selected_box(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.annotations):
            messagebox.showinfo("未选中标注框", "请先在右侧列表中选中一个框。")
            return
        class_id = int(self.class_id_var.get())
        box = self._sanitize_box(
            {
                "class_id": class_id,
                "class_name": self.class_name_var.get().strip() or self.class_names.get(class_id, str(class_id)),
                "score": self.annotations[self.selected_index].get("score", 1.0),
                "x1": float(self.x1_var.get()),
                "y1": float(self.y1_var.get()),
                "x2": float(self.x2_var.get()),
                "y2": float(self.y2_var.get()),
            }
        )
        if box is None:
            messagebox.showerror("无效框", "请确保 x2 > x1 且 y2 > y1。")
            return
        self.annotations[self.selected_index] = box
        self._refresh_annotation_list()
        self._render_canvas()

    def delete_selected_box(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.annotations):
            return
        del self.annotations[self.selected_index]
        self.selected_index = min(self.selected_index, len(self.annotations) - 1) if self.annotations else None
        self._refresh_annotation_list()
        self._populate_editor()
        self._render_canvas()

    def _sanitize_box(self, box: dict) -> dict | None:
        if self.original_image is None:
            return None
        x1 = max(0.0, min(float(box["x1"]), float(self.original_image.width)))
        y1 = max(0.0, min(float(box["y1"]), float(self.original_image.height)))
        x2 = max(0.0, min(float(box["x2"]), float(self.original_image.width)))
        y2 = max(0.0, min(float(box["y2"]), float(self.original_image.height)))
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            return None
        return {
            "class_id": max(0, min(int(box.get("class_id", 0)), 79)),
            "class_name": str(box.get("class_name", "")),
            "score": float(box.get("score", 1.0)),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }

    def _canvas_to_image(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        ox, oy = self.canvas_offset
        return (canvas_x - ox) / self.canvas_scale, (canvas_y - oy) / self.canvas_scale

    def _image_to_canvas(self, image_x: float, image_y: float) -> tuple[float, float]:
        ox, oy = self.canvas_offset
        return ox + image_x * self.canvas_scale, oy + image_y * self.canvas_scale

    def _on_canvas_press(self, event) -> None:
        if self.original_image is None:
            return
        if self.draw_mode.get():
            self.drag_start = self._canvas_to_image(event.x, event.y)
            if self.preview_rect_id is not None:
                self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#22c55e", width=2, dash=(4, 3))
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        for index, box in reversed(list(enumerate(self.annotations))):
            if box["x1"] <= image_x <= box["x2"] and box["y1"] <= image_y <= box["y2"]:
                self.selected_index = index
                self._refresh_annotation_list()
                self._populate_editor()
                self._render_canvas()
                return

    def _on_canvas_drag(self, event) -> None:
        if self.drag_start is None or self.preview_rect_id is None:
            return
        sx, sy = self._image_to_canvas(*self.drag_start)
        self.canvas.coords(self.preview_rect_id, sx, sy, event.x, event.y)

    def _on_canvas_release(self, event) -> None:
        if not self.draw_mode.get() or self.drag_start is None:
            return
        end_x, end_y = self._canvas_to_image(event.x, event.y)
        start_x, start_y = self.drag_start
        self.drag_start = None
        if self.preview_rect_id is not None:
            self.canvas.delete(self.preview_rect_id)
            self.preview_rect_id = None
        class_id = int(self.class_id_var.get())
        box = self._sanitize_box(
            {
                "class_id": class_id,
                "class_name": self.class_name_var.get().strip() or self.class_names.get(class_id, str(class_id)),
                "score": 1.0,
                "x1": start_x,
                "y1": start_y,
                "x2": end_x,
                "y2": end_y,
            }
        )
        if box is None:
            return
        self.annotations.append(box)
        self.selected_index = len(self.annotations) - 1
        self._refresh_annotation_list()
        self._populate_editor()
        self._render_canvas()
        self.save_annotation_button.configure(state="normal")

    def _draw_annotations(self, image: Image.Image) -> Image.Image:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, image.width // 85))
        except OSError:
            font = ImageFont.load_default()
        line_width = max(2, image.width // 320)
        for index, box in enumerate(self.annotations):
            class_id = int(box.get("class_id", 0))
            color = "#22c55e" if index == self.selected_index else BOX_COLORS[class_id % len(BOX_COLORS)]
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            if self.show_labels.get():
                label = f"{box.get('class_name') or class_id} {float(box.get('score', 1.0)):.2f}"
                label_box = draw.textbbox((x1, y1), label, font=font)
                label_w = label_box[2] - label_box[0]
                label_h = label_box[3] - label_box[1]
                label_y = max(0, y1 - label_h - 6)
                draw.rectangle((x1, label_y, x1 + label_w + 8, label_y + label_h + 6), fill=color)
                draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)
        return image

    def _render_canvas(self) -> None:
        if self.original_image is None:
            self.canvas.delete("all")
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            self.canvas.create_text(width // 2, height // 2, text="打开图片后运行模型预标注", fill="#d1d5db", font=("Microsoft YaHei UI", 16))
            return
        display_image = self._draw_annotations(self.original_image.copy())
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        self.canvas_scale = min(canvas_w / display_image.width, canvas_h / display_image.height)
        display_w = max(1, int(display_image.width * self.canvas_scale))
        display_h = max(1, int(display_image.height * self.canvas_scale))
        resized = display_image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        x = (canvas_w - display_w) // 2
        y = (canvas_h - display_h) // 2
        self.canvas_offset = (x, y)
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

    def save_result_image(self) -> None:
        if self.original_image is None:
            return
        default_name = f"{self.image_path.stem}_detected.jpg" if self.image_path else "yolo_detection_result.jpg"
        save_path = filedialog.asksaveasfilename(title="保存检测结果", defaultextension=".jpg", initialfile=default_name, filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png"), ("All files", "*.*")])
        if not save_path:
            return
        self._draw_annotations(self.original_image.copy()).save(save_path)
        self.status_var.set(f"已保存结果图: {Path(save_path).name}")

    def save_annotation(self) -> None:
        if self.original_image is None or self.image_path is None:
            return
        image_dir = ANNOTATION_ROOT / "images"
        label_dir = ANNOTATION_ROOT / "labels"
        vis_dir = ANNOTATION_ROOT / "visualizations"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)
        stem = self.image_path.stem
        image_ext = self.image_path.suffix.lower() or ".jpg"
        saved_image = image_dir / f"{stem}{image_ext}"
        label_path = label_dir / f"{stem}.txt"
        vis_path = vis_dir / f"{stem}_annotation.jpg"
        shutil.copy2(self.image_path, saved_image)
        label_path.write_text(self._to_yolo_text(), encoding="utf-8")
        self._draw_annotations(self.original_image.copy()).save(vis_path)
        manifest = {
            "image_path": str(saved_image),
            "label_path": str(label_path),
            "visualization_path": str(vis_path),
            "width": self.original_image.width,
            "height": self.original_image.height,
            "num_boxes": len(self.annotations),
        }
        (ANNOTATION_ROOT / "latest_annotation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_var.set(f"标注已保存: {label_path.name}")
        self.loop_button.configure(state="normal")
        messagebox.showinfo("保存完成", f"标注已保存到:\n{label_path}\n\n可视化图:\n{vis_path}")

    def _to_yolo_text(self) -> str:
        if self.original_image is None:
            return ""
        width = float(self.original_image.width)
        height = float(self.original_image.height)
        lines = []
        for box in self.annotations:
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            cx = ((x1 + x2) / 2.0) / width
            cy = ((y1 + y2) / 2.0) / height
            bw = (x2 - x1) / width
            bh = (y2 - y1) / height
            lines.append(f"{int(box['class_id'])} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        return "\n".join(lines) + ("\n" if lines else "")

    def run_closed_loop_check(self) -> None:
        script = PROJECT_ROOT / "desktop_app" / "closed_loop_train_verify.py"
        if not script.exists():
            messagebox.showerror("缺少脚本", f"找不到闭环验证脚本:\n{script}")
            return
        self.status_var.set("正在执行闭环验证...")
        self.loop_button.configure(state="disabled")
        threading.Thread(target=self._closed_loop_worker, args=(script,), daemon=True).start()

    def _closed_loop_worker(self, script: Path) -> None:
        cmd = [sys.executable, str(script), "--annotation-root", str(ANNOTATION_ROOT)]
        completed = subprocess.run(cmd, cwd=str(PROJECT_ROOT), text=True, capture_output=True, check=False)
        self.root.after(0, self._on_closed_loop_done, completed)

    def _on_closed_loop_done(self, completed: subprocess.CompletedProcess) -> None:
        self.loop_button.configure(state="normal")
        output = (completed.stdout or "") + ("\n" + completed.stderr if completed.stderr else "")
        if completed.returncode == 0:
            self.status_var.set("闭环验证完成")
            messagebox.showinfo("闭环验证完成", output[-1800:] or "验证完成")
        else:
            self.status_var.set("闭环验证失败")
            messagebox.showerror("闭环验证失败", output[-1800:] or "请检查训练环境")


def main() -> None:
    os.environ.setdefault("YOLO_BACKEND_MODEL_FORMAT", "torchscript")
    os.environ.setdefault("YOLO_BACKEND_DEVICE", "auto")
    root = tk.Tk()
    YoloDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
