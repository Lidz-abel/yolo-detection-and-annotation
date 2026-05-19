"""Simple Windows-friendly desktop UI for YOLO image detection.

The app reuses the exported TorchScript predictor and Python post-processing
from the Flask backend. It does not start a web server.
"""

from __future__ import annotations

import os
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
BOX_COLORS = [
    "#f97316",
    "#22c55e",
    "#3b82f6",
    "#eab308",
    "#ec4899",
    "#14b8a6",
    "#a855f7",
    "#ef4444",
]


class YoloDesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO 检测工具")
        self.root.geometry("1180x760")
        self.root.minsize(980, 640)

        self.predictor: TorchScriptPredictor | None = None
        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.result_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.last_result: dict | None = None

        self.score_var = tk.DoubleVar(value=DEFAULT_SCORE_THRESHOLD)
        self.nms_var = tk.DoubleVar(value=DEFAULT_NMS_IOU_THRESHOLD)
        self.topk_var = tk.IntVar(value=DEFAULT_TOP_K)
        self.status_var = tk.StringVar(value="请选择一张图片")
        self.summary_var = tk.StringVar(value="未检测")

        self._build_ui()
        self._schedule_predictor_load()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self.root, padding=(12, 10))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(8, weight=1)

        self.open_button = ttk.Button(toolbar, text="打开图片", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=(0, 8))

        self.detect_button = ttk.Button(toolbar, text="运行检测", command=self.run_detection, state="disabled")
        self.detect_button.grid(row=0, column=1, padx=(0, 16))

        self.save_button = ttk.Button(toolbar, text="保存结果", command=self.save_result, state="disabled")
        self.save_button.grid(row=0, column=2, padx=(0, 18))

        ttk.Label(toolbar, text="置信度").grid(row=0, column=3, padx=(0, 4))
        ttk.Spinbox(toolbar, from_=0.0, to=1.0, increment=0.05, textvariable=self.score_var, width=6).grid(
            row=0, column=4, padx=(0, 12)
        )

        ttk.Label(toolbar, text="NMS IoU").grid(row=0, column=5, padx=(0, 4))
        ttk.Spinbox(toolbar, from_=0.0, to=1.0, increment=0.05, textvariable=self.nms_var, width=6).grid(
            row=0, column=6, padx=(0, 12)
        )

        ttk.Label(toolbar, text="TopK").grid(row=0, column=7, padx=(0, 4))
        ttk.Spinbox(toolbar, from_=1, to=500, increment=10, textvariable=self.topk_var, width=6).grid(
            row=0, column=8, sticky="w"
        )

        ttk.Label(toolbar, textvariable=self.status_var, anchor="e").grid(row=0, column=9, sticky="e")

        content = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)

        image_frame = ttk.Frame(content)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(image_frame, bg="#111827", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._render_canvas())

        side_panel = ttk.Frame(content, width=310)
        side_panel.grid(row=0, column=1, sticky="ns")
        side_panel.grid_propagate(False)
        side_panel.columnconfigure(0, weight=1)

        ttk.Label(side_panel, text="检测概览", font=("Microsoft YaHei UI", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 6)
        )
        ttk.Label(side_panel, textvariable=self.summary_var, wraplength=280, justify="left").grid(
            row=1, column=0, sticky="ew", pady=(0, 12)
        )

        list_frame = ttk.Frame(side_panel)
        list_frame.grid(row=2, column=0, sticky="nsew")
        side_panel.rowconfigure(2, weight=1)

        self.result_list = tk.Listbox(list_frame, height=24)
        self.result_list.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.result_list.yview)
        scrollbar.pack(side="right", fill="y")
        self.result_list.configure(yscrollcommand=scrollbar.set)

    def _schedule_predictor_load(self) -> None:
        self.status_var.set("正在加载模型...")
        thread = threading.Thread(target=self._load_predictor_worker, daemon=True)
        thread.start()

    def _load_predictor_worker(self) -> None:
        try:
            settings = load_settings()
            predictor = TorchScriptPredictor(
                model_path=settings.torchscript_model_path,
                config_path=settings.config_path,
                device_name=settings.device,
                metadata_path=settings.metadata_path,
                use_fp16=settings.use_fp16,
            )
            self.root.after(0, self._on_predictor_loaded, predictor)
        except Exception as exc:
            self.root.after(0, self._on_predictor_error, exc)

    def _on_predictor_loaded(self, predictor: TorchScriptPredictor) -> None:
        self.predictor = predictor
        device = getattr(predictor, "device", "unknown")
        self.status_var.set(f"模型已加载: {device}")
        if self.original_image is not None:
            self.detect_button.configure(state="normal")

    def _on_predictor_error(self, exc: Exception) -> None:
        self.status_var.set("模型加载失败")
        messagebox.showerror("模型加载失败", str(exc))

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
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
        self.result_image = image.copy()
        self.last_result = None
        self.result_list.delete(0, tk.END)
        self.summary_var.set(f"图片尺寸: {image.width} x {image.height}\n等待检测")
        self.status_var.set(f"已打开: {self.image_path.name}")
        self.save_button.configure(state="disabled")
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

        self.detect_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.status_var.set("检测中...")
        image = self.original_image.copy()
        score = float(self.score_var.get())
        nms_iou = float(self.nms_var.get())
        top_k = int(self.topk_var.get())

        thread = threading.Thread(
            target=self._detect_worker,
            args=(image, score, nms_iou, top_k),
            daemon=True,
        )
        thread.start()

    def _detect_worker(self, image: Image.Image, score: float, nms_iou: float, top_k: int) -> None:
        try:
            assert self.predictor is not None
            result = self.predictor.predict(
                image=image,
                score_threshold=score,
                top_k=top_k,
                nms_iou_threshold=nms_iou,
            )
            self.root.after(0, self._on_detection_done, result)
        except Exception as exc:
            self.root.after(0, self._on_detection_error, exc)

    def _on_detection_done(self, result: dict) -> None:
        self.last_result = result
        self.result_image = self._draw_result(self.original_image.copy(), result.get("bboxes", []))
        self._update_result_list(result)
        self._render_canvas()
        self.open_button.configure(state="normal")
        self.detect_button.configure(state="normal")
        self.save_button.configure(state="normal")
        latency = result.get("latency_ms", {})
        total_ms = float(latency.get("total", 0.0))
        self.status_var.set(f"检测完成: {total_ms:.1f} ms")

    def _on_detection_error(self, exc: Exception) -> None:
        self.open_button.configure(state="normal")
        if self.original_image is not None and self.predictor is not None:
            self.detect_button.configure(state="normal")
        self.status_var.set("检测失败")
        messagebox.showerror("检测失败", str(exc))

    def _update_result_list(self, result: dict) -> None:
        bboxes = result.get("bboxes", [])
        latency = result.get("latency_ms", {})
        self.result_list.delete(0, tk.END)
        for index, box in enumerate(bboxes, start=1):
            name = box.get("class_name", str(box.get("class_id", "")))
            score = float(box.get("score", 0.0))
            self.result_list.insert(tk.END, f"{index:02d}. {name}  {score:.3f}")
        if not bboxes:
            self.result_list.insert(tk.END, "没有超过阈值的检测框")

        self.summary_var.set(
            "检测框数量: {count}\n图片尺寸: {width} x {height}\n总耗时: {total:.1f} ms\n推理耗时: {infer:.1f} ms".format(
                count=len(bboxes),
                width=result.get("image_width", "-"),
                height=result.get("image_height", "-"),
                total=float(latency.get("total", 0.0)),
                infer=float(latency.get("inference", 0.0)),
            )
        )

    def _draw_result(self, image: Image.Image, bboxes: list[dict]) -> Image.Image:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", max(14, image.width // 80))
        except OSError:
            font = ImageFont.load_default()

        line_width = max(2, image.width // 300)
        for box in bboxes:
            class_id = int(box.get("class_id", 0))
            color = BOX_COLORS[class_id % len(BOX_COLORS)]
            x1 = float(box["x1"])
            y1 = float(box["y1"])
            x2 = float(box["x2"])
            y2 = float(box["y2"])
            label = f"{box.get('class_name', class_id)} {float(box.get('score', 0.0)):.2f}"

            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            label_box = draw.textbbox((x1, y1), label, font=font)
            label_w = label_box[2] - label_box[0]
            label_h = label_box[3] - label_box[1]
            label_y = max(0, y1 - label_h - 6)
            draw.rectangle((x1, label_y, x1 + label_w + 8, label_y + label_h + 6), fill=color)
            draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)
        return image

    def _render_canvas(self) -> None:
        image = self.result_image
        if image is None:
            self.canvas.delete("all")
            width = max(1, self.canvas.winfo_width())
            height = max(1, self.canvas.winfo_height())
            self.canvas.create_text(
                width // 2,
                height // 2,
                text="打开图片后运行检测",
                fill="#d1d5db",
                font=("Microsoft YaHei UI", 16),
            )
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        scale = min(canvas_w / image.width, canvas_h / image.height)
        display_w = max(1, int(image.width * scale))
        display_h = max(1, int(image.height * scale))
        resized = image.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        x = (canvas_w - display_w) // 2
        y = (canvas_h - display_h) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

    def save_result(self) -> None:
        if self.result_image is None:
            return
        default_name = "yolo_detection_result.jpg"
        if self.image_path is not None:
            default_name = f"{self.image_path.stem}_detected.jpg"
        save_path = filedialog.asksaveasfilename(
            title="保存检测结果",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[
                ("JPEG image", "*.jpg"),
                ("PNG image", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not save_path:
            return
        try:
            self.result_image.save(save_path)
            self.status_var.set(f"已保存: {Path(save_path).name}")
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))


def main() -> None:
    os.environ.setdefault("YOLO_BACKEND_MODEL_FORMAT", "torchscript")
    os.environ.setdefault("YOLO_BACKEND_DEVICE", "auto")
    root = tk.Tk()
    app = YoloDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
