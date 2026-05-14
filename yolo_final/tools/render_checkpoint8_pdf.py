"""Render checkpoint 8 export report to PDF without LaTeX CJK dependencies."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "report"
OUTPUT_PDF = REPORT_DIR / "checkpoint8_export_report.pdf"
BENCHMARK_JSON = PROJECT_ROOT / "outputs/export_benchmark/checkpoint8_lr7e4_benchmark.json"
QUANT_BENCHMARK_JSON = PROJECT_ROOT / "outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_benchmark.json"
QUANT_COCO_JSON = PROJECT_ROOT / "outputs/evaluations/checkpoint8_lr7e4_int8_backbone_calib128_coco_eval.json"
EXPORT_META_JSON = PROJECT_ROOT / "exports/checkpoint8/best_yolofinal_416_lr7e4.export_metadata.json"
QUANT_META_JSON = PROJECT_ROOT / "exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.metadata.json"
VIS_DIR = PROJECT_ROOT / "outputs/export_benchmark/checkpoint8_lr7e4_vis16/torchscript"

SERIF_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
SANS_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

PAGE_W, PAGE_H = 1654, 2339
MARGIN_X = 145
MARGIN_TOP = 120
MARGIN_BOTTOM = 120
LINE_GAP = 12


class PdfPainter:
    def __init__(self):
        self.pages: list[Image.Image] = []
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP
        self.font_title = ImageFont.truetype(SERIF_FONT, 48, index=2)
        self.font_h1 = ImageFont.truetype(SERIF_FONT, 34, index=2)
        self.font_h2 = ImageFont.truetype(SERIF_FONT, 28, index=2)
        self.font_body = ImageFont.truetype(SERIF_FONT, 23, index=2)
        self.font_small = ImageFont.truetype(SERIF_FONT, 19, index=2)
        self.font_code = ImageFont.truetype(MONO_FONT, 19, index=2)
        self.font_table = ImageFont.truetype(SANS_FONT, 20, index=2)

    def _new_page(self) -> Image.Image:
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        self.pages.append(page)
        return page

    def new_page(self):
        self.page = self._new_page()
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_TOP

    def ensure(self, height: int):
        if self.y + height > PAGE_H - MARGIN_BOTTOM:
            self.new_page()

    def text_height(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = self.draw.multiline_textbbox((0, 0), text, font=font, spacing=LINE_GAP)
        return bbox[3] - bbox[1]

    def wrap(self, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
        lines: list[str] = []
        for para in text.splitlines():
            if not para.strip():
                lines.append("")
                continue
            current = ""
            for char in para:
                test = current + char
                if self.draw.textlength(test, font=font) <= width or not current:
                    current = test
                else:
                    lines.append(current)
                    current = char
            if current:
                lines.append(current)
        return lines

    def add_title(self, title: str, subtitle: str, date: str):
        for line in title.splitlines():
            w = self.draw.textlength(line, font=self.font_title)
            self.draw.text(((PAGE_W - w) / 2, self.y), line, font=self.font_title, fill="#111111")
            self.y += 66
        self.y += 18
        for line in [subtitle, date]:
            w = self.draw.textlength(line, font=self.font_h2)
            self.draw.text(((PAGE_W - w) / 2, self.y), line, font=self.font_h2, fill="#333333")
            self.y += 48
        self.y += 44

    def add_heading(self, text: str, level: int = 1):
        font = self.font_h1 if level == 1 else self.font_h2
        self.ensure(80)
        self.y += 18 if level == 1 else 8
        self.draw.text((MARGIN_X, self.y), text, font=font, fill="#111111")
        self.y += 54 if level == 1 else 44
        self.draw.line((MARGIN_X, self.y, PAGE_W - MARGIN_X, self.y), fill="#d0d0d0", width=2)
        self.y += 22

    def add_paragraph(self, text: str):
        width = PAGE_W - 2 * MARGIN_X
        lines = self.wrap(text, self.font_body, width)
        height = len(lines) * (self.font_body.size + LINE_GAP) + 16
        self.ensure(height)
        for line in lines:
            self.draw.text((MARGIN_X, self.y), line, font=self.font_body, fill="#222222")
            self.y += self.font_body.size + LINE_GAP
        self.y += 16

    def add_bullets(self, items: list[str]):
        width = PAGE_W - 2 * MARGIN_X - 34
        for item in items:
            lines = self.wrap(item, self.font_body, width)
            height = len(lines) * (self.font_body.size + LINE_GAP) + 12
            self.ensure(height)
            self.draw.text((MARGIN_X + 4, self.y), "•", font=self.font_body, fill="#222222")
            for index, line in enumerate(lines):
                self.draw.text((MARGIN_X + 34, self.y), line, font=self.font_body, fill="#222222")
                self.y += self.font_body.size + LINE_GAP
            self.y += 8
        self.y += 8

    def add_code(self, text: str):
        width = PAGE_W - 2 * MARGIN_X - 42
        lines = []
        for raw in text.strip("\n").splitlines():
            lines.extend(textwrap.wrap(raw, width=92, replace_whitespace=False) or [""])
        height = len(lines) * 28 + 36
        self.ensure(height)
        x0, y0 = MARGIN_X, self.y
        x1, y1 = PAGE_W - MARGIN_X, self.y + height
        self.draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill="#f6f6f6", outline="#d6d6d6")
        y = y0 + 18
        for line in lines:
            self.draw.text((x0 + 22, y), line, font=self.font_code, fill="#222222")
            y += 28
        self.y = y1 + 24

    def add_table(self, headers: list[str], rows: list[list[str]], col_widths: list[int]):
        row_h = 44
        height = row_h * (len(rows) + 1) + 8
        self.ensure(height)
        x = MARGIN_X
        y = self.y
        table_w = sum(col_widths)
        self.draw.rectangle((x, y, x + table_w, y + height - 8), outline="#999999", width=2)
        self.draw.rectangle((x, y, x + table_w, y + row_h), fill="#eeeeee")
        cx = x
        for header, width in zip(headers, col_widths):
            self.draw.text((cx + 10, y + 10), header, font=self.font_table, fill="#111111")
            cx += width
            self.draw.line((cx, y, cx, y + height - 8), fill="#bbbbbb", width=1)
        y += row_h
        for row in rows:
            cx = x
            for cell, width in zip(row, col_widths):
                self.draw.text((cx + 10, y + 10), cell, font=self.font_table, fill="#222222")
                cx += width
            self.draw.line((x, y, x + table_w, y), fill="#cccccc", width=1)
            y += row_h
        self.y += height + 22

    def add_image_grid(self, image_paths: list[Path], caption: str):
        self.new_page()
        self.add_heading("16 张推理可视化", level=1)
        cols = 4
        gap = 22
        cell_w = (PAGE_W - 2 * MARGIN_X - gap * (cols - 1)) // cols
        cell_h = 300
        x0 = MARGIN_X
        y0 = self.y
        for index, path in enumerate(image_paths):
            row, col = divmod(index, cols)
            x = x0 + col * (cell_w + gap)
            y = y0 + row * (cell_h + 64)
            with Image.open(path).convert("RGB") as img:
                img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
                bx = x + (cell_w - img.width) // 2
                by = y + (cell_h - img.height) // 2
                self.draw.rectangle((x, y, x + cell_w, y + cell_h), outline="#d0d0d0", width=1)
                self.page.paste(img, (bx, by))
            name = path.stem.replace("coco2017_val_", "")
            tw = self.draw.textlength(name, font=self.font_small)
            self.draw.text((x + (cell_w - tw) / 2, y + cell_h + 10), name, font=self.font_small, fill="#333333")
        self.y = y0 + 4 * (cell_h + 64) + 12
        self.add_paragraph(caption)

    def save(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first, rest = self.pages[0], self.pages[1:]
        first.save(output_path, "PDF", resolution=200.0, save_all=True, append_images=rest)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    benchmark = load_json(BENCHMARK_JSON)
    metadata = load_json(EXPORT_META_JSON)
    quant_benchmark = load_json(QUANT_BENCHMARK_JSON)
    quant_coco = load_json(QUANT_COCO_JSON)
    quant_metadata = load_json(QUANT_META_JSON)
    pytorch = benchmark["formats"]["pytorch"]["summary"]
    torchscript = benchmark["formats"]["torchscript"]["summary"]
    pytorch_cpu = quant_benchmark["formats"]["pytorch"]["summary"]
    int8_cpu = quant_benchmark["formats"]["torchscript"]["summary"]
    artifacts = metadata["artifacts"]
    quant_artifact = quant_metadata["artifact"]

    painter = PdfPainter()
    painter.add_title("YOLO Final 检查点 8 报告", "模型导出、推理服务与速度测试", "2026-05-13")

    painter.add_heading("任务目标")
    painter.add_paragraph("本检查点聚焦模型导出、推理服务与基础量化处理。本轮工作基于当前已经完成训练和正式评估的最佳策略。")
    painter.add_bullets(
        [
            "配置：configs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4.toml",
            "checkpoint：outputs/dual_scale_three_box_coco_only_noobj1_416_basic_aug_scale_jitter_50_lr7e4_ddp_20260512_130823/best.pth",
            "模型：deep_residual_dual_scale_three_box，416 输入，P4/P5 双尺度，三框检测头。",
            "训练口径：COCO-only，basic augmentation + scale jitter，50 epoch，学习率 0.0007。",
            "正式 COCO 指标：AP 0.192170，AP50 0.310452，AP75 0.200965，APS 0.084709，AR100 0.351947。",
        ]
    )

    painter.add_heading("现有前后端接入")
    painter.add_paragraph(
        "项目中已经存在可用的前后端。后端为 Flask REST API，前端通过 /model_predict 调用推理服务。"
        "因此，本轮没有重写服务框架，而是在后端 predictor 预留位置补充 PyTorch、TorchScript 与 ONNX 三种格式切换。"
    )
    painter.add_table(
        ["模块", "文件", "说明"],
        [
            ["后端", "backend/app.py", "Flask REST API"],
            ["前端", "frontend/app.js", "调用 /health 和 /model_predict"],
            ["共享运行时", "backend/exported_predictor_utils.py", "预处理、解码、NMS、响应格式"],
            ["TorchScript", "backend/torchscript_predictor.py", "加载导出 .pt"],
            ["ONNX", "backend/onnx_predictor.py", "加载 ONNX Runtime session"],
        ],
        [180, 520, 610],
    )

    painter.add_heading("导出实现")
    painter.add_paragraph(
        "新增 tools/export_model.py。模型原始输出为多尺度字典，导出时通过轻量 wrapper 转换为稳定 tuple 输出，"
        "输出顺序为 p4、p5。后处理保留在 Python 侧，保证 PyTorch、TorchScript 和 ONNX 对比时只替换模型前向部分。"
    )
    painter.add_code(
        """python tools/export_model.py \\
  --formats torchscript,onnx \\
  --output-dir exports/checkpoint8 \\
  --prefix best_yolofinal_416_lr7e4 \\
  --device cuda \\
  --batch-size 1"""
    )
    painter.add_table(
        ["产物", "路径/状态"],
        [
            ["TorchScript", "exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt"],
            ["大小", f"{artifacts['torchscript']['bytes'] / 1024 / 1024:.1f} MB"],
            ["metadata", "exports/checkpoint8/best_yolofinal_416_lr7e4.export_metadata.json"],
            ["ONNX", artifacts["onnx"]["reason"]],
        ],
        [260, 1020],
    )
    correctness = artifacts["torchscript"]["correctness"]
    painter.add_paragraph(
        "TorchScript 导出后与 PyTorch wrapper 的 raw tensor 输出做了数值检查："
        f"P4 max/mean abs diff = {correctness['output_0_max_abs_diff']:.6f}/"
        f"{correctness['output_0_mean_abs_diff']:.6f}；"
        f"P5 max/mean abs diff = {correctness['output_1_max_abs_diff']:.6f}/"
        f"{correctness['output_1_mean_abs_diff']:.6f}。"
    )

    painter.add_heading("速度测试")
    painter.add_paragraph(
        "新增 tools/benchmark_exported_model.py。测试使用 64 张 COCO val 图像，warmup 8 张，score threshold 0.5，"
        "top-k 10，NMS IoU threshold 0.5，且所有格式共用同一套 Python 后处理。"
    )
    painter.add_table(
        ["格式", "样本", "推理均值 ms", "总耗时均值 ms", "P95 总耗时", "端到端 FPS"],
        [
            [
                "PyTorch .pth",
                str(pytorch["samples"]),
                f"{pytorch['inference_ms_mean']:.3f}",
                f"{pytorch['total_ms_mean']:.3f}",
                f"{pytorch['total_ms_p95']:.3f}",
                f"{pytorch['fps_end_to_end']:.2f}",
            ],
            [
                "TorchScript",
                str(torchscript["samples"]),
                f"{torchscript['inference_ms_mean']:.3f}",
                f"{torchscript['total_ms_mean']:.3f}",
                f"{torchscript['total_ms_p95']:.3f}",
                f"{torchscript['fps_end_to_end']:.2f}",
            ],
        ],
        [220, 120, 220, 240, 210, 220],
    )
    infer_gain = (pytorch["inference_ms_mean"] - torchscript["inference_ms_mean"]) / pytorch["inference_ms_mean"] * 100.0
    total_gain = (pytorch["total_ms_mean"] - torchscript["total_ms_mean"]) / pytorch["total_ms_mean"] * 100.0
    painter.add_paragraph(
        f"TorchScript 相比原始 PyTorch checkpoint，模型前向平均耗时降低约 {infer_gain:.1f}%，"
        f"端到端平均耗时降低约 {total_gain:.1f}%，端到端 FPS 从 "
        f"{pytorch['fps_end_to_end']:.2f} 提升到 {torchscript['fps_end_to_end']:.2f}。"
    )
    painter.add_paragraph(
        "ONNX 后端代码已经接入，但当前 yolov1 环境缺少 onnxruntime；ONNX 导出也因为缺少 onnx 包被跳过。"
        "这属于环境依赖未安装，不是模型结构不支持。"
    )

    painter.add_heading("INT8 PTQ 量化")
    painter.add_paragraph(
        "本轮补充了 CPU 侧 FX graph mode PTQ 静态量化。直接全模型量化会让 decoupled head 中 reg/cls cat 的量化尺度不一致，"
        "因此正式产物采用更稳妥的策略：backbone INT8，detection head 保持 FP32。校准数据使用 128 张 COCO val 图片。"
    )
    painter.add_code(
        """python tools/quantize_model.py \\
  --calibration-samples 128 \\
  --output-dir exports/checkpoint8 \\
  --prefix best_yolofinal_416_lr7e4_int8_backbone_calib128 \\
  --backend x86"""
    )
    painter.add_table(
        ["产物", "数值"],
        [
            ["INT8 TorchScript", "exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt"],
            ["文件大小", f"{quant_artifact['bytes'] / 1024 / 1024:.1f} MB"],
            ["校准样本", f"{quant_metadata['calibration_samples_used']} 张 COCO val"],
            ["量化后端", quant_metadata["backend"]],
            ["保持 FP32", quant_metadata["float_module_regex"]],
        ],
        [260, 1020],
    )
    painter.add_table(
        ["CPU 格式", "样本", "推理均值 ms", "总耗时均值 ms", "P95 总耗时", "端到端 FPS"],
        [
            [
                "PyTorch .pth",
                str(pytorch_cpu["samples"]),
                f"{pytorch_cpu['inference_ms_mean']:.3f}",
                f"{pytorch_cpu['total_ms_mean']:.3f}",
                f"{pytorch_cpu['total_ms_p95']:.3f}",
                f"{pytorch_cpu['fps_end_to_end']:.2f}",
            ],
            [
                "INT8 TS",
                str(int8_cpu["samples"]),
                f"{int8_cpu['inference_ms_mean']:.3f}",
                f"{int8_cpu['total_ms_mean']:.3f}",
                f"{int8_cpu['total_ms_p95']:.3f}",
                f"{int8_cpu['fps_end_to_end']:.2f}",
            ],
        ],
        [220, 120, 220, 240, 210, 220],
    )
    int8_gain = (pytorch_cpu["total_ms_mean"] - int8_cpu["total_ms_mean"]) / pytorch_cpu["total_ms_mean"] * 100.0
    painter.add_paragraph(
        f"在 CPU 端，INT8 TorchScript 的端到端 FPS 为 {int8_cpu['fps_end_to_end']:.2f}，"
        f"PyTorch CPU baseline 为 {pytorch_cpu['fps_end_to_end']:.2f}；端到端平均耗时降低约 {int8_gain:.1f}%。"
        "该结果说明量化产物适合 CPU 推理服务演示。"
    )
    painter.add_table(
        ["模型", "COCO AP", "AP50", "AP75", "AR100"],
        [
            ["FP32 best", "0.192170", "0.310452", "0.200965", "0.351947"],
            [
                "INT8 backbone",
                f"{quant_coco['coco_ap']:.6f}",
                f"{quant_coco['coco_ap50']:.6f}",
                f"{quant_coco['coco_ap75']:.6f}",
                f"{quant_coco['coco_ar100']:.6f}",
            ],
        ],
        [260, 240, 240, 240, 240],
    )
    painter.add_paragraph(
        f"量化模型完整 COCO val 评估使用 {int(quant_coco['num_samples'])} 张图片和正式 score 设置。"
        f"相比 FP32 best，AP 下降约 {0.192170 - quant_coco['coco_ap']:.6f}，"
        "说明 backbone INT8 + head FP32 的 PTQ 策略在当前模型上精度损失很小。"
    )

    image_names = [
        "coco2017_val_000000397133.png",
        "coco2017_val_000000037777.png",
        "coco2017_val_000000252219.png",
        "coco2017_val_000000087038.png",
        "coco2017_val_000000174482.png",
        "coco2017_val_000000403385.png",
        "coco2017_val_000000006818.png",
        "coco2017_val_000000480985.png",
        "coco2017_val_000000458054.png",
        "coco2017_val_000000331352.png",
        "coco2017_val_000000296649.png",
        "coco2017_val_000000386912.png",
        "coco2017_val_000000502136.png",
        "coco2017_val_000000491497.png",
        "coco2017_val_000000184791.png",
        "coco2017_val_000000348881.png",
    ]
    painter.add_image_grid(
        [VIS_DIR / name for name in image_names],
        "以上图片由 TorchScript 后端生成，红框为预测框，标签包含类别名和置信度。"
        "展示参数与速度测试一致：score threshold 0.5，top-k 10，NMS IoU threshold 0.5。",
    )

    painter.add_heading("复现记录")
    painter.add_bullets(
        [
            "当前 Git commit：1c0c2c1。",
            "本轮新增或修改的核心文件：tools/export_model.py、tools/benchmark_exported_model.py、backend/exported_predictor_utils.py、backend/torchscript_predictor.py、backend/onnx_predictor.py、backend/pytorch_predictor.py、backend/app.py、backend/config.py、backend/README.md。",
            "导出产物：exports/checkpoint8/best_yolofinal_416_lr7e4.torchscript.pt。",
            "测速记录：outputs/export_benchmark/checkpoint8_lr7e4_benchmark.json。",
            "16 张可视化：outputs/export_benchmark/checkpoint8_lr7e4_vis16/torchscript/。",
            "INT8 量化产物：exports/checkpoint8/best_yolofinal_416_lr7e4_int8_backbone_calib128.torchscript.pt。",
            "INT8 速度记录：outputs/export_benchmark/checkpoint8_lr7e4_int8_backbone_calib128_cpu_benchmark.json。",
            "INT8 COCO eval：outputs/evaluations/checkpoint8_lr7e4_int8_backbone_calib128_coco_eval.json。",
            "当前工作区仍包含实验输出和报告文件的未提交状态；正式归档时应将代码、配置、导出脚本、测速 JSON、展示图和 PDF 一起保存。",
        ]
    )

    painter.add_heading("结论")
    painter.add_paragraph(
        "检查点 8 的核心目标已经完成：在已有前后端框架中接入导出模型推理路径，并完成 TorchScript 导出、速度测试和 16 张图片可视化。"
        "TorchScript 后端在不改变前端接口、不改变 Python 后处理逻辑的前提下，将端到端 FPS 从 "
        f"{pytorch['fps_end_to_end']:.2f} 提升到 {torchscript['fps_end_to_end']:.2f}，"
        "是当前可直接用于 GPU/常规导出演示的方案。补充的 INT8 PTQ 产物在 CPU 上将端到端 FPS 提升到 "
        f"{int8_cpu['fps_end_to_end']:.2f}，适合 CPU 推理服务展示。"
    )
    painter.save(OUTPUT_PDF)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
