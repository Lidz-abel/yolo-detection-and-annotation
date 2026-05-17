from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "final_loss_formula.pdf"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

fp = FontProperties(fname=FONT)
fp_bold = FontProperties(fname=BOLD_FONT)


def wrap_text(text, width=64):
    lines = []
    for raw in text.splitlines():
        if not raw:
            lines.append("")
            continue
        if raw.startswith("    ") or raw.startswith("  "):
            lines.append(raw)
            continue
        lines.extend(textwrap.wrap(raw, width=width, break_long_words=False, replace_whitespace=False))
    return lines


def new_page(pdf, title, page_no):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.08, 0.955, title, fontproperties=fp_bold, fontsize=16, va="top")
    ax.plot([0.08, 0.92], [0.925, 0.925], color="#444444", linewidth=0.8)
    ax.text(0.5, 0.035, f"{page_no}", fontproperties=fp, fontsize=9, ha="center", color="#666666")
    return fig, ax, 0.89


def draw_lines(pdf, title, sections):
    page_no = 1
    fig, ax, y = new_page(pdf, title, page_no)
    line_h = 0.024
    small_h = 0.020
    for kind, text in sections:
        if kind == "h1":
            if y < 0.14:
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1
                fig, ax, y = new_page(pdf, title, page_no)
            ax.text(0.08, y, text, fontproperties=fp_bold, fontsize=13, va="top", color="#1f2937")
            y -= 0.040
        elif kind == "h2":
            if y < 0.12:
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1
                fig, ax, y = new_page(pdf, title, page_no)
            ax.text(0.08, y, text, fontproperties=fp_bold, fontsize=11.5, va="top", color="#111827")
            y -= 0.032
        elif kind == "code":
            lines = wrap_text(text, width=76)
            box_h = max(0.04, small_h * len(lines) + 0.018)
            if y - box_h < 0.07:
                pdf.savefig(fig)
                plt.close(fig)
                page_no += 1
                fig, ax, y = new_page(pdf, title, page_no)
            rect = plt.Rectangle((0.075, y - box_h + 0.006), 0.85, box_h, facecolor="#f3f4f6", edgecolor="#d1d5db")
            ax.add_patch(rect)
            yy = y - 0.010
            for line in lines:
                ax.text(0.09, yy, line, fontproperties=fp, fontsize=8.8, va="top", color="#111827")
                yy -= small_h
            y -= box_h + 0.018
        else:
            for line in wrap_text(text, width=70):
                if y < 0.075:
                    pdf.savefig(fig)
                    plt.close(fig)
                    page_no += 1
                    fig, ax, y = new_page(pdf, title, page_no)
                ax.text(0.08, y, line, fontproperties=fp, fontsize=10.2, va="top", color="#111827")
                y -= line_h
            y -= 0.008
    pdf.savefig(fig)
    plt.close(fig)


sections = [
    ("h1", "1. 最终 Loss 完整数学公式"),
    ("p", "我们最终使用的是双尺度 YOLO-style detection loss。模型输出 p4 和 p5 两个尺度，每个尺度每个 grid cell 有 3 个 anchors，每个 anchor 输出 4 个 bbox 参数、1 个 objectness logit 和 80 个 class logits。"),
    ("code", "pred_p4: [B, 26, 26, 255]\npred_p5: [B, 13, 13, 255]\n255 = 3 * (4 bbox + 1 objectness + 80 classes)"),
    ("p", "最终总 loss 是两个尺度 loss 的和："),
    ("code", "L_total = L_p4 + L_p5"),
    ("p", "单个尺度的 loss 为："),
    ("code", "L_scale = lambda_box * L_box + lambda_obj * L_obj + lambda_cls * L_cls"),
    ("p", "最终超参数选择为：lambda_box=5.0，lambda_obj=1.0，lambda_cls=1.0，lambda_noobj=1.0。因此："),
    ("code", "L_scale = 5.0 * L_box + L_obj + L_cls\nL_obj = L_obj_pos + 1.0 * L_obj_neg"),
    ("p", "展开到双尺度后，最终训练目标是："),
    ("code", "L_total = [5 * L_box_p4 + L_obj_p4 + L_cls_p4]\n        + [5 * L_box_p5 + L_obj_p5 + L_cls_p5]"),
    ("h2", "Dynamic Assignment"),
    ("p", "在计算 loss 前，需要先决定哪些 prediction slot 是正样本。一个 slot 表示 (grid_y, grid_x, anchor_index)。最终使用 dynamic_cost，而不是固定静态分配。"),
    ("code", "cost = 3.0 * (1 - IoU(pred_box, gt_box)) + 1.0 * BCE(pred_cls, gt_class)"),
    ("p", "候选区域由 dynamic_center_radius=1 控制，即 GT 中心 grid 周围 3x3 区域。每个 cell 有 3 个 anchors，因此每个尺度每个 GT 最多先考虑 3*3*3=27 个候选 slot。"),
    ("p", "每个 GT 按 cost 从小到大排序，最多选择 dynamic_topk=2 个正样本。如果某个 slot 没有被选为正样本，但和任意 GT 的 IoU >= dynamic_ignore_iou=0.5，则它不会被当作普通背景参与 no-object loss。"),
    ("h2", "Box Loss"),
    ("p", "记动态分配后的正样本集合为 P。对于正样本 i，预测框为 pred_box_i，GT 框为 gt_box_i。box loss 使用 GIoU："),
    ("code", "L_box = (1 / |P|) * sum_{i in P} [1 - GIoU(pred_box_i, gt_box_i)]"),
    ("p", "当预测框越接近 GT 框时，GIoU 越接近 1，因此 1-GIoU 越接近 0，box loss 越小。"),
    ("h2", "Objectness Loss"),
    ("p", "最终 objectness 正样本 target 不是固定 1，而是当前预测框与 GT 的 IoU，并设置下限 0.05："),
    ("code", "q_i = clamp(IoU(pred_box_i, gt_box_i), min=0.05, max=1.0)\nL_obj_pos = (1 / |P|) * sum_{i in P} BCEWithLogits(obj_logit_i, q_i)\nL_obj_neg = (1 / |N|) * sum_{j in N} BCEWithLogits(obj_logit_j, 0)\nL_obj = L_obj_pos + 1.0 * L_obj_neg"),
    ("p", "这样 objectness 不只表示有没有物体，也和框质量绑定。框质量越高，objectness target 越高。"),
    ("h2", "Classification Loss"),
    ("p", "最终分类使用 quality BCE。BCE 公式没有变化，但 target 从普通 one-hot 改成 IoU quality target。若正样本 i 的 GT 类别为 c_i："),
    ("code", "t_{i,k} = IoU(pred_box_i, gt_box_i), if k = c_i\n         = 0, otherwise\n\nL_cls = (1 / (|P| * C)) * sum_{i in P} sum_{k=1}^{C} BCEWithLogits(cls_logit_{i,k}, t_{i,k})\nC = 80"),
    ("p", "这意味着分类置信度也会受到定位质量约束。预测框质量低时，即使类别方向正确，分类 target 也不会是满分 1。"),
    ("h1", "2. 一个具体 pred 向量的 Loss 计算例子"),
    ("p", "考虑一个正样本 anchor 的预测向量："),
    ("code", "pred = [bbox_raw_1, bbox_raw_2, bbox_raw_3, bbox_raw_4, obj_logit,\n        cls_logit_1, ..., cls_logit_80]\nlength = 85 = 4 + 1 + 80"),
    ("p", "假设 bbox decode 后："),
    ("code", "pred_box = [0.40, 0.40, 0.70, 0.70]\ngt_box   = [0.45, 0.45, 0.75, 0.75]\ngt_class = dog\nobj_logit = 1.2\ndog class logit = 2.0\nother 79 class logits = -2.0"),
    ("h2", "IoU 计算"),
    ("p", "交集框为 [0.45, 0.45, 0.70, 0.70]，交集面积为 0.25*0.25=0.0625。预测框面积为 0.30*0.30=0.09，GT 框面积同样为 0.09。"),
    ("code", "union_area = 0.09 + 0.09 - 0.0625 = 0.1175\nIoU = 0.0625 / 0.1175 ≈ 0.5319"),
    ("h2", "GIoU 与 Box Loss"),
    ("p", "最小外接框为 [0.40,0.40,0.75,0.75]，面积为 0.35*0.35=0.1225。"),
    ("code", "GIoU = IoU - (C_area - union_area) / C_area\n     = 0.5319 - (0.1225 - 0.1175) / 0.1225\n     ≈ 0.4911\n\nL_box = 1 - GIoU = 1 - 0.4911 = 0.5089"),
    ("h2", "Objectness Loss"),
    ("p", "正样本 objectness target 为 IoU soft target："),
    ("code", "q = clamp(0.5319, 0.05, 1.0) = 0.5319\nL_obj_pos = BCEWithLogits(1.2, 0.5319) ≈ 0.8250"),
    ("p", "假设有一个负样本 slot，objectness logit 为 -1.0，target 为 0："),
    ("code", "L_obj_neg = BCEWithLogits(-1.0, 0) ≈ 0.3133\nL_obj = 0.8250 + 1.0 * 0.3133 = 1.1383"),
    ("h2", "Classification Loss"),
    ("p", "dog 类 target 为 IoU=0.5319，其他 79 类 target 为 0。"),
    ("code", "BCEWithLogits(2.0, 0.5319) ≈ 1.0631\nBCEWithLogits(-2.0, 0) ≈ 0.1269\n79 * 0.1269 ≈ 10.0251\nL_cls = (1.0631 + 10.0251) / 80 ≈ 0.1386"),
    ("h2", "单尺度总 Loss"),
    ("code", "L_scale = 5 * L_box + L_obj + L_cls\n        = 5 * 0.5089 + 1.1383 + 0.1386\n        ≈ 3.8214"),
    ("p", "这个例子说明：box loss 权重最大；objectness 同时考虑正负样本；classification 的正确类别 target 是 IoU，而不是固定 1。"),
    ("h1", "3. IoU、GIoU、BCE 分别是什么"),
    ("h2", "IoU"),
    ("p", "IoU 是 Intersection over Union，即交并比。它衡量预测框和真实框的重叠程度。"),
    ("code", "IoU(A, B) = area(A ∩ B) / area(A ∪ B)"),
    ("p", "IoU 取值在 0 到 1 之间。完全重合时 IoU=1，完全不重合时 IoU=0。"),
    ("h2", "GIoU"),
    ("p", "GIoU 是 Generalized IoU。它在 IoU 基础上加入最小外接框惩罚。"),
    ("code", "GIoU(A, B) = IoU(A, B) - (area(C) - area(A ∪ B)) / area(C)"),
    ("p", "其中 C 是同时包住 A 和 B 的最小外接框。GIoU 的优势是：即使两个框不重叠，仍然可以通过外接框惩罚反映两个框之间的距离关系。我们的 box loss 使用 L_box = 1 - GIoU。"),
    ("h2", "BCE"),
    ("p", "BCE 是 Binary Cross Entropy，即二分类交叉熵。若预测概率为 p，target 为 y："),
    ("code", "BCE(p, y) = - [ y * log(p) + (1 - y) * log(1 - p) ]"),
    ("p", "代码中使用 BCEWithLogitsLoss，输入是 logit z，而不是概率。它内部会先做 sigmoid："),
    ("code", "p = sigmoid(z) = 1 / (1 + exp(-z))\nBCEWithLogits(z, y) = BCE(sigmoid(z), y)"),
    ("p", "我们在 objectness 和 classification 两处使用 BCE。最终 target 是 quality target：objectness target 等于 IoU，classification 正类 target 也等于 IoU。"),
    ("h1", "总结"),
    ("p", "最终 loss 的核心设计是：用 dynamic assignment 根据当前预测质量选择正样本；用 GIoU 监督框定位；用 IoU soft target 监督 objectness；用 quality BCE 监督 classification；最后把 p4 和 p5 双尺度 loss 相加。"),
    ("code", "L_total = L_p4 + L_p5\nL_scale = 5 * L_box + L_obj + L_cls"),
]


with PdfPages(OUT) as pdf:
    draw_lines(pdf, "最终 YOLO Loss 设计：公式、例子与基础概念", sections)

print(OUT)
