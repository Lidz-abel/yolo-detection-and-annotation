import type { Box } from "./types";

export const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(value, max));

export function normalizeBox(box: Box, imageWidth: number, imageHeight: number): Box {
  const rawX1 = Number.isFinite(box.x1) ? box.x1 : 0;
  const rawY1 = Number.isFinite(box.y1) ? box.y1 : 0;
  const rawX2 = Number.isFinite(box.x2) ? box.x2 : rawX1 + 1;
  const rawY2 = Number.isFinite(box.y2) ? box.y2 : rawY1 + 1;
  const x1 = clamp(Math.min(rawX1, rawX2), 0, imageWidth);
  const y1 = clamp(Math.min(rawY1, rawY2), 0, imageHeight);
  const x2 = clamp(Math.max(rawX1, rawX2), 0, imageWidth);
  const y2 = clamp(Math.max(rawY1, rawY2), 0, imageHeight);

  return {
    ...box,
    class_id: Math.max(0, Math.trunc(box.class_id || 0)),
    x1,
    y1,
    x2: Math.max(x1 + 1, x2),
    y2: Math.max(y1 + 1, y2)
  };
}

export function createManualBox(imageWidth: number, imageHeight: number): Box {
  const size = Math.max(32, Math.min(imageWidth, imageHeight) * 0.18);
  return normalizeBox(
    {
      id: crypto.randomUUID(),
      class_id: 0,
      class_name: "manual",
      score: null,
      x1: imageWidth * 0.5 - size * 0.5,
      y1: imageHeight * 0.5 - size * 0.5,
      x2: imageWidth * 0.5 + size * 0.5,
      y2: imageHeight * 0.5 + size * 0.5
    },
    imageWidth,
    imageHeight
  );
}

export function toYoloTxt(boxes: Box[], imageWidth: number, imageHeight: number): string {
  return boxes
    .map((box) => {
      const width = box.x2 - box.x1;
      const height = box.y2 - box.y1;
      const cx = box.x1 + width * 0.5;
      const cy = box.y1 + height * 0.5;
      return [
        box.class_id,
        cx / imageWidth,
        cy / imageHeight,
        width / imageWidth,
        height / imageHeight
      ]
        .map((value, index) => (index === 0 ? String(value) : Number(value).toFixed(6)))
        .join(" ");
    })
    .join("\n")
    .concat(boxes.length ? "\n" : "");
}

export function downloadTxt(filename: string, content: string): void {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}
