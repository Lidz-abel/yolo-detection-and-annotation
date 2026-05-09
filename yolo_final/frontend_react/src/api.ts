import type { AnnotateResponse, Box, PredictResponse } from "./types";

const trimBase = (baseUrl: string) => baseUrl.replace(/\/$/, "");

export async function health(baseUrl: string): Promise<boolean> {
  const response = await fetch(`${trimBase(baseUrl)}/health`);
  if (!response.ok) return false;
  const payload = await response.json();
  return Boolean(payload.success);
}

export async function predictImage(options: {
  baseUrl: string;
  file: File;
  scoreThreshold: number;
  topK: number;
  nmsIouThreshold: number;
}): Promise<PredictResponse> {
  const form = new FormData();
  form.append("image", options.file);
  form.append("score_threshold", String(options.scoreThreshold));
  form.append("top_k", String(options.topK));
  form.append("nms_iou_threshold", String(options.nmsIouThreshold));

  const response = await fetch(`${trimBase(options.baseUrl)}/model_predict`, {
    method: "POST",
    body: form
  });
  const payload = (await response.json()) as PredictResponse;
  if (!response.ok || !payload.success) {
    throw new Error(payload.error || `Predict failed with HTTP ${response.status}`);
  }
  return payload;
}

export async function saveAnnotation(options: {
  baseUrl: string;
  file?: File;
  imageId: string;
  imageWidth: number;
  imageHeight: number;
  boxes: Box[];
}): Promise<AnnotateResponse> {
  const annotation = {
    image_id: options.imageId,
    image_width: options.imageWidth,
    image_height: options.imageHeight,
    bboxes: options.boxes.map(({ class_id, x1, y1, x2, y2 }) => ({ class_id, x1, y1, x2, y2 }))
  };
  const requestInit: RequestInit = options.file
    ? (() => {
        const form = new FormData();
        form.append("image", options.file as File);
        form.append("annotation", JSON.stringify(annotation));
        return { method: "POST", body: form };
      })()
    : {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(annotation)
      };

  const response = await fetch(`${trimBase(options.baseUrl)}/human_annotate`, {
    ...requestInit
  });
  const payload = (await response.json()) as AnnotateResponse;
  if (!response.ok || !payload.success) {
    throw new Error(payload.error || `Save failed with HTTP ${response.status}`);
  }
  return payload;
}
