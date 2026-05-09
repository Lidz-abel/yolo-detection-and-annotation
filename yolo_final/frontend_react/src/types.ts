export type Box = {
  id: string;
  class_id: number;
  class_name?: string;
  score: number | null;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type PredictResponse = {
  success: boolean;
  error?: string;
  image_id: string;
  image_width: number;
  image_height: number;
  bboxes: Array<Omit<Box, "id">>;
  latency_ms?: {
    preprocess: number;
    inference: number;
    postprocess: number;
    total: number;
  };
  model?: {
    format: string;
    config: string;
    checkpoint: string;
    device: string;
    fp16?: boolean;
  };
};

export type AnnotateResponse = {
  success: boolean;
  error?: string;
  saved_path?: string;
  saved_image_path?: string;
  num_boxes?: number;
};

export type HistoryState = {
  past: Box[][];
  present: Box[];
  future: Box[][];
};

export type DragMode = "move" | "resize";

export type Interaction = {
  id: string;
  mode: DragMode;
  handle: "nw" | "ne" | "sw" | "se" | "";
  startPoint: { x: number; y: number };
  startBox: Box;
};
