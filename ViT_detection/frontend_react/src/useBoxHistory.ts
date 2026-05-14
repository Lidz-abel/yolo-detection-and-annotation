import { useCallback, useState } from "react";
import type { Box, HistoryState } from "./types";

const cloneBoxes = (boxes: Box[]) => boxes.map((box) => ({ ...box }));

export function useBoxHistory() {
  const [history, setHistory] = useState<HistoryState>({ past: [], present: [], future: [] });

  const replace = useCallback((boxes: Box[], checkpoint = true) => {
    setHistory((current) => {
      if (!checkpoint) {
        return { ...current, present: cloneBoxes(boxes) };
      }
      return {
        past: [...current.past, cloneBoxes(current.present)].slice(-80),
        present: cloneBoxes(boxes),
        future: []
      };
    });
  }, []);

  const reset = useCallback((boxes: Box[] = []) => {
    setHistory({ past: [], present: cloneBoxes(boxes), future: [] });
  }, []);

  const undo = useCallback(() => {
    setHistory((current) => {
      if (current.past.length === 0) return current;
      const previous = current.past[current.past.length - 1];
      return {
        past: current.past.slice(0, -1),
        present: cloneBoxes(previous),
        future: [cloneBoxes(current.present), ...current.future].slice(0, 80)
      };
    });
  }, []);

  const redo = useCallback(() => {
    setHistory((current) => {
      if (current.future.length === 0) return current;
      const next = current.future[0];
      return {
        past: [...current.past, cloneBoxes(current.present)].slice(-80),
        present: cloneBoxes(next),
        future: current.future.slice(1)
      };
    });
  }, []);

  return {
    boxes: history.present,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    replace,
    reset,
    undo,
    redo
  };
}
