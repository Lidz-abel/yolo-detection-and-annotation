# YOLO React Annotation Workbench

React/Vite frontend for checkpoint 9.3. It keeps the original Flask API contract:

- `POST /model_predict`
- `POST /human_annotate`

The older static frontend remains available at `/`; this React version is an upgrade path with undo/redo state rollback.

## Development

Start the Flask backend from `yolo_final/`:

```bash
/home/lidz/miniconda3/envs/yolov1/bin/python backend/app.py
```

Start the React dev server from `yolo_final/frontend_react/`:

```bash
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

Vite proxies `/health`, `/model_predict`, and `/human_annotate` to Flask.

## Production Build

```bash
npm run build
```

Then Flask can serve the built React page at:

```text
http://127.0.0.1:5000/react
```

## Rollback

- Interface rollback: use the existing static version at `http://127.0.0.1:5000/`.
- Annotation rollback: React keeps a bbox history stack and supports undo/redo from the toolbar and keyboard shortcuts.
