"""Experiment bookkeeping helpers for reproducible yolov0 runs."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def get_git_commit(project_root: Path) -> str:
    """Read the current git commit so each run stays traceable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def init_run(project_root: Path, config_path: Path, config: dict) -> dict:
    """Create run directories and initialize the standard record files."""
    logging_cfg = config["logging"]
    run_name = str(logging_cfg["run_name"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"

    runs_dir = Path(logging_cfg["runs_dir"])
    output_root = Path(logging_cfg["output_dir"])
    records_root = Path(logging_cfg["records_dir"])

    run_tb_dir = runs_dir / run_id
    run_output_dir = output_root / run_id
    run_record_dir = records_root / run_id

    for path in (run_tb_dir, run_output_dir, run_record_dir):
        path.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_record_dir / "config.toml"
    metadata_path = run_record_dir / "metadata.json"
    result_path = run_record_dir / "result.txt"

    shutil.copy2(config_path, config_snapshot)

    metadata = {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "project_root": str(project_root),
        "config_path": str(config_path),
        "config_snapshot": str(config_snapshot),
        "tensorboard_dir": str(run_tb_dir),
        "output_dir": str(run_output_dir),
        "record_dir": str(run_record_dir),
        "git_commit": get_git_commit(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "username": os.environ.get("USER", "unknown"),
        "status": "initialized",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "run_name": run_name,
        "timestamp": timestamp,
        "tensorboard_dir": run_tb_dir,
        "output_dir": run_output_dir,
        "record_dir": run_record_dir,
        "config_snapshot": config_snapshot,
        "metadata_path": metadata_path,
        "result_path": result_path,
        "metadata": metadata,
    }


def update_metadata(metadata_path: Path, **kwargs) -> dict:
    """Update metadata.json in place as the run progresses."""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(kwargs)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def write_result_summary(result_path: Path, lines: list[str]) -> None:
    """Write a plain-text run summary for quick inspection."""
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
