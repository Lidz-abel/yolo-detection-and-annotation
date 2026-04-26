#!/usr/bin/env python3
"""Wait for GPUs to free up, then run and finalize Stage D Round 1."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/home/lidz/YOLO/yolov0")
PYTHON = "/home/lidz/anaconda3/envs/yolov1/bin/python"
CONFIG_NAME = "deep_residual_multiscale_singlebox.toml"
EXPERIMENT_NAME = "deep_residual_multiscale_singlebox"
TRAIN_LOG = ROOT / "logs" / "deep_residual_multiscale_singlebox_tmux.log"
AUTOPILOT_LOG = ROOT / "logs" / "stage_d_round1_autopilot.log"


def log(message: str) -> None:
    """Append one timestamp-free line to the autopilot log and stdout."""
    line = message.rstrip()
    print(line, flush=True)
    AUTOPILOT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUTOPILOT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def query_free_memory() -> list[int]:
    """Return free memory in MiB for all visible GPUs."""
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return [int(line.strip()) for line in output.splitlines() if line.strip()]


def wait_for_free_gpus(min_free_mib: int = 20000, poll_seconds: int = 120) -> None:
    """Block until every GPU has enough free memory for the Stage D run."""
    while True:
        free_memory = query_free_memory()
        if free_memory and min(free_memory) >= min_free_mib:
            log(f"GPU gate open: free memory per GPU = {free_memory}")
            return
        log(f"GPU gate blocked: free memory per GPU = {free_memory}, waiting...")
        time.sleep(poll_seconds)


def run_training() -> str:
    """Launch train.py, mirror logs, and return the discovered run id."""
    TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        PYTHON,
        "-u",
        "tools/train.py",
        "--config",
        f"configs/{CONFIG_NAME}",
    ]
    run_id = ""
    with TRAIN_LOG.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            if not run_id:
                match = re.search(r"run id:\s*(\S+)", line)
                if match:
                    run_id = match.group(1)
                    log(f"Captured run id: {run_id}")
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage D train.py failed with exit code {return_code}")
    if not run_id:
        raise RuntimeError("Stage D train.py finished without emitting a run id")
    return run_id


def finalize(run_id: str) -> None:
    """Run the generic Stage D finalize helper for the finished run."""
    subprocess.run(
        [
            PYTHON,
            "-u",
            "tools/finalize_stage_d_round1.py",
            "--run-id",
            run_id,
            "--config-name",
            CONFIG_NAME,
            "--experiment-name",
            EXPERIMENT_NAME,
        ],
        cwd=str(ROOT),
        check=True,
    )


def main() -> None:
    """Wait, train, evaluate, summarize, and commit Stage D Round 1."""
    log("Stage D Round 1 autopilot waiting for GPUs.")
    wait_for_free_gpus()
    run_id = run_training()
    finalize(run_id)
    log(f"Stage D Round 1 autopilot completed for {run_id}.")


if __name__ == "__main__":
    main()
