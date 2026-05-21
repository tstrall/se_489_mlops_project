"""Monitor CPU and memory usage during a training run.

Spawns the training pipeline in a subprocess and polls ``psutil`` every
``POLL_INTERVAL`` seconds.  Writes a CSV to
``reports/monitoring/training_monitor.csv`` and prints a summary on exit.

Usage
-----
From the repo root::

    python scripts/monitor_training.py

The CSV columns are:

    elapsed_s, cpu_percent, rss_mb, vms_mb

where ``rss_mb`` is the resident set size (physical RAM) and ``vms_mb`` is
the virtual memory size of the training process.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POLL_INTERVAL = 0.5  # seconds between samples
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "reports" / "monitoring"
CSV_FILE = OUTPUT_DIR / "training_monitor.csv"
FIELDNAMES = ["elapsed_s", "cpu_percent", "rss_mb", "vms_mb"]


def _bytes_to_mb(n: int) -> float:
    return round(n / (1024 * 1024), 2)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the training command — same invocation used in Makefile / Docker.
    cmd = [
        sys.executable,
        "-m",
        "se_489_mlops_project.train_model",
    ]

    print(f"Starting training process: {' '.join(cmd)}")
    print(f"Monitoring at {POLL_INTERVAL}s intervals -> {CSV_FILE}\n")

    start = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    ps_proc = psutil.Process(proc.pid)

    rows: list[dict[str, float]] = []

    try:
        while proc.poll() is None:
            try:
                mem = ps_proc.memory_info()
                cpu = ps_proc.cpu_percent(interval=None)
                elapsed = round(time.monotonic() - start, 2)
                rows.append(
                    {
                        "elapsed_s": elapsed,
                        "cpu_percent": cpu,
                        "rss_mb": _bytes_to_mb(mem.rss),
                        "vms_mb": _bytes_to_mb(mem.vms),
                    }
                )
            except psutil.NoSuchProcess:
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        proc.terminate()
        print("\nMonitoring interrupted — writing partial results.")

    # Flush remaining stdout from training process
    if proc.stdout:
        for line in proc.stdout:
            print(line, end="")

    returncode = proc.wait()

    # Write CSV
    with CSV_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    if rows:
        peak_rss = max(r["rss_mb"] for r in rows)
        peak_cpu = max(r["cpu_percent"] for r in rows)
        avg_cpu = round(sum(r["cpu_percent"] for r in rows) / len(rows), 1)
        total_elapsed = rows[-1]["elapsed_s"]

        print("\n--- Training Monitor Summary ---")
        print(f"  Total time   : {total_elapsed}s")
        print(f"  Peak RAM     : {peak_rss} MB")
        print(f"  Peak CPU     : {peak_cpu}%")
        print(f"  Avg CPU      : {avg_cpu}%")
        print(f"  Samples      : {len(rows)}")
        print(f"  CSV saved to : {CSV_FILE}")

    sys.exit(returncode)


if __name__ == "__main__":
    main()
