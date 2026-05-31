"""Profile the scikit-learn training path with Scalene.

Scalene complements ``scripts/profile_training.py`` by reporting CPU time and
memory behavior at line granularity, which is the recommended profiler path for
this project's scikit-learn workload.

Usage
-----
From the repo root:

    python scripts/profile_with_scalene.py

Output:

    reports/profiling/scalene_training_profile.txt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "profiling"
SUMMARY_FILE = OUTPUT_DIR / "scalene_training_profile.txt"


def main() -> None:
    """Run Scalene against the existing training profiling workload."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scalene",
        "--cli",
        "--outfile",
        str(SUMMARY_FILE),
        str(PROJECT_ROOT / "scripts" / "profile_training.py"),
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    print(f"Scalene profile saved to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
