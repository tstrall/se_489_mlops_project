"""Profile the training pipeline using cProfile.

Runs the full training pipeline under cProfile and writes two outputs:
  - reports/profiling/train_profile.prof   (binary, loadable with pstats)
  - reports/profiling/train_profile_summary.txt  (human-readable top-25 hot spots)

Usage
-----
From the repo root:

    python scripts/profile_training.py

You can then explore the binary profile interactively:

    python -m pstats reports/profiling/train_profile.prof
"""

from __future__ import annotations

import cProfile
import pstats
import sys
from io import StringIO
from pathlib import Path

# Make sure the src package is importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from se_489_mlops_project.config import PROCESSED_DATA_DIR  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports" / "profiling"
PROF_FILE = OUTPUT_DIR / "train_profile.prof"
SUMMARY_FILE = OUTPUT_DIR / "train_profile_summary.txt"


def _run_training() -> None:
    """Thin wrapper that calls the training function directly (no Hydra overhead)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from se_489_mlops_project.models.model import Model

    data_file = PROCESSED_DATA_DIR / "processed_data.csv"
    df = pd.read_csv(data_file)

    leaky_cols = [
        "id",
        "sla_violation",
        "wf_total_time",
        "total_time_days",
        "log_total_time",
    ]
    x = df.drop(columns=[c for c in leaky_cols if c in df.columns])
    y = df["sla_violation"]

    categorical_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=categorical_cols, drop_first=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Model()
    model.fit(x_train, y_train)
    model.predict(x_test)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Profiling training pipeline... output -> {PROF_FILE}")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_training()
    profiler.disable()

    # Save binary profile
    profiler.dump_stats(str(PROF_FILE))
    print(f"Binary profile saved to {PROF_FILE}")

    # Save human-readable summary (top 25 by cumulative time)
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(25)
    summary = stream.getvalue()

    SUMMARY_FILE.write_text(summary)
    print(f"Summary saved to {SUMMARY_FILE}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
