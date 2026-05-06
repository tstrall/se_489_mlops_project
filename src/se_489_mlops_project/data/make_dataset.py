"""Raw-to-processed data pipeline entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from se_489_mlops_project.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from se_489_mlops_project.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# SLA threshold: 7 days in seconds
SLA_THRESHOLD_SECONDS = 7 * 24 * 3600


def process_data(input_dir: Path, output_dir: Path) -> None:
    """Transform raw data in ``input_dir`` and write outputs to ``output_dir``."""
    logger.info("Reading raw data from %s", input_dir)

    # Load raw data
    issues_path = input_dir / "HelpDeskTickets" / "issues.csv"
    history_path = input_dir / "HelpDeskTickets" / "issues_change_history.csv"

    issues = pd.read_csv(issues_path)
    history = pd.read_csv(history_path)

    logger.info("Loaded %d issues and %d history records", len(issues), len(history))

    # Ensure timestamps
    history["created"] = pd.to_datetime(history["created"], errors="coerce")

    # Count events per ticket
    event_counts = history.groupby("issueid").size().reset_index(name="num_events")

    # Calculate ticket duration from history
    ticket_durations = (
        history.groupby("issueid")["created"].agg(["min", "max"]).reset_index()
    )
    ticket_durations["duration_seconds"] = (
        ticket_durations["max"] - ticket_durations["min"]
    ).dt.total_seconds()

    # Merge with issues data
    processed = issues.merge(event_counts, left_on="id", right_on="issueid", how="left")
    processed = processed.merge(
        ticket_durations[["issueid", "duration_seconds"]],
        left_on="id",
        right_on="issueid",
        how="left",
    )

    # Fill missing values
    processed["num_events"] = processed["num_events"].fillna(0)
    processed["duration_seconds"] = processed["duration_seconds"].fillna(0)

    # Create target: SLA violation
    processed["sla_violation"] = (
        processed["wf_total_time"] > SLA_THRESHOLD_SECONDS
    ).astype(int)

    # Select features for modeling
    feature_cols = [
        "id",
        "issue_num",
        "issue_contr_count",
        "issue_priority",
        "issue_type",
        "issue_comments_count",
        "wf_total_time",
        "processing_steps",
        "num_events",
        "duration_seconds",
    ]

    # Add workflow features (wf_* columns)
    wf_cols = [col for col in issues.columns if col.startswith("wf_")]
    feature_cols.extend(wf_cols)

    processed_data = processed[feature_cols + ["sla_violation"]].copy()

    # Save processed data
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed_data.csv"
    processed_data.to_csv(output_path, index=False)

    logger.info(
        "Saved processed data to %s with %d records",
        output_path,
        len(processed_data),
    )
    logger.info("Feature columns: %s", feature_cols)
    logger.info(
        "Target distribution: %s",
        processed_data["sla_violation"].value_counts().to_dict(),
    )


def main() -> None:
    """CLI entrypoint for data processing."""
    parser = argparse.ArgumentParser(description="Process raw data into model inputs")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--output", type=Path, default=PROCESSED_DATA_DIR)
    args = parser.parse_args()

    setup_logging()
    process_data(args.input, args.output)
    logger.info("Data processing complete")


if __name__ == "__main__":
    main()
