"""Tests for the raw-to-processed data pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from se_489_mlops_project.data.make_dataset import (
    SLA_THRESHOLD_SECONDS,
    process_data,
)


def test_process_data_writes_processed_dataset(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    helpdesk_dir = raw_dir / "HelpDeskTickets"
    helpdesk_dir.mkdir(parents=True)

    pd.DataFrame(
        {
            "id": [1, 2],
            "issue_num": [101, 102],
            "issue_contr_count": [2, 0],
            "issue_priority": ["High", "Low"],
            "issue_type": ["Ticket", "Bug"],
            "issue_comments_count": [6, 3],
            "wf_total_time": [SLA_THRESHOLD_SECONDS + 1, 3_600],
            "processing_steps": [4, 1],
            "wf_open": [60, 30],
            "wf_closed": [30, 10],
        }
    ).to_csv(helpdesk_dir / "issues.csv", index=False)

    pd.DataFrame(
        {
            "issueid": [1, 1, 2],
            "created": [
                "2024-01-01T00:00:00",
                "2024-01-02T00:00:00",
                "2024-01-03T00:00:00",
            ],
        }
    ).to_csv(helpdesk_dir / "issues_change_history.csv", index=False)

    process_data(raw_dir, output_dir)

    result = pd.read_csv(output_dir / "processed_data.csv")

    assert len(result) == 2
    assert result["sla_violation"].tolist() == [1, 0]
    assert result["num_events"].tolist() == [2, 1]
    assert result["duration_seconds"].tolist() == [86_400.0, 0.0]
    assert result["events_per_day"].tolist() == [2.0, 1.0]
    assert result["comments_per_contributor"].tolist() == [3.0, 3.0]
    assert result["is_high_priority"].tolist() == [1, 0]
