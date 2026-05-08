"""Feature engineering transformations.

All transformations are deterministic and side-effect free so this module
can be called identically at training time and inference time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from se_489_mlops_project.logging_config import get_logger

logger = get_logger(__name__)

# Seconds per day — used for human-readable time-based ratios
_SECONDS_PER_DAY = 86_400.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from a processed dataframe.

    Engineered features
    -------------------
    events_per_day
        Ticket event density: num_events / duration in days.  Tickets that
        accumulate many interactions quickly tend to be harder to resolve.
    comments_per_contributor
        Average message load per contributor.  High values can indicate a
        single person is carrying the ticket, which correlates with delays.
    total_time_days
        wf_total_time converted to days for interpretability.
    is_high_priority
        Binary flag: issue_priority == "High".  Priority is one of the
        strongest single predictors of SLA outcome.
    log_total_time
        log1p of wf_total_time reduces the heavy right-skew of resolution
        durations and improves linear model performance.
    log_num_events
        log1p of num_events for the same skew-reduction reason.

    Args:
        df: DataFrame produced by ``make_dataset.process_data``.  Must
            contain the columns listed in ``make_dataset.feature_cols``.

    Returns:
        A new DataFrame with all original columns plus the engineered ones.
        The input DataFrame is never modified in place.
    """
    logger.info("Building features for %d rows", len(df))
    out = df.copy()

    # -- time-based features --------------------------------------------------
    duration_days = out["duration_seconds"] / _SECONDS_PER_DAY
    # avoid division by zero for tickets with only one event (duration == 0)
    out["events_per_day"] = np.where(
        duration_days > 0,
        out["num_events"] / duration_days,
        out["num_events"],  # treat as "all events on day 0"
    )

    out["total_time_days"] = out["wf_total_time"] / _SECONDS_PER_DAY

    # -- contributor / comment ratio ------------------------------------------
    # avoid division by zero for tickets with no contributors logged
    out["comments_per_contributor"] = np.where(
        out["issue_contr_count"] > 0,
        out["issue_comments_count"] / out["issue_contr_count"],
        out["issue_comments_count"],
    )

    # -- priority flag --------------------------------------------------------
    if "issue_priority" in out.columns:
        out["is_high_priority"] = (out["issue_priority"] == "High").astype(int)

    # -- log-transformed skewed numerics --------------------------------------
    out["log_total_time"] = np.log1p(out["wf_total_time"])
    out["log_num_events"] = np.log1p(out["num_events"])

    logger.info(
        "Feature engineering complete — added columns: %s",
        [
            "events_per_day",
            "total_time_days",
            "comments_per_contributor",
            "is_high_priority",
            "log_total_time",
            "log_num_events",
        ],
    )
    return out
