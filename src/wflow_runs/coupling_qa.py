from __future__ import annotations

import pandas as pd

from wflow_runs.qa import (
    read_acceptance as read_dynamic_handoff_acceptance,
    write_dynamic_handoff_acceptance,
)
from wflow_runs.qa import validate_event_boundary as _validate_event_boundary
from wflow_runs.qa import validate_handoff_gauge_locations


def validate_dynamic_handoff(
    event_discharge_nc,
    *,
    zero_rain_discharge_nc=None,
    expected_source_ids: set[str] | None = None,
    max_zero_peak_fraction: float | None = None,
    max_source_shape_correlation: float = 0.9999,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    report = _validate_event_boundary(
        event_discharge_nc,
        zero_rain_discharge_nc=zero_rain_discharge_nc,
        expected_source_ids=expected_source_ids,
        max_zero_peak_fraction=max_zero_peak_fraction,
        max_shape_correlation=max_source_shape_correlation,
        raise_on_error=False,
    )
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Dynamic Wflow handoff QA failed: {details}")
    return report


__all__ = [
    "read_dynamic_handoff_acceptance",
    "validate_dynamic_handoff",
    "validate_handoff_gauge_locations",
    "write_dynamic_handoff_acceptance",
]
