from __future__ import annotations

import pandas as pd

# long driver field -> wide production column name(s), formatted with the driver name.
_FIELD_TO_WIDE = {
    "x": ["{d}"],                       # physical magnitude (Driver Probability Index target)
    "u": ["{d}_u"],                     # marginal probability index
    "member_id": ["{d}_member_id", "{d}_template_member_id"],
    "member_file": ["{d}_member_file"],
    "member_time": ["{d}_member_time"],
    "template_value": ["{d}_template_value"],
    "scale_factor": ["{d}_scale_factor"],
    "lag_hours": ["{d}_realization_lag_hours"],
    "realization_policy": ["{d}_design_method"],
    "source": ["{d}_source"],
}


def to_wide_handoff(events: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Return ``events`` widened with one set of ``<driver>_*`` columns per driver.

    One row per event (event-level columns preserved); each driver's long realization
    fields are mapped onto the production wide column names. Drivers absent for an event
    yield NaN in that event's row.
    """
    wide = events.copy()
    for driver, group in drivers.groupby("driver"):
        indexed = group.drop_duplicates("event_id").set_index("event_id")
        for field, templates in _FIELD_TO_WIDE.items():
            if field not in indexed.columns:
                continue
            values = wide["event_id"].map(indexed[field])
            for template in templates:
                wide[template.format(d=driver)] = values.to_numpy()
    return wide


__all__ = ["to_wide_handoff"]
