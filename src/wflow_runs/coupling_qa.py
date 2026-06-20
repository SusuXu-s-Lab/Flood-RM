from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def discharge_source_ids(discharge_nc) -> list[str]:
    with xr.open_dataset(discharge_nc) as ds:
        if "name" in ds:
            return [str(value) for value in ds["name"].values.tolist()]
        return [str(value) for value in ds.get("index", []).values.tolist()]


def validate_dynamic_handoff(
    event_discharge_nc,
    *,
    zero_rain_discharge_nc=None,
    expected_source_ids: set[str] | None = None,
    max_zero_peak_fraction: float = 0.2,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    rows: list[dict] = []
    event_peak = _total_peak(event_discharge_nc)
    rows.append({"check": "event_peak", "status": "passed" if event_peak > 0 else "failed", "message": f"peak_m3s={event_peak:g}"})
    if expected_source_ids is not None:
        found = set(discharge_source_ids(event_discharge_nc))
        missing = sorted(expected_source_ids - found)
        stale = sorted(found - expected_source_ids)
        status = "passed" if not missing and not stale else "failed"
        rows.append({"check": "source_ids", "status": status, "message": f"missing={missing or 'none'}; stale={stale or 'none'}"})
    if zero_rain_discharge_nc is not None:
        zero_peak = _total_peak(zero_rain_discharge_nc)
        fraction = zero_peak / event_peak if event_peak > 0 else np.inf
        status = "passed" if fraction <= float(max_zero_peak_fraction) else "failed"
        rows.append({"check": "zero_rain_peak_fraction", "status": status, "message": f"zero_peak_m3s={zero_peak:g}; fraction={fraction:g}"})
    report = pd.DataFrame(rows)
    failed = report[report["status"].isin(["failed", "review_required"])]
    if raise_on_error and not failed.empty:
        details = "; ".join(f"{row.check}: {row.message}" for row in failed.itertuples())
        raise RuntimeError(f"Dynamic Wflow handoff QA failed: {details}")
    return report


def write_dynamic_handoff_acceptance(path, *, event_id: str, discharge_nc, qa_report: pd.DataFrame, metadata: dict | None = None) -> Path:
    path = Path(path)
    accepted = bool(not qa_report["status"].isin(["failed", "review_required"]).any())
    payload = {
        "event_id": str(event_id),
        "status": "accepted" if accepted else "failed",
        "discharge_source": "wflow_dynamic",
        "discharge_nc": str(discharge_nc),
        "checks": qa_report.to_dict(orient="records"),
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_dynamic_handoff_acceptance(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _total_peak(discharge_nc) -> float:
    with xr.open_dataset(discharge_nc) as ds:
        if "discharge" not in ds:
            raise ValueError(f"{discharge_nc} lacks discharge variable")
        da = ds["discharge"]
        if "index" in da.dims:
            da = da.sum("index")
        return float(da.max(skipna=True))
