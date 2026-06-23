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
    max_source_shape_correlation: float = 0.9999,
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
    rows.append(_source_hydrograph_shape_diversity(event_discharge_nc, max_correlation=max_source_shape_correlation))
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


def _source_hydrograph_shape_diversity(discharge_nc, *, max_correlation: float) -> dict:
    with xr.open_dataset(discharge_nc) as ds:
        if "discharge" not in ds:
            raise ValueError(f"{discharge_nc} lacks discharge variable")
        da = ds["discharge"]
        if "index" not in da.dims or da.sizes.get("index", 0) < 2:
            return {
                "check": "source_hydrograph_shape_diversity",
                "status": "passed",
                "message": "sources<2; diversity check skipped",
            }
        values = da.transpose("index", "time").values.astype(float)
        names = discharge_source_ids(discharge_nc)

    max_seen = -np.inf
    duplicate_pairs: list[str] = []
    for i in range(values.shape[0]):
        left = _normalized_shape(values[i])
        for j in range(i + 1, values.shape[0]):
            right = _normalized_shape(values[j])
            if np.allclose(left, right, rtol=1e-6, atol=1e-8):
                corr = 1.0
            else:
                corr = float(np.corrcoef(left, right)[0, 1])
            max_seen = max(max_seen, corr)
            if corr >= float(max_correlation):
                duplicate_pairs.append(f"{names[i]}~{names[j]}:{corr:.6f}")
    status = "failed" if duplicate_pairs else "passed"
    pairs = ", ".join(duplicate_pairs[:5]) if duplicate_pairs else "none"
    if len(duplicate_pairs) > 5:
        pairs += f", +{len(duplicate_pairs) - 5} more"
    return {
        "check": "source_hydrograph_shape_diversity",
        "status": status,
        "message": f"sources={values.shape[0]}; max_corr={max_seen:.6f}; duplicate_shape_pairs={pairs}",
    }


def _normalized_shape(values: np.ndarray) -> np.ndarray:
    clean = np.asarray(values, dtype=float)
    if clean.size == 0:
        return clean
    fill = np.nanmedian(clean) if np.isfinite(clean).any() else 0.0
    clean = np.nan_to_num(clean, nan=float(fill), posinf=float(fill), neginf=float(fill))
    span = float(clean.max() - clean.min())
    if span <= 0.0:
        return np.zeros_like(clean, dtype=float)
    return (clean - float(clean.min())) / span
