from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def gauge_discharge(run_model_root: str | Path, gauges_geojson: str | Path, *, run_output_dir: str | Path | None = None, model_cls=None):
    """Return Wflow gauge discharge series keyed by ``sfincs_handoff_id``.

    Preferred path: HydroMT-Wflow's native ``utils.read_csv_output`` parses the TOML
    output-csv contract and staticmaps. Fallback path: lightweight ``Q_<index>`` parser
    for older or externally produced Wflow CSVs.
    """
    try:
        return _native_gauge_discharge(run_model_root, gauges_geojson, run_output_dir=run_output_dir, model_cls=model_cls)
    except Exception:
        return _fallback_gauge_discharge(run_output_dir or Path(run_model_root) / "run_event", gauges_geojson)


def _native_gauge_discharge(run_model_root, gauges_geojson, *, run_output_dir=None, model_cls=None):
    import geopandas as gpd
    from hydromt_wflow import utils as wflow_utils
    from wflow_v2.wflow_boundary_compat.hydromt_native import read_model

    root = Path(run_model_root)
    model = read_model(root, model_cls=model_cls, mode="r")
    csv_path = _output_csv(run_output_dir or root / "run_event")
    outputs = wflow_utils.read_csv_output(csv_path, model.config.data, model.staticmaps.data)
    gauges = gpd.read_file(gauges_geojson)
    if "sfincs_handoff_id" not in gauges:
        raise ValueError(f"{gauges_geojson} lacks sfincs_handoff_id")

    # HydroMT returns one GeoDataArray per TOML csv column. Prefer the discharge layer.
    da = _select_discharge_output(outputs)
    frame = da.to_pandas()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if frame.index.name != "time":
        frame.index = pd.DatetimeIndex(frame.index)

    gauges_by_index = {int(r["index"]): str(r["sfincs_handoff_id"]) for _, r in gauges.iterrows() if "index" in gauges}
    series: dict[str, pd.Series] = {}
    points: dict[str, tuple[float, float]] = {}
    for col in frame.columns:
        try:
            idx = int(col)
        except Exception:
            idx = _index_from_label(col)
        hid = gauges_by_index.get(idx)
        if hid is None:
            continue
        series[hid] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    for _, row in gauges.iterrows():
        hid = str(row["sfincs_handoff_id"])
        if hid in series:
            points[hid] = (float(row.geometry.x), float(row.geometry.y))
    if not series:
        raise ValueError("native read_csv_output did not produce matching SFINCS gauge series")
    return series, points, gauges.crs


def _fallback_gauge_discharge(run_output_dir, gauges_geojson):
    import geopandas as gpd

    gauges = gpd.read_file(gauges_geojson)
    table = pd.read_csv(_output_csv(run_output_dir), index_col=0, parse_dates=True)
    table.index = pd.DatetimeIndex(table.index)
    series: dict[str, pd.Series] = {}
    points: dict[str, tuple[float, float]] = {}
    for _, row in gauges.iterrows():
        hid = str(row.get("sfincs_handoff_id") or row.get("name"))
        col = match_gauge_column(table.columns, row.get("index"))
        if hid and col:
            series[hid] = pd.to_numeric(table[col], errors="coerce").astype(float)
            points[hid] = (float(row.geometry.x), float(row.geometry.y))
    if not series:
        raise ValueError(f"No Wflow gauge discharge columns matched {gauges_geojson}")
    return series, points, gauges.crs


def _select_discharge_output(outputs: dict[str, Any]):
    for key, value in outputs.items():
        low = str(key).lower()
        if "river_q" in low or "volume_flow" in low or low.startswith("q"):
            return value
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    raise ValueError(f"could not identify discharge output among {sorted(outputs)}")


def _index_from_label(value) -> int | None:
    import re

    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def match_gauge_column(columns, index_value) -> str | None:
    try:
        index_text = str(int(float(index_value)))
    except Exception:
        return None
    exact = {f"Q_{index_text}", f"Q_gauges_sfincs_{index_text}", f"river_q_{index_text}"}
    for column in columns:
        if str(column) in exact:
            return str(column)
    for column in columns:
        text = str(column)
        if text.startswith(("Q", "river_q")) and text.endswith(f"_{index_text}"):
            return text
    return None


def _output_csv(run_output_dir: str | Path) -> Path:
    root = Path(run_output_dir)
    for name in ("output.csv", "output_scalar.csv"):
        path = root / name
        if path.exists():
            return path
    matches = sorted(root.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"no Wflow output CSV under {root}")
    return matches[0]
