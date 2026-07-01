from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_submodel_gauge_discharge(
    run_output_dir: Path,
    gauges_geojson: Path,
    *,
    csv_name: str | None = None,
):
    """Read one submodel's Wflow gauge discharge and handoff locations."""
    if csv_name is None:
        series, points, _crs = gauge_discharge(
            Path(run_output_dir).parent,
            gauges_geojson,
            run_output_dir=run_output_dir,
        )
        return series, points

    import geopandas as gpd

    gauges = gpd.read_file(gauges_geojson)
    output_csv = resolve_wflow_output_csv(Path(run_output_dir), csv_name)
    table = pd.read_csv(output_csv, index_col=0, parse_dates=True)
    series_by_handoff: dict[str, pd.Series] = {}
    points_by_handoff: dict[str, tuple[float, float]] = {}
    for _, gauge in gauges.iterrows():
        handoff_id = str(gauge["sfincs_handoff_id"])
        column = match_gauge_column(table.columns, gauge["index"])
        if column is not None:
            series_by_handoff[handoff_id] = pd.to_numeric(table[column], errors="coerce").astype(float)
            points_by_handoff[handoff_id] = (float(gauge.geometry.x), float(gauge.geometry.y))
    if not series_by_handoff:
        raise ValueError(f"no Q_<index> gauge columns matched in {output_csv}")
    return series_by_handoff, points_by_handoff


def read_wflow_event_output_csv(
    event_id,
    *,
    events_root,
    submodel_id=None,
    control: bool = False,
) -> pd.DataFrame:
    submodel_id = submodel_id or first_event_submodel_id(events_root, event_id)
    if not submodel_id:
        return pd.DataFrame()
    event_root = Path(events_root) / str(event_id)
    model_root = event_root / "_zero_rain" / submodel_id if control else event_root / submodel_id
    csv_path = model_root / "run_event" / "output.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame


def gauge_output_map(
    event_id,
    *,
    events_root,
    wflow_base_root,
    layer="gauges_usgs",
    submodel_id=None,
) -> pd.DataFrame:
    gauges = read_wflow_gauge_layer(
        event_id,
        events_root=events_root,
        wflow_base_root=wflow_base_root,
        layer=layer,
        submodel_id=submodel_id,
    )
    if gauges.empty:
        return gauges
    gauges["q_column"] = "Q_" + gauges["index"].astype(int).astype(str)
    return pd.DataFrame(gauges.drop(columns="geometry"))


def read_wflow_gauge_layer(
    event_id,
    *,
    events_root,
    wflow_base_root,
    layer: str,
    submodel_id=None,
) -> pd.DataFrame:
    import geopandas as gpd

    submodel_id = submodel_id or first_event_submodel_id(events_root, event_id) or first_base_submodel_id(wflow_base_root)
    if not submodel_id:
        return pd.DataFrame()
    gauges_path = Path(events_root) / str(event_id) / submodel_id / "staticgeoms" / f"{layer}.geojson"
    if not gauges_path.exists():
        gauges_path = Path(wflow_base_root) / submodel_id / "staticgeoms" / f"{layer}.geojson"
    if not gauges_path.exists():
        return pd.DataFrame()
    return gpd.read_file(gauges_path)


def first_event_submodel_id(events_root, event_id) -> str | None:
    event_root = Path(events_root) / str(event_id)
    if not event_root.exists():
        return None
    for child in sorted(event_root.iterdir()):
        if child.is_dir() and (child / "run_event").exists():
            return child.name
    return None


def first_base_submodel_id(wflow_base_root) -> str | None:
    root = Path(wflow_base_root)
    if not root.exists():
        return None
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "staticgeoms").exists():
            return child.name
    return None


def gauge_discharge(
    run_model_root: str | Path,
    gauges_geojson: str | Path,
    *,
    run_output_dir: str | Path | None = None,
    model_cls=None,
):
    """Return Wflow gauge discharge series keyed by ``sfincs_handoff_id``."""
    try:
        return _native_gauge_discharge(
            run_model_root,
            gauges_geojson,
            run_output_dir=run_output_dir,
            model_cls=model_cls,
        )
    except Exception:
        return _fallback_gauge_discharge(
            run_output_dir or Path(run_model_root) / "run_event",
            gauges_geojson,
        )


def _native_gauge_discharge(run_model_root, gauges_geojson, *, run_output_dir=None, model_cls=None):
    import geopandas as gpd
    from hydromt_wflow import utils as wflow_utils

    root = Path(run_model_root)
    model = _read_model(root, model_cls=model_cls, mode="r")
    csv_path = wflow_output_csv(run_output_dir or root / "run_event")
    outputs = wflow_utils.read_csv_output(csv_path, model.config.data, model.staticmaps.data)
    gauges = gpd.read_file(gauges_geojson)
    if "sfincs_handoff_id" not in gauges:
        raise ValueError(f"{gauges_geojson} lacks sfincs_handoff_id")
    da = _select_discharge_output(outputs)
    frame = da.to_pandas()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    if frame.index.name != "time":
        frame.index = pd.DatetimeIndex(frame.index)

    gauges_by_index = {
        int(row["index"]): str(row["sfincs_handoff_id"])
        for _, row in gauges.iterrows()
        if "index" in gauges
    }
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
    table = pd.read_csv(wflow_output_csv(run_output_dir), index_col=0, parse_dates=True)
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


def _read_model(root: str | Path, *, model_cls=None, mode: str = "r"):
    cls = _wflow_model_cls(model_cls)
    model = cls(root=str(root), mode=mode)
    model.read()
    return model


def _wflow_model_cls(model_cls=None):
    if model_cls is not None:
        return model_cls
    from hydromt_wflow import WflowSbmModel

    return WflowSbmModel


def resolve_wflow_output_csv(run_output_dir: Path, csv_name: str | None) -> Path:
    if csv_name:
        candidate = run_output_dir / csv_name
        if candidate.exists():
            return candidate
    return wflow_output_csv(run_output_dir)


def wflow_output_csv(run_output_dir: str | Path) -> Path:
    root = Path(run_output_dir)
    for name in ["output.csv", "output_scalar.csv"]:
        candidate = root / name
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"no Wflow output CSV under {root}")
    return matches[0]


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
