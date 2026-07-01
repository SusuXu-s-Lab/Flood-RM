from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_submodel_gauge_discharge(
    run_output_dir: Path,
    gauges_geojson: Path,
    *,
    csv_name: str | None = None,
):
    """Read one submodel's Wflow gauge discharge and handoff locations."""
    if csv_name is None:
        from wflow_runs.event import gauge_discharge

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
