from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

FT_TO_M = 0.3048
ACRE_TO_M2 = 4046.8564224
ACRE_FT_TO_M3 = 1233.48183754752
TWDB_RESERVOIR_BASE_URL = "https://waterdatafortexas.org/reservoirs/individual"


@dataclass(frozen=True)
class ReservoirConditionSummary:
    frame: gpd.GeoDataFrame
    summary: pd.DataFrame
    provenance: dict


def enrich_wflow_reservoirs_with_public_conditions(
    reservoirs,
    condition_cfg: dict | None,
    *,
    location_root,
    output_path=None,
    skip_existing=True,
    session_get=None,
) -> ReservoirConditionSummary:
    """Join public reservoir condition statistics into a Wflow reservoir source layer."""
    condition_cfg = condition_cfg or {}
    if not condition_cfg.get("enabled", False):
        frame = _read_reservoirs(reservoirs)
        return ReservoirConditionSummary(frame=frame, summary=pd.DataFrame(), provenance={"status": "disabled"})

    location_root = Path(location_root)
    cache_dir = _location_path(
        location_root,
        condition_cfg.get("cache_dir", "data/sources/twdb_reservoirs"),
    )
    summary_csv = _location_path(
        location_root,
        condition_cfg.get("summary_csv", "data/sources/twdb_reservoirs/reservoir_condition_summary.csv"),
    )
    provenance_json = _location_path(
        location_root,
        condition_cfg.get("provenance_json", "data/sources/twdb_reservoirs/reservoir_condition_provenance.json"),
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.parent.mkdir(parents=True, exist_ok=True)

    frame = _read_reservoirs(reservoirs)
    if frame.empty:
        provenance = {"status": "empty", "rows": 0}
        _write_json(provenance_json, provenance)
        summary_csv.write_text("", encoding="utf-8")
        return ReservoirConditionSummary(frame=frame, summary=pd.DataFrame(), provenance=provenance)

    slug_map = _slug_map(condition_cfg)
    statistic = str(condition_cfg.get("statistic", "median")).lower()
    period_suffix = str(condition_cfg.get("period_suffix", "-1year"))
    start_date = condition_cfg.get("start_date")
    end_date = condition_cfg.get("end_date")
    timeout = float(condition_cfg.get("request_timeout_seconds", 60))
    base_url = str(condition_cfg.get("base_url", TWDB_RESERVOIR_BASE_URL)).rstrip("/")
    get = session_get or requests.get

    records = []
    normalized_names = frame.get("waterbody_name", pd.Series("", index=frame.index)).fillna("").astype(str)
    for idx, name in normalized_names.items():
        slug = _match_slug(name, slug_map)
        if not slug:
            records.append(_missing_condition_record(idx, name, reason="no configured TWDB slug"))
            continue
        url = f"{base_url}/{slug}{period_suffix}.csv"
        raw_path = cache_dir / f"{slug}{period_suffix}.csv"
        try:
            raw_text = _fetch_text(url, raw_path, skip_existing=skip_existing, timeout=timeout, session_get=get)
            history = parse_twdb_reservoir_csv(raw_text)
            record = summarize_twdb_reservoir_history(
                history,
                waterbody_index=idx,
                waterbody_name=name,
                slug=slug,
                url=url,
                statistic=statistic,
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as exc:  # keep collection auditable instead of half-writing silently
            record = _missing_condition_record(idx, name, slug=slug, reason=f"{type(exc).__name__}: {exc}", url=url)
        records.append(record)

    summary = pd.DataFrame(records)
    frame = apply_reservoir_condition_summary(frame, summary, condition_cfg=condition_cfg)

    summary.to_csv(summary_csv, index=False)
    provenance = {
        "status": "collected",
        "provider": condition_cfg.get("provider", "twdb_water_data_for_texas"),
        "base_url": base_url,
        "period_suffix": period_suffix,
        "statistic": statistic,
        "summary_csv": str(summary_csv),
        "matched_reservoirs": int(summary["condition_status"].eq("matched").sum()) if not summary.empty else 0,
        "unmatched_reservoirs": int(summary["condition_status"].ne("matched").sum()) if not summary.empty else 0,
        "source_note": (
            "TWDB reservoir history supplies water level, storage, and surface area. "
            "Wflow Dis_avg remains the configured river-geometry fallback unless a reviewed release/outflow source is supplied."
        ),
    }
    _write_json(provenance_json, provenance)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_file(output_path, driver="GPKG")
    return ReservoirConditionSummary(frame=frame, summary=summary, provenance=provenance)


def parse_twdb_reservoir_csv(text: str) -> pd.DataFrame:
    """Parse a Water Data for Texas reservoir CSV with its comment preamble."""
    lines = text.splitlines()
    header_index = next(
        (idx for idx, line in enumerate(lines) if line.strip().lower().startswith("date,")),
        None,
    )
    if header_index is None:
        raise ValueError("TWDB reservoir CSV did not contain a date header")
    frame = pd.read_csv(StringIO("\n".join(lines[header_index:])))
    if "date" not in frame:
        raise ValueError("TWDB reservoir CSV missing date column")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).copy()


def summarize_twdb_reservoir_history(
    history: pd.DataFrame,
    *,
    waterbody_index,
    waterbody_name: str,
    slug: str,
    url: str,
    statistic: str = "median",
    start_date=None,
    end_date=None,
) -> dict:
    """Summarize one TWDB reservoir history into Wflow no-control parameters."""
    frame = history.copy()
    if start_date is not None:
        frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame.loc[frame["date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        raise ValueError("no TWDB rows in configured condition window")

    values = {
        "water_level_ft": _aggregate(frame, "water_level", statistic),
        "surface_area_acres": _aggregate(frame, "surface_area", statistic),
        "reservoir_storage_acft": _aggregate(frame, "reservoir_storage", statistic),
        "conservation_storage_acft": _aggregate(frame, "conservation_storage", statistic),
        "percent_full": _aggregate(frame, "percent_full", statistic),
        "conservation_capacity_acft": _aggregate(frame, "conservation_capacity", statistic),
    }
    area_m2 = _to_float(values["surface_area_acres"]) * ACRE_TO_M2
    storage_m3 = _to_float(values["reservoir_storage_acft"]) * ACRE_FT_TO_M3
    depth_m = storage_m3 / area_m2 if area_m2 and np.isfinite(area_m2) else np.nan
    return {
        "waterbody_index": waterbody_index,
        "waterbody_name": waterbody_name,
        "twdb_slug": slug,
        "condition_status": "matched",
        "condition_source": "TWDB Water Data for Texas",
        "condition_url": url,
        "condition_statistic": statistic,
        "condition_period_start": frame["date"].min().date().isoformat(),
        "condition_period_end": frame["date"].max().date().isoformat(),
        "condition_rows": int(len(frame)),
        "Area_avg": area_m2,
        "Vol_avg": storage_m3,
        "Depth_avg": depth_m,
        **values,
    }


def apply_reservoir_condition_summary(
    reservoirs: gpd.GeoDataFrame,
    summary: pd.DataFrame,
    *,
    condition_cfg: dict | None = None,
) -> gpd.GeoDataFrame:
    """Apply condition summaries to a Wflow reservoir GeoDataFrame."""
    condition_cfg = condition_cfg or {}
    result = reservoirs.copy()
    if result.empty or summary.empty:
        return result

    metadata_columns = [
        "twdb_slug",
        "condition_status",
        "condition_source",
        "condition_url",
        "condition_statistic",
        "condition_period_start",
        "condition_period_end",
        "condition_reason",
    ]
    for column in metadata_columns:
        if column in result:
            result[column] = result[column].astype("object")
        else:
            result[column] = pd.Series("", index=result.index, dtype="object")

    min_depth = float(condition_cfg.get("min_depth_m", 0.25))
    max_depth = float(condition_cfg.get("max_depth_m", 200.0))
    source_label = condition_cfg.get("provider_label", "TWDB Water Data for Texas")
    for _, record in summary.iterrows():
        idx = record["waterbody_index"]
        if idx not in result.index:
            continue
        status = str(record.get("condition_status", "missing"))
        if status == "matched":
            area = _to_float(record.get("Area_avg"))
            volume = _to_float(record.get("Vol_avg"))
            depth = float(np.clip(_to_float(record.get("Depth_avg")), min_depth, max_depth))
            if np.isfinite(area) and area > 0:
                result.loc[idx, "Area_avg"] = area
            if np.isfinite(volume) and volume > 0:
                result.loc[idx, "Vol_avg"] = volume
            if np.isfinite(depth) and depth > 0:
                result.loc[idx, "Depth_avg"] = depth
            result.loc[idx, "reservoir_parameter_source"] = (
                f"{source_label} {record.get('condition_statistic', 'median')} storage/surface area; "
                "Dis_avg from river-geometry fallback"
            )
            result.loc[idx, "review_status"] = "review_required_twdb_storage_no_control_outflow_fallback"
        else:
            result.loc[idx, "review_status"] = "review_required_missing_twdb_condition_bootstrap"

        for column in [
            "twdb_slug",
            "condition_status",
            "condition_source",
            "condition_url",
            "condition_statistic",
            "condition_period_start",
            "condition_period_end",
            "condition_rows",
            "water_level_ft",
            "surface_area_acres",
            "reservoir_storage_acft",
            "conservation_storage_acft",
            "percent_full",
            "conservation_capacity_acft",
            "condition_reason",
        ]:
            if column in record:
                result.loc[idx, column] = record.get(column)
    return result


def _fetch_text(url: str, path: Path, *, skip_existing: bool, timeout: float, session_get) -> str:
    if skip_existing and path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    response = session_get(url, timeout=timeout)
    response.raise_for_status()
    text = response.text
    path.write_text(text, encoding="utf-8")
    return text


def _aggregate(frame: pd.DataFrame, column: str, statistic: str):
    if column not in frame:
        return np.nan
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    if statistic == "mean":
        return float(values.mean())
    if statistic == "last":
        return float(values.iloc[-1])
    if statistic == "min":
        return float(values.min())
    if statistic == "max":
        return float(values.max())
    return float(values.median())


def _slug_map(condition_cfg: dict) -> dict[str, str]:
    configured = condition_cfg.get("reservoir_slugs", {}) or {}
    return {_normalize_name(name): str(slug) for name, slug in configured.items()}


def _match_slug(name: str, slug_map: dict[str, str]) -> str | None:
    normalized = _normalize_name(name)
    if normalized in slug_map:
        return slug_map[normalized]
    compact = normalized.replace("lake ", "")
    for candidate, slug in slug_map.items():
        if compact and compact == candidate.replace("lake ", ""):
            return slug
    return None


def _missing_condition_record(waterbody_index, waterbody_name: str, *, slug=None, reason: str, url: str | None = None) -> dict:
    return {
        "waterbody_index": waterbody_index,
        "waterbody_name": waterbody_name,
        "twdb_slug": slug or "",
        "condition_status": "missing",
        "condition_source": "",
        "condition_url": url or "",
        "condition_statistic": "",
        "condition_period_start": "",
        "condition_period_end": "",
        "condition_rows": 0,
        "condition_reason": reason,
    }


def _read_reservoirs(reservoirs) -> gpd.GeoDataFrame:
    if isinstance(reservoirs, gpd.GeoDataFrame):
        return reservoirs.copy()
    return gpd.read_file(reservoirs)


def _normalize_name(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).lower())).strip()


def _location_path(location_root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else location_root / path


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
