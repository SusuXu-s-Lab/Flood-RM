from __future__ import annotations

# Source-collection workflow: planning, prerequisite preparation, collection execution,
# and notebook-facing review helpers live together so the Flood Notebook Workflow has one
# orchestration module to import.

# Collection plan
from dataclasses import dataclass
from importlib import import_module
import json

import pandas as pd

from location_runtime import static_sources_with_defaults

COLLECTORS = {
    "aorc": "collect_sources.aorc:collect",
    "aorc_sst": "collect_sources.rainfall:collect",
    "cora": "collect_sources.cora:collect_cora",
    "era5": "collect_sources.era5:collect",
    "era5_waves": "collect_sources.era5:collect",
    "nwm": "collect_sources.nwm:collect_nwm",
    "usgs": "collect_sources.usgs:collect",
    "usgs_streamgages": "collect_sources.usgs_streamgages:collect_usgs_streamgages",
    "hurdat2": "collect_sources.hurdat2:collect_hurdat2",
    "lcra_hydromet": "collect_sources.lcra_hydromet:collect_lcra_hydromet",
    "stream_geo": "collect_sources.stream_geo:collect",
    "stream_geo_nldi": "collect_sources.stream_geo_nldi:collect_stream_geo_nldi",
    "ssurgo": "collect_sources.ssurgo:collect_ssurgo",
    "national_hydrography": "collect_sources.national_hydrography:collect_national_hydrography",
}


def _collector(name: str):
    module_name, function_name = COLLECTORS[name].split(":")
    return getattr(import_module(module_name), function_name)


@dataclass(frozen=True)
class SourceCollectionStep:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    spec: dict

    @property
    def start_date(self):
        return self.start.date().isoformat()

    @property
    def end_date(self):
        return self.end.date().isoformat()


@dataclass(frozen=True)
class SourceCollectionPlan:
    config: dict
    paths: dict
    start: pd.Timestamp
    end: pd.Timestamp
    steps: tuple[SourceCollectionStep, ...]

    @property
    def source_names(self):
        return tuple(step.name for step in self.steps)

    def has(self, name):
        return name in self.source_names

    def step(self, name):
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"source is not configured: {name}")

    def settings_for(self, name):
        step = self.step(name)
        return {
            "config": self.config,
            "paths": self.paths,
            "start": step.start,
            "end": step.end,
            name: step.spec,
        }

    def summary_rows(self):
        return [
            {
                "source": step.name,
                "start": step.start_date,
                "end": step.end_date,
            }
            for step in self.steps
        ]


source_order = (
    "cora",
    "usgs_streamgages",
    "lcra_hydromet",
    "stream_geo_nldi",
    "national_hydrography",
    "nwm",
    "aorc_sst",
    "era5_waves",
    "hurdat2",
)


def plan(config, paths, *, start=None, end=None):
    base_start, base_end = _collection_window(config, start=start, end=end)
    collection = config.get("collection", {})
    steps = []
    for name in source_order:
        if name not in collection:
            continue
        spec = collection.get(name) or {}
        source_start, source_end = _source_window(name, spec, base_start, base_end)
        steps.append(
            SourceCollectionStep(
                name=name,
                start=source_start,
                end=source_end,
                spec=spec,
            )
        )
    return SourceCollectionPlan(
        config=config,
        paths=paths,
        start=base_start,
        end=base_end,
        steps=tuple(steps),
    )


def _collection_window(config, start=None, end=None):
    collection = config.get("collection", {})
    start_ts = pd.Timestamp(start or collection.get("start", "1979-01-01"))
    end_ts = pd.Timestamp(end or collection.get("end", "2022-12-31"))
    if end_ts < start_ts:
        raise ValueError("end date must be on or after start date")
    return start_ts, end_ts


def _source_window(name, spec, base_start, base_end):
    if name == "nwm":
        return _bounded_window(base_start, base_end, spec, "start", "end", "nwm")
    if name == "aorc_sst":
        return _bounded_window(base_start, base_end, spec, "start_date", "end_date", "aorc_sst")
    return base_start, base_end


def _bounded_window(base_start, base_end, spec, start_key, end_key, label):
    start_ts = pd.Timestamp(spec.get(start_key, base_start))
    end_ts = pd.Timestamp(spec.get(end_key, base_end))
    if start_ts < base_start or end_ts > base_end or end_ts < start_ts:
        raise ValueError(f"{label} collection dates must stay within the base collection window")
    return start_ts, end_ts


# Collection runner
import time

from tqdm.auto import tqdm as iter_progress


def _default_run_collect_funcs():
    from collect_sources.aorc_sst import collect_aorc_sst
    from collect_sources.cora import collect_cora
    from collect_sources.era5_waves import collect_era5_waves
    from collect_sources.hurdat2 import collect_hurdat2
    from collect_sources.lcra_hydromet import collect_lcra_hydromet
    from collect_sources.national_hydrography import collect_national_hydrography
    from collect_sources.nwm import collect_nwm
    from collect_sources.stream_geo_nldi import collect_stream_geo_nldi
    from collect_sources.usgs_streamgages import collect_usgs_streamgages

    return {
        "collect_aorc_sst": collect_aorc_sst,
        "collect_cora": collect_cora,
        "collect_era5_waves": collect_era5_waves,
        "collect_hurdat2": collect_hurdat2,
        "collect_lcra_hydromet": collect_lcra_hydromet,
        "collect_national_hydrography": collect_national_hydrography,
        "collect_nwm": collect_nwm,
        "collect_stream_geo_nldi": collect_stream_geo_nldi,
        "collect_usgs_streamgages": collect_usgs_streamgages,
    }


def _record(records, source, status, started, **details):
    records.append(
        {
            "source": source,
            "status": status,
            "duration_seconds": round(time.monotonic() - started, 2),
            **details,
        }
    )


def _steps(plan, progress):
    if not progress:
        return plan.steps
    return iter_progress(plan.steps, desc="Collecting sources", unit="source")


def run_collect(
    config,
    paths,
    plan,
    *,
    run_collection=True,
    skip_existing=True,
    stop_on_error=True,
    progress=True,
    funcs=None,
):
    """Run the configured source collection plan and return notebook-friendly rows."""
    funcs = {**_default_run_collect_funcs(), **(funcs or {})}
    rows = []
    if not run_collection:
        return pd.DataFrame(
            [
                {
                    "source": "collection",
                    "status": "dry_run",
                    "duration_seconds": 0.0,
                    "rows": pd.NA,
                    "artifact": pd.NA,
                }
            ]
        )

    for step in _steps(plan, progress):
        started = time.monotonic()
        settings = plan.settings_for(step.name)
        try:
            if step.name == "cora":
                frame = funcs["collect_cora"](settings, skip_existing=skip_existing, smoke=False)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=len(frame),
                    artifact=str(paths["waterlevel_csv"]),
                )
            elif step.name == "usgs_streamgages":
                result = funcs["collect_usgs_streamgages"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else "collected"
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("candidate_count", 0),
                    artifact=str(result.get("candidate_geojson")),
                )
            elif step.name == "lcra_hydromet":
                result = funcs["collect_lcra_hydromet"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else "collected"
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("candidate_count", 0),
                    artifact=str(result.get("candidate_geojson")),
                )
            elif step.name == "nwm":
                result = funcs["collect_nwm"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else "collected"
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("soil_moisture_rows", 0),
                    artifact=str(result.get("soil_moisture_csv")),
                )
            elif step.name == "stream_geo_nldi":
                result = funcs["collect_stream_geo_nldi"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else result.get("status", "collected")
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("rows", 0),
                    artifact=str(result.get("stream_geo_table")),
                )
            elif step.name == "national_hydrography":
                result = funcs["collect_national_hydrography"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else "collected"
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("artifact_count", 0),
                    artifact=str(result.get("hydromt_basemap")),
                )
            elif step.name == "aorc_sst":
                result = funcs["collect_aorc_sst"](settings, skip_existing=skip_existing)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=result.get("ranked_rows", 0),
                    artifact=str(result.get("ranked_storms_csv")),
                )
            elif step.name == "era5_waves":
                result = funcs["collect_era5_waves"](settings, skip_existing=skip_existing, smoke=False)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=result.get("time_count", 0),
                    artifact=str(result.get("wave_netcdf")),
                )
            elif step.name == "hurdat2":
                result = funcs["collect_hurdat2"](settings, skip_existing=skip_existing, smoke=False)
                _record(
                    rows,
                    step.name,
                    "reused" if result.get("reused") else "collected",
                    started,
                    rows=result.get("track_points", 0),
                    artifact=str(result.get("hurdat2_tracks_csv")),
                )
        except Exception as exc:
            print(f"{step.name}: failed with {type(exc).__name__}: {exc}")
            _record(
                rows,
                step.name,
                "failed",
                started,
                rows=pd.NA,
                artifact=pd.NA,
                error=f"{type(exc).__name__}: {exc}",
            )
            if stop_on_error:
                raise

    started = time.monotonic()
    try:
        if plan.has("aorc_sst"):
            rainfall_members = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
        else:
            rainfall_members = pd.DataFrame()
        _record(
            rows,
            "rainfall_members",
            "collected" if plan.has("aorc_sst") else "not_configured",
            started,
            rows=len(rainfall_members),
            artifact=str(paths["aorc_sst_rainfall_members_csv"]),
        )
    except Exception as exc:
        _record(
            rows,
            "rainfall_members",
            "failed",
            started,
            rows=pd.NA,
            artifact=str(paths["aorc_sst_rainfall_members_csv"]),
            error=f"{type(exc).__name__}: {exc}",
        )
        if stop_on_error:
            raise

    return pd.DataFrame(rows)


def collect_all_sources(
    config,
    paths,
    *,
    start=None,
    end=None,
    skip_existing=False,
    smoke=False,
    funcs=None,
):
    funcs = {**_default_run_collect_funcs(), **(funcs or {})}
    collection_plan = plan(config, paths, start=start, end=end)
    cora_frame = pd.DataFrame()
    if collection_plan.has("cora"):
        cora_frame = funcs["collect_cora"](
            collection_plan.settings_for("cora"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    usgs_streamgages_result = None
    if collection_plan.has("usgs_streamgages"):
        usgs_streamgages_result = funcs["collect_usgs_streamgages"](
            collection_plan.settings_for("usgs_streamgages"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    nwm_result = None
    stream_geo_nldi_result = None
    if collection_plan.has("stream_geo_nldi"):
        stream_geo_nldi_result = funcs["collect_stream_geo_nldi"](
            collection_plan.settings_for("stream_geo_nldi"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    national_hydrography_result = None
    if collection_plan.has("national_hydrography"):
        national_hydrography_result = funcs["collect_national_hydrography"](
            collection_plan.settings_for("national_hydrography"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    if collection_plan.has("nwm"):
        nwm_result = funcs["collect_nwm"](
            collection_plan.settings_for("nwm"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    aorc_sst_result = None
    if collection_plan.has("aorc_sst"):
        aorc_sst_result = funcs["collect_aorc_sst"](
            collection_plan.settings_for("aorc_sst"),
            skip_existing=skip_existing,
        )
    era5_result = None
    if collection_plan.has("era5_waves"):
        era5_result = funcs["collect_era5_waves"](
            collection_plan.settings_for("era5_waves"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    hurdat2_result = None
    if collection_plan.has("hurdat2"):
        hurdat2_result = funcs["collect_hurdat2"](
            collection_plan.settings_for("hurdat2"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    return {
        "cora_rows": int(len(cora_frame)),
        "waterlevel_csv": paths.get("waterlevel_csv"),
        "usgs_streamgages": usgs_streamgages_result,
        "nwm": nwm_result,
        "stream_geo_nldi": stream_geo_nldi_result,
        "national_hydrography": national_hydrography_result,
        "aorc_sst": aorc_sst_result,
        "era5_waves": era5_result,
        "hurdat2": hurdat2_result,
    }


# Collection prerequisites
from pathlib import Path

import geopandas as gpd


def prepare(config, paths):
    """Create lightweight review-required inputs needed before source collection."""
    rows = []
    aorc_row = prepare_aorc_transposition_region(config, paths)
    if aorc_row is not None:
        rows.append(aorc_row)
    soil_row = prepare_nwm_soil_moisture_points(config, paths)
    if soil_row is not None:
        rows.append(soil_row)
    return pd.DataFrame(
        rows,
        columns=[
            "artifact",
            "path",
            "status",
            "source_geometry",
            "buffer_km",
            "review_status",
        ],
    )


def prepare_nwm_soil_moisture_points(config, paths):
    from collect_sources.ssurgo import ensure_points_geojson, has_footprint

    spec = config.get("collection", {}).get("nwm", {}).get("soil_moisture", {})
    if not spec or spec.get("points"):
        # No NWM soil-moisture source, or explicit points already pinned in the YAML.
        return None
    if not has_footprint(spec, paths):
        return None

    result = ensure_points_geojson(spec, paths)
    return {
        "artifact": "nwm soil-moisture sampling points",
        "path": str(result["path"]),
        "status": result["status"],
        "source_geometry": str(result["source_geometry"]) if result.get("source_geometry") else pd.NA,
        "buffer_km": pd.NA,
        "review_status": "review_required",
    }


def prepare_aorc_transposition_region(config, paths):
    collection = config.get("collection", {})
    spec = collection.get("aorc_sst", {})
    region = spec.get("transposition_region", {})
    geometry_file = region.get("geometry_file")
    if not geometry_file:
        return None

    output_path = _location_path(paths, geometry_file)
    if output_path.exists():
        return _result_row(
            output_path,
            status="reused",
            source_geometry=pd.NA,
            buffer_km=pd.NA,
            review_status=pd.NA,
        )

    source_path = _source_geometry_path(config, paths)
    source = gpd.read_file(source_path)
    if source.empty:
        raise ValueError(f"AORC transposition source geometry is empty: {source_path}")

    buffer_km = _transposition_buffer_km(config, region)
    model_crs = _model_crs(config)
    geometry = source.to_crs(model_crs).geometry.union_all().buffer(buffer_km * 1000.0)
    output = gpd.GeoDataFrame(
        {
            "region_id": [region.get("id", "review-required")],
            "source_geometry": [str(source_path)],
            "buffer_km": [float(buffer_km)],
            "review_status": ["review_required"],
            "review_notes": ["Generated from the evaluation footprint for source collection; review before production SST use."],
        },
        geometry=[geometry],
        crs=model_crs,
    ).to_crs("EPSG:4326")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_file(output_path, driver="GeoJSON")
    return _result_row(
        output_path,
        status="created_review_required",
        source_geometry=source_path,
        buffer_km=float(buffer_km),
        review_status="review_required",
    )


def _result_row(output_path, *, status, source_geometry, buffer_km, review_status):
    return {
        "artifact": "aorc_sst transposition region",
        "path": str(output_path),
        "status": status,
        "source_geometry": str(source_geometry) if isinstance(source_geometry, Path) else source_geometry,
        "buffer_km": buffer_km,
        "review_status": review_status,
    }


def _source_geometry_path(config, paths):
    candidates = [
        config.get("collection", {}).get("aorc_sst", {}).get("transposition_region", {}).get("source_geometry"),
        config.get("smart_ds_evaluation_footprint", {}).get("output"),
        config.get("grid_footprint", {}).get("source"),
        "data/static/aoi/evaluation_footprint.geojson",
        "data/static/aoi/study_area.geojson",
    ]
    for value in candidates:
        if not value:
            continue
        path = _location_path(paths, value)
        if path.exists():
            return path
    raise FileNotFoundError("could not find a source geometry for the AORC SST transposition region")


def _study_footprint_path(config, paths):
    candidates = [
        config.get("smart_ds_evaluation_footprint", {}).get("output"),
        config.get("grid_footprint", {}).get("source"),
        "data/static/aoi/evaluation_footprint.geojson",
        "data/static/aoi/study_area.geojson",
    ]
    for value in candidates:
        if not value:
            continue
        path = _source_location_path(paths, value)
        if path.exists():
            return path
    raise FileNotFoundError("could not find a study footprint for AORC SST plotting")


def _transposition_buffer_km(config, region):
    if region.get("buffer_km") is not None:
        return float(region["buffer_km"])
    if region.get("buffer_m") is not None:
        return float(region["buffer_m"]) / 1000.0
    discovery = config.get("collection", {}).get("usgs_streamgages", {}).get("discovery", {})
    if discovery.get("hydrologic_buffer_km") is not None:
        return float(discovery["hydrologic_buffer_km"])
    return 50.0


def _model_crs(config):
    return (
        config.get("project", {}).get("model_crs")
        or config.get("crs")
        or config.get("sfincs", {}).get("crs")
        or "EPSG:32617"
    )


def _location_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(root) / path
    return Path(paths.get("repo_root", root)) / path


# Notebook-facing helpers

import matplotlib.pyplot as plt

import collect_sources.era5_waves as era5_waves_module
import collect_sources.lcra_hydromet as lcra_hydromet_module
import collect_sources.usgs_streamgages as usgs_streamgages_module
from collect_sources.nwm import soil_moisture_csv_has_variables
from collect_sources.source_artifacts import source_artifact_covers
from collect_sources.usgs_streamgages import (
    active_streamgage_candidate_artifact_ready,
    build_reviewed_streamgage_decisions,
    collect_usgs_streamflow_records,
    write_reviewed_streamgage_network,
)
from design_events.runtime import build_paths
from study_location import define_location


source_artifacts = {
    "cora": ("cora", "boundary_water_level"),
    "usgs_streamgages": ("usgs_streamgages", "active_candidates"),
    "stream_geo_nldi": ("stream_geo_nldi", "river_geometry_lookup"),
    "nwm": ("nwm", "retrospective_hydrologic_state"),
    "aorc_sst": ("aorc_sst", "rainfall_catalog"),
    "era5_waves": ("era5", "snapwave_boundary_forcing"),
}


def _read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CollectSourcesNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    runtime_config: dict
    config: dict
    grid_config: dict
    data_sources: dict
    sfincs_config: dict
    wflow_config: dict
    runtime_paths: dict
    collection: dict
    usgs_streamgages: dict
    candidate_path: Path
    reviewed_network_path: Path
    streamflow_records_cfg: dict
    streamflow_records_path: Path

    def resolve_location_path(self, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.location_root / path


@dataclass(frozen=True)
class StreamgageReviewQa:
    figure: object
    artifact_summary: pd.DataFrame
    gage_domain_summary: pd.DataFrame
    candidate_gages: gpd.GeoDataFrame
    reviewed_gages: gpd.GeoDataFrame
    used_reviewed_gages: gpd.GeoDataFrame
    other_candidate_gages: gpd.GeoDataFrame


@dataclass(frozen=True)
class ReviewedStreamgageNetworkWrite:
    decision_table: pd.DataFrame
    result: dict


def load_runtime(
    location_root,
    *,
    streamgage_review_settings: dict | None = None,
    wflow_domain_review_required: bool | None = None,
) -> CollectSourcesNotebookRuntime:
    """Load one inland collect-sources notebook runtime."""
    location_root = Path(location_root).resolve()
    repo_root = location_root.parents[1]
    definition = define_location(location_root / "config.yaml")
    runtime_config = definition.config
    collection = runtime_config["collection"]
    # Fill in default static_sources keys (bbox, wflow_collection_extent, …) the same
    # way the region-setup runtime does, so locations that don't override them in YAML
    # (greensboro, austin) still resolve the review/QA layers instead of KeyError'ing.
    runtime_config["static_sources"] = static_sources_with_defaults(runtime_config)

    if streamgage_review_settings:
        collection["usgs_streamgages"].update(streamgage_review_settings)
    if wflow_domain_review_required is not None:
        runtime_config["wflow"]["domain_set"]["review_required"] = bool(wflow_domain_review_required)
    runtime_paths = build_paths(runtime_config)

    event_catalog = runtime_config.setdefault("event_catalog", {})
    event_catalog.setdefault("forcing_members", {})
    event_catalog["forcing_members"].setdefault("rainfall", runtime_paths["aorc_sst_rainfall_members_csv"])
    event_catalog["forcing_members"].setdefault("soil_moisture", runtime_paths["nwm_soil_moisture_csv"])
    if runtime_config.get("flood_setting") == "inland":
        event_catalog["forcing_members"].setdefault(
            "streamflow",
            location_root / "data/sources/usgs_streamgages/streamflow_members.csv",
        )

    if "national_hydrography" in collection:
        national_hydrography = collection["national_hydrography"]
        national_hydrography.setdefault("hydromt_basemap", "data/wflow/hydrography/us_hydrography_basemap.nc")
        national_hydrography.setdefault("river_geometry", "data/sources/national_hydrography/nhdplus_hr_river_geometry.gpkg")
        national_hydrography.setdefault("catchments", "data/sources/national_hydrography/nhdplus_hr_catchments.gpkg")
        national_hydrography.setdefault("wflow_soil_parameters", "data/wflow/static/ssurgo_wflow_soil_parameters.nc")

    collection.setdefault("nwm", {}).setdefault("soil_moisture", {}).setdefault("variables", [])

    usgs_streamgages = collection.get("usgs_streamgages", {})
    streamflow_records = usgs_streamgages.get("streamflow_records", {})
    if not isinstance(streamflow_records, dict):
        streamflow_records = {"output": streamflow_records}
    usgs_streamgages["streamflow_records"] = streamflow_records
    usgs_streamgages.setdefault("reviewed_network", runtime_paths["usgs_streamgage_network_geojson"])
    usgs_streamgages.setdefault("candidate_output", runtime_paths["usgs_streamgage_candidates_geojson"])
    usgs_streamgages.setdefault("accept_unreviewed_streamgage_network", False)
    streamflow_records.setdefault("output", "data/sources/usgs_streamgages/streamflow_records.csv")
    streamflow_records.setdefault("service", usgs_streamgages_module._streamflow_service(usgs_streamgages))

    candidate_path = _ensure_location_parent(location_root, usgs_streamgages["candidate_output"])
    reviewed_network_path = _ensure_location_parent(location_root, usgs_streamgages["reviewed_network"])
    streamflow_records_cfg = streamflow_records
    streamflow_records_path = _ensure_location_parent(location_root, streamflow_records_cfg["output"])

    return CollectSourcesNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=repo_root,
        runtime_config=runtime_config,
        config=runtime_config,
        grid_config=runtime_config,
        data_sources=runtime_config,
        sfincs_config=runtime_config,
        wflow_config={"wflow": runtime_config.get("wflow", {})},
        runtime_paths=runtime_paths,
        collection=collection,
        usgs_streamgages=usgs_streamgages,
        candidate_path=candidate_path,
        reviewed_network_path=reviewed_network_path,
        streamflow_records_cfg=streamflow_records_cfg,
        streamflow_records_path=streamflow_records_path,
    )


def source_records(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    collection = runtime.collection
    sources = runtime.data_sources["event_catalog"]["forcing_members"]
    records = {
        "active USGS streamgage candidates": runtime.usgs_streamgages["candidate_output"],
        "reviewed streamgage network": runtime.usgs_streamgages["reviewed_network"],
        "reviewed discharge records": runtime.usgs_streamgages["streamflow_records"]["output"],
        "AORC rainfall members": sources["rainfall"],
        "NWM soil-moisture members": sources["soil_moisture"],
        "STREAM-geo river geometry cache": collection.get("stream_geo_nldi", {}).get(
            "stream_geo_table",
            collection["national_hydrography"].get("stream_geo_table", "data/sources/national_hydrography/stream_geo.parquet"),
        ),
        "NLDI STREAM-geo COMID join cache": collection.get("stream_geo_nldi", {}).get(
            "nldi_lookup_cache",
            collection["national_hydrography"].get(
                "nldi_lookup_cache",
                "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv",
            ),
        ),
        "NHDPlusV2 flowlines for STREAM-geo join": collection.get("stream_geo_nldi", {}).get(
            "nhdplus_v2_flowlines",
            collection["national_hydrography"].get(
                "nhdplus_v2_flowlines",
                "data/sources/national_hydrography/nhdplus_v2_flowlines.gpkg",
            ),
        ),
        "NHDPlus river geometry": collection["national_hydrography"]["river_geometry"],
        "NHDPlus catchments": collection["national_hydrography"]["catchments"],
        "Wflow soil parameters": collection["national_hydrography"]["wflow_soil_parameters"],
    }
    reservoirs_cfg = collection["national_hydrography"].get("reservoirs", {})
    if reservoirs_cfg.get("enabled", False):
        records["NHDPlus Wflow reservoirs"] = reservoirs_cfg.get(
            "output",
            collection["national_hydrography"].get(
                "reservoirs_output",
                "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg",
            ),
        )
        condition_cfg = reservoirs_cfg.get("conditions", {}) or {}
        if condition_cfg.get("enabled", False):
            records["TWDB reservoir condition summary"] = condition_cfg.get(
                "summary_csv",
                "data/sources/twdb_reservoirs/reservoir_condition_summary.csv",
            )
            records["TWDB reservoir condition provenance"] = condition_cfg.get(
                "provenance_json",
                "data/sources/twdb_reservoirs/reservoir_condition_provenance.json",
            )
    if "lcra_hydromet" in collection:
        hydromet = collection["lcra_hydromet"]
        records["supplemental LCRA Hydromet flow sites"] = hydromet.get(
            "output",
            lcra_hydromet_module.DEFAULT_OUTPUT,
        )
        records["supplemental LCRA Hydromet current flow"] = hydromet.get(
            "current_output",
            lcra_hydromet_module.DEFAULT_CURRENT_OUTPUT,
        )
    return _location_record_table(runtime.location_root, records)


def source_role_labels() -> dict[str, str]:
    return {
        "usgs_streamgages": "active records for POT, validation, and handoff",
        "stream_geo_nldi": "STREAM-geo width/depth cache with NLDI COMID lookup provenance",
        "national_hydrography": "USA hydrography and SSURGO pedology for HydroMT-Wflow build sources",
        "lcra_hydromet": "supplemental distinct LCRA/COA flow-gage coverage for Austin review",
        "aorc_sst": "direct rainfall members shared by Wflow and SFINCS",
        "nwm": "antecedent soil-moisture context",
    }


def source_collection_plan_table(collection_plan, source_roles: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(collection_plan.summary_rows()).assign(
        role=lambda frame: frame["source"].map(source_roles)
    )


def review_summary(runtime: CollectSourcesNotebookRuntime) -> pd.Series:
    usgs = runtime.usgs_streamgages
    review = usgs_streamgages_module.streamgage_review_config(usgs)
    if str(review.get("method", "")).lower() == "huc_region":
        return pd.Series(
            {
                "review_required": bool(usgs.get("review_required", True)),
                "accept_unreviewed_streamgage_network": bool(
                    usgs.get("accept_unreviewed_streamgage_network", False)
                ),
                "method": review.get("method"),
                "source_geometry": review.get("source_geometry"),
                "geometry_predicate": review.get("geometry_predicate", "covers"),
                "review_status": review.get("review_status", "accepted_with_warning"),
                "roles": ", ".join(review.get("roles", [])),
                "frequency_basis_from": review.get("frequency_basis_from"),
                "wflow_submodel_id_from": review.get("wflow_submodel_id_from"),
                "sfincs_domain_id_from": review.get("sfincs_domain_id_from"),
            },
            name="streamgage_review",
        )

    policy = usgs.get("review_policy") or {}
    handoff_site_nos = policy.get("handoff_site_nos", {})
    basin_rules = policy.get("basin_rules", [])
    return pd.Series(
        {
            "review_required": bool(usgs.get("review_required", True)),
            "accept_unreviewed_streamgage_network": bool(
                usgs.get("accept_unreviewed_streamgage_network", False)
            ),
            "review_status_default": policy.get("default_review_status"),
            "default_sfincs_domain_id": policy.get("default_sfincs_domain_id"),
            "long_record_years": policy.get("long_record_years"),
            "handoff_site_count": len(handoff_site_nos),
            "basin_rule_count": len(basin_rules),
            "roles": ", ".join(usgs.get("roles", [])),
        },
        name="streamgage_review_policy",
    )


def review_sources(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    review = usgs_streamgages_module.streamgage_review_config(runtime.usgs_streamgages)
    if str(review.get("method", "")).lower() == "huc_region":
        return pd.DataFrame(
            [
                {
                    "method": review.get("method"),
                    "source_geometry": review.get("source_geometry"),
                    "frequency_basis_from": review.get("frequency_basis_from"),
                    "wflow_submodel_id_from": review.get("wflow_submodel_id_from"),
                    "sfincs_domain_id_from": review.get("sfincs_domain_id_from"),
                }
            ]
        )
    rules = (runtime.usgs_streamgages.get("review_policy") or {}).get("basin_rules", [])
    return pd.DataFrame(rules, columns=["contains", "frequency_basis", "wflow_submodel_id"])


def streamgage_basin_rules_table(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    return review_sources(runtime)


def streamgage_review_policy_summary(runtime: CollectSourcesNotebookRuntime) -> pd.Series:
    return review_summary(runtime)


def collect_configured_source_artifacts(
    runtime: CollectSourcesNotebookRuntime,
    collection_plan,
    *,
    skip_existing: bool = True,
    stop_on_error: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Collect missing configured artifacts and return one audit table."""
    collectable_readiness = readiness(runtime)
    prerequisite_result = prepare(runtime.runtime_config, runtime.runtime_paths)
    collection_result = run_collect(
        runtime.runtime_config,
        runtime.runtime_paths,
        collection_plan,
        run_collection=not collectable_readiness["ready"].all(),
        skip_existing=skip_existing,
        stop_on_error=stop_on_error,
        progress=progress,
    )
    return pd.concat(
        [
            collectable_readiness.assign(table="pre_collection_readiness"),
            prerequisite_result.assign(table="collection_prerequisite"),
            collection_result.assign(table="collection_result"),
        ],
        ignore_index=True,
        sort=False,
    )


def refresh_wflow_hydrography_basemap(
    runtime: CollectSourcesNotebookRuntime,
    *,
    force: bool = True,
) -> dict:
    """Refresh only the Wflow HydroMT hydrography basemap for notebook use."""
    from collect_sources.national_hydrography import (
        refresh_wflow_hydrography_basemap as _refresh_wflow_hydrography_basemap,
    )

    collection_plan = plan(runtime.runtime_config, runtime.runtime_paths)
    if not collection_plan.has("national_hydrography"):
        raise KeyError("collection.national_hydrography is not configured for this location")
    return _refresh_wflow_hydrography_basemap(
        collection_plan.settings_for("national_hydrography"),
        skip_existing=not force,
    )


def readiness(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    collection = runtime.collection
    usgs_streamgages = runtime.usgs_streamgages
    national_hydrography = collection["national_hydrography"]
    stream_geo_nldi = collection.get("stream_geo_nldi", {})
    outputs = {
        "streamgage candidates": usgs_streamgages["candidate_output"],
        "STREAM-geo river geometry cache": stream_geo_nldi.get(
            "stream_geo_table",
            national_hydrography.get("stream_geo_table", "data/sources/national_hydrography/stream_geo.parquet"),
        ),
        "NLDI STREAM-geo COMID join cache": stream_geo_nldi.get(
            "nldi_lookup_cache",
            national_hydrography.get(
                "nldi_lookup_cache",
                "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv",
            ),
        ),
        "NHDPlusV2 flowlines for STREAM-geo join": stream_geo_nldi.get(
            "nhdplus_v2_flowlines",
            national_hydrography.get(
                "nhdplus_v2_flowlines",
                "data/sources/national_hydrography/nhdplus_v2_flowlines.gpkg",
            ),
        ),
        "wflow HydroMT hydrography basemap": national_hydrography["hydromt_basemap"],
        "wflow US hydrography river geometry": national_hydrography["river_geometry"],
        "wflow US hydrography catchments": national_hydrography["catchments"],
        "wflow SSURGO soil parameters": national_hydrography["wflow_soil_parameters"],
        "rainfall members": runtime.data_sources["event_catalog"]["forcing_members"]["rainfall"],
        "soil moisture": runtime.data_sources["event_catalog"]["forcing_members"]["soil_moisture"],
    }
    reservoirs_cfg = national_hydrography.get("reservoirs", {})
    if reservoirs_cfg.get("enabled", False):
        outputs["NHDPlus Wflow reservoirs"] = reservoirs_cfg.get(
            "output",
            national_hydrography.get(
                "reservoirs_output",
                "data/sources/national_hydrography/nhdplus_hr_wflow_reservoirs.gpkg",
            ),
        )
        condition_cfg = reservoirs_cfg.get("conditions", {}) or {}
        if condition_cfg.get("enabled", False):
            outputs["TWDB reservoir condition summary"] = condition_cfg.get(
                "summary_csv",
                "data/sources/twdb_reservoirs/reservoir_condition_summary.csv",
            )
            outputs["TWDB reservoir condition provenance"] = condition_cfg.get(
                "provenance_json",
                "data/sources/twdb_reservoirs/reservoir_condition_provenance.json",
            )
    if "lcra_hydromet" in collection:
        hydromet = collection["lcra_hydromet"]
        outputs["supplemental LCRA Hydromet flow sites"] = hydromet.get(
            "output",
            lcra_hydromet_module.DEFAULT_OUTPUT,
        )
    readiness = pd.DataFrame(
        [
            {"artifact": name, "path": str(path), "exists": path.exists()}
            for name, value in outputs.items()
            for path in [_location_path({"location_root": runtime.location_root}, value)]
        ]
    )
    readiness["ready"] = readiness["exists"]
    readiness.loc[readiness["artifact"].eq("streamgage candidates"), "ready"] = (
        active_streamgage_candidate_artifact_ready(runtime.runtime_config, runtime.runtime_paths)
    )
    soil_variables = collection["nwm"]["soil_moisture"]["variables"]
    soil_moisture_path = runtime.resolve_location_path(
        runtime.data_sources["event_catalog"]["forcing_members"]["soil_moisture"]
    )
    readiness.loc[readiness["artifact"].eq("soil moisture"), "ready"] = (
        soil_moisture_csv_has_variables(soil_moisture_path, soil_variables)
    )
    return readiness


def gage_readiness(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    return readiness(runtime).loc[
        lambda frame: frame["artifact"].isin(
            ["streamgage candidates", "reviewed streamgage network", "soil moisture"]
        )
    ]


def write_gage_network(
    runtime: CollectSourcesNotebookRuntime,
    *,
    write_file: bool = True,
) -> ReviewedStreamgageNetworkWrite:
    decisions = build_reviewed_streamgage_decisions(
        runtime.runtime_config,
        runtime.runtime_paths,
    )
    decision_table = pd.DataFrame(decisions)

    if write_file:
        result = write_reviewed_streamgage_network(
            runtime.runtime_config,
            runtime.runtime_paths,
            decisions,
        )
    else:
        accepted_count = int(decision_table["review_status"].str.startswith("accepted").sum())
        result = {
            "status": "review_pending",
            "reviewed_network_geojson": str(runtime.reviewed_network_path),
            "accepted_count": accepted_count,
            "reason": "Review candidate gages, then set write_file=True.",
        }
    return ReviewedStreamgageNetworkWrite(decision_table, result)


def review_layers(runtime: CollectSourcesNotebookRuntime) -> list[dict]:
    return [
        {
            "label": "SMART-DS evaluation footprint",
            "path": runtime.grid_config["smart_ds_evaluation_footprint"]["output"],
            "edgecolor": "black",
            "linestyle": "-",
        },
        {
            "label": "SFINCS coverage bbox",
            "path": runtime.data_sources["static_sources"]["bbox"]["output"],
            "edgecolor": "#dc2626",
            "linestyle": "--",
        },
        {
            "label": "Wflow watershed search domain",
            "path": runtime.data_sources["static_sources"]["wflow_collection_extent"]["watersheds"],
            "edgecolor": "#059669",
            "linestyle": "-.",
        },
        {
            "label": "AORC SST transposition region",
            "path": runtime.collection["aorc_sst"]["transposition_region"]["geometry_file"],
            "edgecolor": "#2563eb",
            "linestyle": ":",
        },
    ]


def plot_review(
    runtime: CollectSourcesNotebookRuntime,
    area_layer_specs: list[dict],
) -> StreamgageReviewQa:
    """Plot reviewed USGS gages over configured evaluation and forcing regions."""
    area_layers = _load_area_layers(runtime, area_layer_specs)
    candidate_gages = _read_gages(runtime.candidate_path)
    reviewed_gages = _read_gages(runtime.reviewed_network_path)
    active_candidate_gages = _active_gages(candidate_gages)
    used_reviewed_gages = _accepted_active_gages(reviewed_gages)
    other_candidate_gages = _other_candidate_gages(active_candidate_gages, used_reviewed_gages)
    figure = _plot_gage_panels(
        runtime.location_name,
        area_layers,
        other_candidate_gages,
        used_reviewed_gages,
    )
    return StreamgageReviewQa(
        figure=figure,
        artifact_summary=_streamgage_artifact_summary(
            runtime,
            area_layer_specs,
            area_layers,
            active_candidate_gages,
            reviewed_gages,
            used_reviewed_gages,
            other_candidate_gages,
        ),
        gage_domain_summary=_gage_domain_summary(
            runtime,
            area_layers,
            active_candidate_gages,
            reviewed_gages,
            used_reviewed_gages,
        ),
        candidate_gages=candidate_gages,
        reviewed_gages=reviewed_gages,
        used_reviewed_gages=used_reviewed_gages,
        other_candidate_gages=other_candidate_gages,
    )


def collect_gage_records(
    runtime: CollectSourcesNotebookRuntime,
    *,
    skip_existing: bool = True,
) -> dict:
    reviewed_streamgage_sites = _accepted_active_site_numbers(runtime.reviewed_network_path)
    streamflow_record_sites = _streamflow_record_site_numbers(runtime.streamflow_records_path)
    missing_sites = sorted(set(reviewed_streamgage_sites) - streamflow_record_sites)
    collect_records = runtime.reviewed_network_path.exists() and (
        not runtime.streamflow_records_path.exists() or bool(missing_sites)
    )

    if collect_records:
        result = collect_usgs_streamflow_records(
            runtime.runtime_config,
            runtime.runtime_paths,
            skip_existing=skip_existing,
        )
        result["missing_sites_before_collection"] = missing_sites
        return result
    if not runtime.reviewed_network_path.exists():
        return {
            "status": "review gated",
            "service": runtime.streamflow_records_cfg["service"],
            "streamflow_records_csv": str(runtime.streamflow_records_path),
            "reason": "Create or accept the reviewed streamgage network before collecting discharge records.",
        }
    return {
        "status": "reused" if runtime.streamflow_records_path.exists() else "missing",
        "site_count": len(streamflow_record_sites),
        "reviewed_site_count": len(reviewed_streamgage_sites),
        "missing_sites_before_collection": missing_sites,
        "service": runtime.streamflow_records_cfg["service"],
        "streamflow_records_csv": str(runtime.streamflow_records_path),
        "reason": "Existing discharge records cover the reviewed streamgage network.",
    }


def summarize_wflow_handoff_review(used_reviewed_gages: gpd.GeoDataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if used_reviewed_gages.empty:
        handoff_review = pd.DataFrame(
            columns=["site_no", "wflow_submodel_id", "sfincs_handoff_id", "review_status"]
        )
    else:
        handoff_review = used_reviewed_gages.drop(columns="geometry").copy()
        if "sfincs_handoff_id" not in handoff_review:
            handoff_review["sfincs_handoff_id"] = None

    handoff_mask = (
        handoff_review.get("sfincs_handoff_id", pd.Series(dtype=object))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    summary = pd.Series(
        {
            "status": "streamgage_review_ready"
            if not handoff_review.empty
            else "review_required_missing_streamgage_network",
            "accepted_reviewed_gages": int(len(handoff_review)),
            "reviewed_streamgage_handoff_tags": int(handoff_mask.sum()),
            "domain_rule": "SFINCS-Wflow handoffs come from stream-boundary crossings; reviewed streamgages support frequency and validation",
        },
        name="wflow_handoff_review",
    )
    return handoff_review, summary


def collection_readiness_table(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    outputs = {
        "streamgage candidates": runtime.usgs_streamgages["candidate_output"],
        "reviewed streamgage network": runtime.usgs_streamgages["reviewed_network"],
        "rainfall members": runtime.data_sources["event_catalog"]["forcing_members"]["rainfall"],
        "reviewed discharge records": runtime.usgs_streamgages["streamflow_records"]["output"],
        "soil moisture": runtime.data_sources["event_catalog"]["forcing_members"]["soil_moisture"],
    }
    readiness = pd.DataFrame(
        [
            {"artifact": name, "path": str(path), "exists": path.exists()}
            for name, value in outputs.items()
            for path in [_location_path({"location_root": runtime.location_root}, value)]
        ]
    )
    readiness["ready_for_catalog"] = readiness["exists"]
    mask = readiness["artifact"].eq("reviewed streamgage network")
    readiness.loc[mask, "ready_for_catalog"] = readiness.loc[mask, "exists"] | bool(
        runtime.usgs_streamgages["accept_unreviewed_streamgage_network"]
    )
    return readiness


def overview() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "driver": "streamflow",
                "source": "USGS active streamgages",
                "event_use": "POT frequency basis and Wflow validation",
            },
            {
                "driver": "rainfall",
                "source": "AORC SST",
                "event_use": "direct rainfall and Wflow precipitation forcing",
            },
            {
                "driver": "soil_moisture",
                "source": "NWM retrospective",
                "event_use": "antecedent state pairing",
            },
        ]
    )


def plan_table(
    plan,
    paths: dict,
    *,
    rerun: bool = False,
) -> pd.DataFrame:
    plan_table = pd.DataFrame(plan.summary_rows())
    will_reuse = [_will_reuse_source(step, paths, rerun=rerun) for step in plan.steps]
    plan_table["will_reuse_existing"] = will_reuse
    plan_table["action"] = plan_table["will_reuse_existing"].map(
        {True: "reuse complete artifact", False: "collect or repair"}
    )
    return plan_table


def summary(config: dict, paths: dict) -> pd.Series:
    return pd.Series(
        {
            "location": paths["location_name"],
            "sources_root": str(paths["location_data_root"] / "sources"),
            "collection_start": config["collection"]["start"],
            "collection_end": config["collection"]["end"],
        },
        name="source_collection",
    )


def _csv_rows(path):
    path = Path(path)
    if not path.exists():
        return None
    return int(len(pd.read_csv(path)))


def _configured_collection_window(config):
    collection = config.get("collection", {})
    if not collection.get("start") or not collection.get("end"):
        return None, None
    return pd.Timestamp(collection["start"]), pd.Timestamp(collection["end"])


def _record_window_issues(record, label, config, start_key="start", end_key="end"):
    expected_start, expected_end = _configured_collection_window(config)
    if expected_start is None or expected_end is None or record is None:
        return []
    issues = []
    actual_start = record.get(start_key)
    actual_end = record.get(end_key)
    if not actual_start or not actual_end:
        issues.append(f"{label} does not declare production window")
        return issues
    actual_start = pd.Timestamp(actual_start)
    actual_end = pd.Timestamp(actual_end)
    if actual_start > expected_start or actual_end < expected_end:
        issues.append(
            f"{label} does not cover production window: "
            f"{actual_start.isoformat()} to {actual_end.isoformat()}, "
            f"expected {expected_start.isoformat()} to {expected_end.isoformat()}"
        )
    return issues


def _source_manifest(paths, filename):
    return _read_json(paths["source_artifacts_root"] / filename)


def _source_artifact_issues(config, paths, filename, label):
    manifest = _source_manifest(paths, filename)
    if manifest is None:
        return [f"missing complete {label} source artifact"]
    issues = []
    if manifest.get("status") != "complete":
        issues.append(f"{label} source artifact status is {manifest.get('status')!r}")
    if manifest.get("metadata", {}).get("smoke") is True:
        issues.append(f"{label} source artifact is smoke-limited")
    issues.extend(_record_window_issues(manifest, f"{label} source artifact", config))
    return issues


def check_aorc_sst_collection(config, paths):
    storms = config.get("collection", {}).get("aorc_sst", {})
    duration = int(storms.get("storm_duration_hours", 72))
    collection_dir = paths["aorc_sst_root"] / paths["location_name"] / f"{duration}hr-events"
    stats_rows = _csv_rows(collection_dir / "storm-stats.csv")
    ranked_rows = _csv_rows(collection_dir / "ranked-storms.csv")
    artifact = _source_manifest(paths, "aorc_sst_rainfall_catalog.json")
    issues = []
    if artifact is None:
        issues.append("missing direct AORC SST source artifact")
    elif artifact.get("status") != "complete":
        issues.append(f"direct AORC SST source artifact status is {artifact.get('status')!r}")
    else:
        issues.extend(_record_window_issues(artifact, "AORC SST source artifact", config))
    if not stats_rows:
        issues.append("missing or empty AORC SST storm-stats.csv")
    if not ranked_rows:
        issues.append("missing or empty AORC SST ranked-storms.csv")
    return {
        "id": "aorc_sst_collection",
        "passed": not issues,
        "issues": issues,
        "details": {
            "status": None if artifact is None else artifact.get("status"),
            "storm_stats_rows": stats_rows or 0,
            "ranked_storm_rows": ranked_rows or 0,
        },
    }


def check_rainfall_catalog(config, paths):
    rainfall_rows = _csv_rows(paths["aorc_sst_rainfall_members_csv"])
    catalog_rows = _csv_rows(paths["event_catalog_csv"])
    audit = _read_json(paths["event_catalog_audit_json"])
    paired_rows = 0
    issues = []
    if not rainfall_rows:
        issues.append("missing or empty rainfall_members.csv")
    if not catalog_rows:
        issues.append("missing or empty event_catalog.csv")
    else:
        catalog = pd.read_csv(paths["event_catalog_csv"])
        required = ["rainfall_source", "rainfall_member_file", "rainfall_member_id"]
        missing_columns = [column for column in required if column not in catalog]
        if missing_columns:
            issues.append(f"event catalog missing rainfall columns: {missing_columns}")
        else:
            paired = catalog[required].notna().all(axis=1) & (catalog[required] != "").all(axis=1)
            paired_rows = int(paired.sum())
            if paired_rows != int(len(catalog)):
                issues.append("event catalog has unpaired rainfall rows")
    if audit is None:
        issues.append("missing event catalog audit")
    elif audit.get("passed") is not True:
        issues.append("event catalog audit did not pass")
    return {
        "id": "rainfall_catalog",
        "passed": not issues,
        "issues": issues,
        "details": {
            "rainfall_member_rows": rainfall_rows or 0,
            "event_catalog_rows": catalog_rows or 0,
            "paired_event_rows": paired_rows,
            "event_catalog_audit_passed": None if audit is None else bool(audit.get("passed")),
        },
    }


def _source_manifest_complete(paths, filename):
    manifest = _source_manifest(paths, filename)
    return manifest is not None and manifest.get("status") == "complete"


def _wave_output_path(config, paths):
    spec = config.get("collection", {}).get("era5_waves", {})
    return _repo_path(paths, spec.get("output_path")) or paths.get("era5_waves_nc")


def _wave_dataset_variables(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return list(ds.data_vars)


def check_wave_forcing(config, paths):
    from collect_sources.era5_waves import era5_wave_short_variables, wave_dataset_covers

    required = bool(config.get("coastal_waves", False))
    spec = config.get("collection", {}).get("era5_waves")
    issues = []
    variables = []
    if not required:
        return {"id": "wave_forcing", "passed": True, "issues": [], "details": {"required": False}}
    if not spec:
        issues.append("missing collection.era5_waves settings")
    elif not spec.get("bbox_wgs84"):
        issues.append("missing era5_waves.bbox_wgs84")
    issues.extend(_source_artifact_issues(config, paths, "era5_snapwave_boundary_forcing.json", "ERA5 wave"))
    output_path = _wave_output_path(config, paths)
    if output_path is None:
        issues.append("missing ERA5 wave output path")
    elif not Path(output_path).exists():
        issues.append(f"missing ERA5 wave NetCDF: {output_path}")
    else:
        try:
            variables = _wave_dataset_variables(output_path)
        except Exception as exc:
            issues.append(f"could not open ERA5 wave NetCDF: {type(exc).__name__}: {exc}")
        else:
            missing = [name for name in era5_wave_short_variables if name not in variables]
            if missing:
                issues.append(f"ERA5 wave NetCDF missing variables: {missing}")
            expected_start, expected_end = _configured_collection_window(config)
            if expected_start is not None and expected_end is not None:
                if not wave_dataset_covers(output_path, expected_start, expected_end):
                    issues.append("ERA5 wave NetCDF does not cover collection window")
    return {
        "id": "wave_forcing",
        "passed": not issues,
        "issues": issues,
        "details": {
            "required": True,
            "variables": [name for name in era5_wave_short_variables if name in variables],
            "wave_netcdf": None if output_path is None else str(output_path),
        },
    }


def check_source_acquisition(config, paths):
    collection = config.get("collection", {})
    rainfall_backend = "aorc_sst" if "aorc_sst" in collection else "not_configured"
    nwm = collection.get("nwm", {})
    streamflow = nwm.get("streamflow", {})
    soil_moisture = nwm.get("soil_moisture", {})
    required_dirs = [paths["outputs_root"], paths["source_artifacts_root"], paths["nwm_root"]]
    if rainfall_backend == "aorc_sst":
        required_dirs.append(paths["aorc_sst_root"])
    issues = []
    for directory in required_dirs:
        if not Path(directory).exists():
            issues.append(f"missing output directory: {directory}")
    issues.extend(_source_artifact_issues(config, paths, "cora_boundary_water_level.json", "CORA"))
    issues.extend(_source_artifact_issues(config, paths, "nwm_retrospective_hydrologic_state.json", "NWM"))
    if rainfall_backend == "not_configured":
        issues.append("AORC SST rainfall collection is not configured")
    if streamflow.get("available") is not False:
        issues.append("configured coastal no-streamflow exception must be explicit")
    if streamflow.get("feature_ids") not in ([], None):
        issues.append("configured coastal no-streamflow feature_ids must be empty")
    if not streamflow.get("reason"):
        issues.append("configured coastal no-streamflow exception needs a reason")
    soil_points = soil_moisture.get("points") or []
    if not soil_points:
        issues.append("NWM soil moisture points are not configured")
    return {
        "id": "source_acquisition",
        "passed": not issues,
        "issues": issues,
        "details": {
            "rainfall_backend": rainfall_backend,
            "streamflow_available": streamflow.get("available"),
            "soil_moisture_point_count": len(soil_points),
            "cora_manifest_complete": _source_manifest_complete(paths, "cora_boundary_water_level.json"),
            "nwm_manifest_complete": _source_manifest_complete(paths, "nwm_retrospective_hydrologic_state.json"),
        },
    }


def write_data_acquisition_readiness(config, paths):
    gates = [
        check_aorc_sst_collection(config, paths),
        check_rainfall_catalog(config, paths),
        check_source_acquisition(config, paths),
    ]
    if config.get("coastal_waves", False):
        gates.append(check_wave_forcing(config, paths))
    audit = {"study_location": paths["location_name"], "passed": all(gate["passed"] for gate in gates), "gates": gates}
    path = Path(paths["data_acquisition_readiness_json"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def readiness_report(config: dict, paths: dict) -> tuple[pd.Series, pd.DataFrame]:

    audit = write_data_acquisition_readiness(config, paths)
    gates = pd.DataFrame(audit.get("gates", []))
    if not gates.empty:
        gates["issues"] = gates["issues"].apply(lambda values: "\n".join(values) if values else "")
    return (
        pd.Series(
            {
                "passed": audit["passed"],
                "report": str(paths["data_acquisition_readiness_json"]),
            },
            name="data_acquisition_readiness",
        ),
        gates,
    )


def aorc_sst_source_summary(config: dict, paths: dict) -> pd.Series:
    aorc_sst = config["collection"].get("aorc_sst", {})
    return pd.Series(
        {
            "source": "direct_aorc_sst",
            "transposition_region": aorc_sst.get("transposition_region", {}).get("geometry_file"),
            "rainfall_members": str(paths["aorc_sst_rainfall_members_csv"]),
            "rainfall_members_exists": paths["aorc_sst_rainfall_members_csv"].exists(),
        },
        name="aorc_sst",
    )


def aorc_sst_params(
    config: dict,
    paths: dict,
    *,
    min_precip_threshold=None,
    decluster_hours=None,
    storm_duration_hours=None,
    check_every_n_hours=None,
    top_n_events=None,
    defer_event_windows=None,
) -> pd.Series:
    """Apply notebook-facing AORC SST overrides into ``config`` and summarize them.

    Selection is threshold-driven POT: the collector keeps every independent storm
    whose footprint-mean depth exceeds ``min_precip_threshold`` (mm over the storm
    window), so the rainfall-member count is data-driven by the threshold and the
    declustering window. ``top_n_events`` is an optional safety cap only. Any value
    left as ``None`` keeps the configured default; the returned Series surfaces the
    parameters a user can retune before Run Collection reads them from ``config``.
    """
    aorc_sst = config.setdefault("collection", {}).setdefault("aorc_sst", {})
    overrides = {
        "min_precip_threshold": min_precip_threshold,
        "decluster_hours": decluster_hours,
        "storm_duration_hours": storm_duration_hours,
        "check_every_n_hours": check_every_n_hours,
        "top_n_events": top_n_events,
        "defer_event_windows": defer_event_windows,
    }
    for key, value in overrides.items():
        if value is not None:
            aorc_sst[key] = value
    return pd.Series(
        {
            "source": "direct_aorc_sst",
            "transposition_region_id": aorc_sst.get("transposition_region", {}).get("id"),
            "transposition_region": aorc_sst.get("transposition_region", {}).get("geometry_file"),
            "start_date": aorc_sst.get("start_date", config["collection"].get("start")),
            "end_date": aorc_sst.get("end_date", config["collection"].get("end")),
            "selection": "threshold-driven POT (every independent storm above threshold)",
            "min_precip_threshold_mm": aorc_sst.get("min_precip_threshold"),
            "storm_duration_hours": aorc_sst.get("storm_duration_hours", 72),
            "decluster_hours": aorc_sst.get("decluster_hours"),
            "check_every_n_hours": aorc_sst.get("check_every_n_hours"),
            "top_n_events_safety_cap": aorc_sst.get("top_n_events"),
            "defer_event_windows": bool(aorc_sst.get("defer_event_windows", False)),
            "event_meteo_enabled": bool((aorc_sst.get("event_meteo") or {}).get("enabled", False)),
            "rainfall_members": str(paths["aorc_sst_rainfall_members_csv"]),
            "rainfall_members_exists": paths["aorc_sst_rainfall_members_csv"].exists(),
        },
        name="aorc_sst",
    )


def collect_aorc_sst_event_windows(
    config: dict,
    paths: dict,
    collection_plan,
    *,
    skip_existing=True,
) -> pd.Series:
    from collect_sources.aorc_sst import collect_aorc_sst_event_windows as _collect_event_windows

    if not collection_plan.has("aorc_sst"):
        raise KeyError("aorc_sst is not configured in the source collection plan")
    result = _collect_event_windows(
        collection_plan.settings_for("aorc_sst"),
        skip_existing=skip_existing,
    )
    return pd.Series(
        {
            "source": "aorc_sst_event_windows",
            "status": "collected",
            "ranked_rows": result["ranked_rows"],
            "rainfall_member_rows": result["rainfall_member_rows"],
            "event_window_count": result["event_window_count"],
            "ranked_storms_csv": str(result["ranked_storms_csv"]),
            "event_windows_dir": str(result["event_windows_dir"]),
            "source_artifact_json": str(result["source_artifact_json"]),
        },
        name="aorc_sst_event_windows",
    )


def repair_aorc_sst_event_window_meteo(
    config: dict,
    paths: dict,
    collection_plan,
    *,
    skip_existing=True,
) -> pd.Series:
    from collect_sources.aorc_sst import repair_aorc_sst_event_window_meteo as _repair_event_meteo

    if not collection_plan.has("aorc_sst"):
        raise KeyError("aorc_sst is not configured in the source collection plan")
    result = _repair_event_meteo(
        collection_plan.settings_for("aorc_sst"),
        skip_existing=skip_existing,
    )
    return pd.Series(
        {
            "source": "aorc_sst_event_window_meteo_repair",
            "status": "repaired" if result["repaired_count"] else "current",
            "ranked_rows": result["ranked_rows"],
            "current_count": result["current_count"],
            "repaired_count": result["repaired_count"],
            "missing_count": result["missing_count"],
            "incomplete_count": result["incomplete_count"],
            "event_windows_dir": str(result["event_windows_dir"]),
        },
        name="aorc_sst_event_window_meteo_repair",
    )


def aorc_sst_event_window_readiness(
    config: dict,
    paths: dict,
    collection_plan,
) -> pd.Series:
    from collect_sources.aorc_sst import event_window_variable_readiness

    if not collection_plan.has("aorc_sst"):
        raise KeyError("aorc_sst is not configured in the source collection plan")
    result = event_window_variable_readiness(collection_plan.settings_for("aorc_sst"))
    return pd.Series(
        {
            "source": "aorc_sst_event_window_readiness",
            "status": "ready" if result["ready_count"] == result["ranked_rows"] else "incomplete",
            "ranked_rows": result["ranked_rows"],
            "ready_count": result["ready_count"],
            "missing_count": result["missing_count"],
            "incomplete_count": result["incomplete_count"],
            "event_windows_dir": str(result["event_windows_dir"]),
        },
        name="aorc_sst_event_window_readiness",
    )


def stream_sources(config: dict, paths: dict) -> pd.Series:
    collection = config["collection"]
    stream_geo_nldi = collection.get("stream_geo_nldi", {})
    national_hydrography = collection.get("national_hydrography", {})
    stream_geo_table = stream_geo_nldi.get(
        "stream_geo_table",
        national_hydrography.get("stream_geo_table", "data/sources/national_hydrography/stream_geo.parquet"),
    )
    nldi_lookup_cache = stream_geo_nldi.get(
        "nldi_lookup_cache",
        national_hydrography.get("nldi_lookup_cache", "data/sources/national_hydrography/nldi_stream_geo_comid_cache.csv"),
    )
    nhdplus_v2_flowlines = stream_geo_nldi.get(
        "nhdplus_v2_flowlines",
        national_hydrography.get("nhdplus_v2_flowlines", "data/sources/national_hydrography/nhdplus_v2_flowlines.gpkg"),
    )
    table_path = _source_location_path(paths, stream_geo_table)
    cache_path = _source_location_path(paths, nldi_lookup_cache)
    nhdplus_v2_path = _source_location_path(paths, nhdplus_v2_flowlines)
    manifest_path = paths["source_artifacts_root"] / "stream_geo_nldi_sources.json"
    return pd.Series(
        {
            "source": "STREAM-geo/NLDI",
            "stream_geo_table": stream_geo_table,
            "stream_geo_table_exists": table_path.exists(),
            "nldi_lookup_cache": nldi_lookup_cache,
            "nldi_lookup_cache_exists": cache_path.exists(),
            "nhdplus_v2_flowlines": nhdplus_v2_flowlines,
            "nhdplus_v2_flowlines_exists": nhdplus_v2_path.exists(),
            "stream_geo_join_method": national_hydrography.get("stream_geo_join_method", "attribute_transfer"),
            "nldi_role": "COMID lookup provenance",
            "manifest": str(manifest_path),
            "manifest_exists": manifest_path.exists(),
        },
        name="stream_geo_nldi",
    )


def reservoir_sources(config: dict, paths: dict) -> pd.Series:
    reservoirs = (config.get("collection", {}).get("national_hydrography", {}) or {}).get("reservoirs", {}) or {}
    conditions = reservoirs.get("conditions", {}) or {}
    summary_path = _source_location_path(
        paths,
        conditions.get("summary_csv", "data/sources/twdb_reservoirs/reservoir_condition_summary.csv"),
    )
    provenance_path = _source_location_path(
        paths,
        conditions.get("provenance_json", "data/sources/twdb_reservoirs/reservoir_condition_provenance.json"),
    )
    return pd.Series(
        {
            "enabled": bool(conditions.get("enabled", False)),
            "provider": conditions.get("provider", "twdb_water_data_for_texas"),
            "period_suffix": conditions.get("period_suffix", "-1year"),
            "statistic": conditions.get("statistic", "median"),
            "reservoir_slugs": ", ".join(sorted((conditions.get("reservoir_slugs") or {}).values())),
            "summary_csv": str(summary_path),
            "summary_exists": summary_path.exists(),
            "provenance_json": str(provenance_path),
            "provenance_exists": provenance_path.exists(),
        },
        name="reservoir_conditions",
    )


def reservoir_condition_table(config: dict, paths: dict) -> pd.DataFrame:
    reservoirs = (config.get("collection", {}).get("national_hydrography", {}) or {}).get("reservoirs", {}) or {}
    conditions = reservoirs.get("conditions", {}) or {}
    summary_path = _source_location_path(
        paths,
        conditions.get("summary_csv", "data/sources/twdb_reservoirs/reservoir_condition_summary.csv"),
    )
    if not summary_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(summary_path)
    display_columns = [
        "waterbody_name",
        "twdb_slug",
        "condition_status",
        "condition_statistic",
        "condition_period_start",
        "condition_period_end",
        "Depth_avg",
        "reservoir_storage_acft",
        "surface_area_acres",
        "percent_full",
        "condition_reason",
    ]
    return frame[[column for column in display_columns if column in frame.columns]]


def soil_sources(config: dict, paths: dict) -> pd.Series:
    nwm = config["collection"].get("nwm", {})
    soil = nwm.get("soil_moisture", {})
    return pd.Series(
        {
            "version": nwm.get("version"),
            "streamflow_available": nwm.get("streamflow", {}).get("available"),
            "streamflow_reason": nwm.get("streamflow", {}).get("reason"),
            "soil_moisture_points": len(soil.get("points", [])),
            "soil_moisture_variables": soil.get("variables", []),
            "soil_moisture_zarr": soil.get("zarr"),
            "soil_moisture_csv": str(paths["nwm_soil_moisture_csv"]),
            "soil_moisture_exists": paths["nwm_soil_moisture_csv"].exists(),
            "soil_moisture_has_requested_variables": soil_moisture_csv_has_variables(
                paths["nwm_soil_moisture_csv"],
                soil.get("variables", []),
            ),
        },
        name="nwm_soil_moisture",
    )


def usgs_streamgage_source_summary(runtime: CollectSourcesNotebookRuntime) -> pd.Series:
    return pd.Series(
        {
            "candidate_gages": str(runtime.candidate_path),
            "candidate_gages_exist": runtime.candidate_path.exists(),
            "reviewed_network": str(runtime.reviewed_network_path),
            "reviewed_network_exists": runtime.reviewed_network_path.exists(),
            "streamflow_records": str(runtime.streamflow_records_path),
            "streamflow_records_exist": runtime.streamflow_records_path.exists(),
        },
        name="usgs_streamgages",
    )


def discover_gages(runtime: CollectSourcesNotebookRuntime) -> tuple[pd.Series, pd.Series]:
    usgs = runtime.usgs_streamgages
    discovery = usgs.get("discovery", {})
    records = usgs.get("streamflow_records", {})
    query_parameters = usgs_streamgages_module._nwis_site_query_params(usgs, runtime.runtime_paths)

    return (
        pd.Series(
            {
                "provider": "USGS NWIS",
                "site_service_url": usgs_streamgages_module.USGS_SITE_SERVICE_URL,
                "parameter_cd": query_parameters.get("parameterCd"),
                "site_status": query_parameters.get("siteStatus"),
                "data_types": query_parameters.get("hasDataTypeCd"),
                "bbox": query_parameters.get("bBox"),
                "search_geometry": discovery.get("search_geometry"),
                "hydrologic_buffer_km": discovery.get("hydrologic_buffer_km"),
                "active_records_only": usgs.get("active_records_only", True),
                "candidate_output": usgs.get("candidate_output"),
                "candidate_output_exists": runtime.candidate_path.exists(),
            },
            name="usgs_active_streamgage_discovery",
        ),
        pd.Series(
            {
                "records_service": records.get("service", "dv"),
                "records_output": records.get("output"),
                "records_output_exists": runtime.streamflow_records_path.exists(),
                "request_timeout_seconds": records.get(
                    "request_timeout_seconds",
                    discovery.get("request_timeout_seconds", 60),
                ),
                "stat_cd": records.get("stat_cd", "00003"),
            },
            name="usgs_reviewed_discharge_records",
        ),
    )


def plot_sst_region(config: dict, paths: dict, *, zoom: int = 9, basemap: bool = True):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        import contextily as ctx
    except ImportError:
        ctx = None

    region_file = config["collection"]["aorc_sst"]["transposition_region"]["geometry_file"]
    region_path = _source_location_path(paths, region_file)
    footprint_path = _study_footprint_path(config, paths)

    region = gpd.read_file(region_path).to_crs("EPSG:4326")
    footprint = gpd.read_file(footprint_path).to_crs("EPSG:4326")
    region_web = region.to_crs(epsg=3857)
    footprint_web = footprint.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(8, 8))
    region_web.plot(ax=ax, facecolor="#d95f0226", edgecolor="#d95f02", linewidth=2.5)
    footprint_web.boundary.plot(ax=ax, color="#1b9e77", linewidth=2.0)

    xmin, ymin, xmax, ymax = region_web.total_bounds
    pad_x = (xmax - xmin) * 0.05
    pad_y = (ymax - ymin) * 0.05
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)

    if basemap and ctx is not None:
        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=zoom, attribution_size=7)
        except Exception as exc:
            ax.text(0.01, 0.01, f"basemap unavailable: {type(exc).__name__}", transform=ax.transAxes, fontsize=8)

    ax.legend(
        handles=[
            Patch(facecolor="#d95f0226", edgecolor="#d95f02", label="Storm transposition region"),
            Line2D([0], [0], color="#1b9e77", linewidth=2.0, label="Study grid footprint"),
        ],
        loc="lower left",
    )
    ax.set_axis_off()
    ax.set_title(f"{paths['location_name'].title()} stochastic storm transposition region")
    fig.tight_layout()
    return fig, ax


def plot_collected_sst_geography(config: dict, paths: dict):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    study_area = gpd.read_file(_study_footprint_path(config, paths)).to_crs("EPSG:4326")
    sst_region = gpd.read_file(
        _source_location_path(paths, config["collection"]["aorc_sst"]["transposition_region"]["geometry_file"])
    ).to_crs("EPSG:4326")
    rainfall_members_csv = paths.get("aorc_sst_rainfall_members_csv")
    if rainfall_members_csv is None:
        rainfall_members_csv = (
            Path(paths["location_root"])
            / config.get("event_catalog", {})
            .get("forcing_members", {})
            .get("rainfall", "data/sources/aorc_sst/rainfall_members.csv")
        )
    rainfall = _read_csv(rainfall_members_csv, parse_dates=["storm_start", "storm_end"])

    historical_lon = _first_column(rainfall, ["historical_footprint_center_lon", "centroid_lon"])
    historical_lat = _first_column(rainfall, ["historical_footprint_center_lat", "centroid_lat"])
    target_lon = _first_column(rainfall, ["target_footprint_center_lon", "transposed_centroid_lon"])
    target_lat = _first_column(rainfall, ["target_footprint_center_lat", "transposed_centroid_lat"])
    value_column = _first_column(rainfall, ["mean_precip_mm", "max_precip_mm", "mean", "max"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sst_region.plot(ax=ax, facecolor="#f4a26133", edgecolor="#d95f02", linewidth=1.8)
    study_area.boundary.plot(ax=ax, color="black", linewidth=1.2)

    plotted_targets = False
    if not rainfall.empty and historical_lon and historical_lat:
        history = rainfall.dropna(subset=[historical_lon, historical_lat]).copy()
        historical_points = gpd.GeoDataFrame(
            history,
            geometry=gpd.points_from_xy(history[historical_lon], history[historical_lat]),
            crs="EPSG:4326",
        )
        historical_points.plot(
            ax=ax,
            column=value_column,
            cmap="viridis",
            markersize=24,
            alpha=0.82,
            edgecolor="black",
            linewidth=0.2,
            legend=value_column is not None,
            legend_kwds={"label": "event precipitation [mm]", "shrink": 0.62},
        )
        plotted_targets = True

    vector_columns = [historical_lon, historical_lat, target_lon, target_lat]
    if not rainfall.empty and all(column is not None for column in vector_columns):
        vectors = rainfall.dropna(subset=[historical_lon, historical_lat, target_lon, target_lat]).copy()
        if not vectors.empty:
            segments = [
                [
                    (float(row[historical_lon]), float(row[historical_lat])),
                    (float(row[target_lon]), float(row[target_lat])),
                ]
                for _, row in vectors.iterrows()
            ]
            ax.add_collection(
                LineCollection(
                    segments,
                    colors="#2b2b2b",
                    linewidths=0.45,
                    alpha=0.22,
                    zorder=3,
                )
            )
            target_points = gpd.GeoDataFrame(
                vectors,
                geometry=gpd.points_from_xy(vectors[target_lon], vectors[target_lat]),
                crs="EPSG:4326",
            )
            target_points.drop_duplicates(subset=[target_lon, target_lat]).plot(
                ax=ax,
                marker="*",
                color="#d7191c",
                edgecolor="white",
                linewidth=0.55,
                markersize=120,
                zorder=5,
            )
            plotted_targets = True

    ax.legend(
        handles=[
            Patch(facecolor="#f4a26133", edgecolor="#d95f02", label="AORC SST region"),
            Patch(facecolor="none", edgecolor="black", label="study area"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#3f8f7f",
                markeredgecolor="black",
                markersize=8,
                label="winning historical footprint center",
            ),
            Line2D(
                [0],
                [0],
                color="#2b2b2b",
                linewidth=1.2,
                alpha=0.55,
                label="field transposition vector",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#d7191c",
                markeredgecolor="white",
                markersize=11,
                label="study footprint center",
            ),
        ],
        loc="best",
    )
    if not plotted_targets:
        ax.text(
            0.5,
            0.04,
            "rainfall member transposition columns not found",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    xmin, ymin, xmax, ymax = sst_region.total_bounds
    sx = xmax - xmin
    sy = ymax - ymin
    ax.set_xlim(xmin - 0.06 * sx, xmax + 0.06 * sx)
    ax.set_ylim(ymin - 0.06 * sy, ymax + 0.06 * sy)
    ax.set_title("AORC SST collection geography and realized transpositions")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig, ax


def plot_nwm_soil_moisture(config: dict, paths: dict):
    soil_moisture = _read_csv(paths["nwm_soil_moisture_csv"], parse_dates=["time"])
    requested = list(config["collection"].get("nwm", {}).get("soil_moisture", {}).get("variables", []))
    available = [name for name in requested if name in soil_moisture.columns]
    legacy = _first_column(soil_moisture, ["SOIL_M", "soil_m", "soil_moisture"])
    if legacy and legacy not in available:
        available.append(legacy)
    missing = [name for name in requested if name not in soil_moisture.columns]

    status = pd.Series(
        {
            "requested_variables": requested,
            "available_variables": available,
            "missing_variables": missing,
            "csv": str(paths["nwm_soil_moisture_csv"]),
        },
        name="nwm_soil_moisture_status",
    )
    if soil_moisture.empty or not available:
        return None, status

    monthly = soil_moisture.groupby("time")[available].mean().resample("MS").mean()
    fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    monthly.plot(ax=ax, linewidth=0.9)
    ax.set_title("NWM soil moisture state")
    ax.set_xlabel("")
    ax.set_ylabel("soil moisture / saturation")
    ax.legend(title="variable", loc="best", fontsize=8)
    return fig, status


def plot_aorc_sst_rainfall(paths: dict):
    rainfall = _read_csv(paths["aorc_sst_rainfall_members_csv"], parse_dates=["storm_start", "storm_end"])
    if rainfall.empty:
        return None
    max_column = _first_column(rainfall, ["max_precip_mm", "max_precip_in", "max"])
    mean_column = _first_column(rainfall, ["mean_precip_mm", "mean_precip_in", "mean"])
    if max_column is None or mean_column is None:
        raise KeyError("rainfall members need max and mean precipitation columns")
    unit = "mm"
    if "precip_units" in rainfall and rainfall["precip_units"].notna().any():
        unit = str(rainfall["precip_units"].dropna().iloc[0])
    elif str(max_column).endswith("_in"):
        unit = "in"
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    rainfall.plot.scatter(x="rank", y=max_column, ax=axes[0], color="#d95f02", s=16, alpha=0.7)
    axes[0].set_title("AORC SST rainfall member maxima")
    axes[0].set_xlabel("rank")
    axes[0].set_ylabel(f"precipitation, {unit}")
    rainfall[[mean_column, max_column]].plot.hist(ax=axes[1], bins=20, alpha=0.65)
    axes[1].set_title("Rainfall distribution")
    axes[1].set_xlabel(f"precipitation, {unit}")
    return fig


def plot_cora_boundary_water_level(paths: dict):
    waterlevel = _read_csv(paths["waterlevel_csv"], parse_dates=["time"])
    if waterlevel.empty:
        return None
    value_column = _first_column(waterlevel, ["value", "zeta", "water_level", "waterlevel"])
    fig, ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
    waterlevel.set_index("time")[value_column].plot(ax=ax, color="#2166ac", linewidth=0.6)
    ax.set_title("CORA boundary water level")
    ax.set_xlabel("")
    ax.set_ylabel("m MSL")
    return fig, ax


def plot_era5_waves(paths: dict):
    import xarray as xr

    wave_path = Path(paths["era5_waves_nc"])
    if not wave_path.exists():
        return None
    with xr.open_dataset(wave_path) as ds:
        time_name = _first_column(
            pd.DataFrame(columns=list(ds.coords) + list(ds.dims)), ["valid_time", "time"]
        )
        wave_vars = [name for name in ["swh", "pp1d", "mwd", "wdw"] if name in ds.data_vars]
        if not wave_vars:
            return None
        fig, axes = plt.subplots(
            len(wave_vars), 1, figsize=(10, 1.8 * len(wave_vars)), sharex=True, constrained_layout=True
        )
        axes = [axes] if len(wave_vars) == 1 else axes
        for ax, name in zip(axes, wave_vars):
            spatial_dims = [dim for dim in ds[name].dims if dim != time_name]
            series = ds[name].mean(dim=spatial_dims).resample({time_name: "MS"}).mean()
            series.to_pandas().plot(ax=ax, linewidth=0.8)
            ax.set_title(f"ERA5 {name}")
            ax.set_xlabel("")
    return fig


def plot_usgs_streamgage_network(runtime: CollectSourcesNotebookRuntime):
    area_specs = review_layers(runtime)
    qa = plot_review(runtime, area_specs)
    return qa.figure, qa.artifact_summary, qa.gage_domain_summary


def _will_reuse_source(step, paths: dict, *, rerun: bool) -> bool:
    if rerun:
        return False
    if step.name == "national_hydrography":
        manifest = paths["source_artifacts_root"] / "national_hydrography_wflow_sources.json"
        return manifest.exists()
    if step.name not in source_artifacts:
        return False
    source, kind = source_artifacts[step.name]
    manifest_covers = source_artifact_covers(paths, source, kind, step.start, step.end)
    if step.name == "era5_waves":
        output_path = era5_waves_module._wave_output_path(paths, step.spec)
        return bool(manifest_covers and era5_waves_module.wave_dataset_covers(output_path, step.start, step.end))
    if step.name == "nwm":
        variables = step.spec.get("soil_moisture", {}).get("variables", [])
        return bool(
            manifest_covers
            and paths["nwm_streamflow_csv"].exists()
            and soil_moisture_csv_has_variables(paths["nwm_soil_moisture_csv"], variables)
        )
    if step.name == "usgs_streamgages":
        candidate_path = _source_location_path(paths, step.spec.get("candidate_output", paths["usgs_streamgage_candidates_geojson"]))
        return bool(manifest_covers and candidate_path.exists())
    return bool(manifest_covers)


def _source_location_path(paths: dict, value) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return paths["location_root"] / path
    return paths["repo_root"] / path


def _read_csv(path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


def _first_column(frame, names):
    return next((name for name in names if name in frame.columns), None)


def _location_record_table(location_root: Path, records: dict[str, object]) -> pd.DataFrame:
    rows = []
    for label, value in records.items():
        path = Path(value)
        resolved = path if path.is_absolute() else Path(location_root) / path
        rows.append(
            {
                "record": label,
                "configured": str(value),
                "location_root_syntax": f'location_root / "{value}"' if not path.is_absolute() else str(value),
                "exists": resolved.exists(),
            }
        )
    return pd.DataFrame(rows)


def _ensure_location_parent(location_root: Path, value) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else location_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _load_area_layers(runtime: CollectSourcesNotebookRuntime, area_layer_specs: list[dict]) -> list[dict]:
    area_layers = []
    for spec in area_layer_specs:
        layer_path = runtime.resolve_location_path(spec["path"])
        if layer_path.exists():
            layer = gpd.read_file(layer_path).to_crs("EPSG:4326")
            area_layers.append({**spec, "resolved_path": layer_path, "layer": layer})
    return area_layers


def _read_gages(path: Path) -> gpd.GeoDataFrame:
    if path.exists():
        return gpd.read_file(path).to_crs("EPSG:4326")
    return gpd.GeoDataFrame(
        columns=["site_no", "site_name", "status", "review_status", "roles", "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def _active_gages(gages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "status" not in gages:
        return gages.copy()
    return gages.loc[gages["status"].fillna("").astype(str).str.lower().eq("active")].copy()


def _accepted_active_gages(gages: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "review_status" not in gages:
        return gages.iloc[0:0].copy()
    accepted = gages["review_status"].fillna("").astype(str).str.lower().isin(
        ["accepted", "accepted_with_warning"]
    )
    return _active_gages(gages).loc[accepted].copy()


def _other_candidate_gages(
    active_candidate_gages: gpd.GeoDataFrame,
    used_reviewed_gages: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    reviewed_site_nos = set(used_reviewed_gages.get("site_no", pd.Series(dtype=str)).astype(str))
    if "site_no" not in active_candidate_gages:
        return active_candidate_gages.copy()
    return active_candidate_gages.loc[
        ~active_candidate_gages["site_no"].astype(str).isin(reviewed_site_nos)
    ].copy()


def _plot_gage_panels(location_name, area_layers, other_candidate_gages, used_reviewed_gages):
    plot_panels = area_layers or [
        {"label": "USGS streamgage source extent", "layer": None, "edgecolor": "#6b7280", "linestyle": "-"}
    ]
    fig, axes = plt.subplots(1, len(plot_panels), figsize=(6 * len(plot_panels), 6), squeeze=False)
    for ax, entry in zip(axes.ravel(), plot_panels):
        if entry["layer"] is not None:
            entry["layer"].boundary.plot(
                ax=ax,
                color=entry["edgecolor"],
                linestyle=entry["linestyle"],
                linewidth=1.4,
                label=entry["label"],
            )
        _plot_gages(ax, other_candidate_gages, color="#9ca3af", marker="o", label="other candidate gages")
        _plot_gages(ax, used_reviewed_gages, color="#be123c", marker="^", label="used reviewed gages")
        ax.set_title(entry["label"])
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.legend(loc="best")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"{location_name.title()} USGS gages over review regions")
    fig.tight_layout()
    return fig


def _plot_gages(ax, gages, *, color, marker, label):
    if gages.empty:
        return
    gages.plot(
        ax=ax,
        color=color,
        marker=marker,
        markersize=70 if marker == "^" else 30,
        alpha=0.55 if marker == "o" else 1.0,
        edgecolor="white" if marker == "^" else None,
        linewidth=0.6 if marker == "^" else None,
        label=label,
        zorder=5 if marker == "^" else 4,
    )


def _streamgage_artifact_summary(
    runtime,
    area_layer_specs,
    area_layers,
    active_candidate_gages,
    reviewed_gages,
    used_reviewed_gages,
    other_candidate_gages,
):
    layer_by_label = {entry["label"]: entry["layer"] for entry in area_layers}
    rows = [
        {
            "artifact": "candidate active gages",
            "label": "all active candidates",
            "path": str(runtime.candidate_path),
            "exists": runtime.candidate_path.exists(),
            "feature_count": len(active_candidate_gages),
        },
        {
            "artifact": "reviewed streamgage network",
            "label": "reviewed artifact",
            "path": str(runtime.reviewed_network_path),
            "exists": runtime.reviewed_network_path.exists(),
            "feature_count": len(reviewed_gages),
        },
    ]
    rows.extend(
        {
            "artifact": "area layer",
            "label": spec["label"],
            "path": str(runtime.resolve_location_path(spec["path"])),
            "exists": runtime.resolve_location_path(spec["path"]).exists(),
            "feature_count": len(layer_by_label.get(spec["label"], [])),
        }
        for spec in area_layer_specs
    )
    rows.extend(
        [
            {
                "artifact": "used_reviewed_gages",
                "label": "active accepted gages",
                "path": str(runtime.reviewed_network_path),
                "exists": not used_reviewed_gages.empty,
                "feature_count": len(used_reviewed_gages),
            },
            {
                "artifact": "other_candidate_gages",
                "label": "active candidates not in reviewed network",
                "path": str(runtime.candidate_path),
                "exists": not other_candidate_gages.empty,
                "feature_count": len(other_candidate_gages),
            },
        ]
    )
    return pd.DataFrame(rows)


def _gage_domain_summary(
    runtime,
    area_layers,
    active_candidate_gages,
    reviewed_gages,
    used_reviewed_gages,
):
    watershed_entry = next((entry for entry in area_layers if entry["label"] == "Wflow watershed search domain"), None)
    watershed_geom = watershed_entry["layer"].geometry.union_all() if watershed_entry else None
    search_geometry_value = runtime.usgs_streamgages.get("discovery", {}).get("search_geometry")
    search_geometry_path = runtime.resolve_location_path(search_geometry_value) if search_geometry_value else None
    search_bbox_geom = _search_bbox_geometry(search_geometry_path)
    return pd.DataFrame(
        [
            {
                "check": "configured NWIS search bbox covers Wflow watershed",
                "value": bool(
                    search_bbox_geom is not None
                    and watershed_geom is not None
                    and search_bbox_geom.covers(watershed_geom)
                ),
                "path": str(search_geometry_path) if search_geometry_path else "missing search_geometry",
            },
            _inside_watershed_row("active candidate gages inside Wflow watershed", active_candidate_gages, watershed_geom, runtime.candidate_path),
            _inside_watershed_row("reviewed gages inside Wflow watershed", reviewed_gages, watershed_geom, runtime.reviewed_network_path),
            _inside_watershed_row("active accepted reviewed gages inside Wflow watershed", used_reviewed_gages, watershed_geom, runtime.reviewed_network_path),
        ]
    )


def _search_bbox_geometry(search_geometry_path):
    if search_geometry_path and search_geometry_path.exists():
        search_geometry_layer = gpd.read_file(search_geometry_path).to_crs("EPSG:4326")
        return search_geometry_layer.geometry.union_all().envelope
    return None


def _inside_watershed_row(check, gages, watershed_geom, path):
    if watershed_geom is None:
        inside_count = 0
    else:
        inside_count = int(gages.geometry.map(lambda point: bool(watershed_geom.covers(point))).sum())
    return {
        "check": check,
        "value": f"{inside_count} of {len(gages)}",
        "path": str(path),
    }


def _accepted_active_site_numbers(reviewed_network_path: Path) -> list[str]:
    if not reviewed_network_path.exists():
        return []
    gages = gpd.read_file(reviewed_network_path)
    return sorted(_accepted_active_gages(gages).get("site_no", pd.Series(dtype=str)).astype(str))


def _streamflow_record_site_numbers(streamflow_records_path: Path) -> set[str]:
    if not streamflow_records_path.exists():
        return set()
    return set(
        pd.read_csv(streamflow_records_path, dtype={"site_no": str}, usecols=["site_no"])["site_no"].astype(str)
    )


# Short notebook-facing API. Implementation names stay descriptive inside this
# module; notebooks can read as compact workflow steps.
