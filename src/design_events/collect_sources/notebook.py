from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from design_events import source_artifacts as source_artifacts_module
from design_events.collect_sources import era5_waves as era5_waves_module
from design_events.collect_sources import usgs_streamgages as usgs_streamgages_module
from design_events.collect_sources.nwm import soil_moisture_csv_has_variables
from design_events.collect_sources.plan import build_source_collection_plan
from design_events.collect_sources.prerequisites import prepare_collection_prerequisites
from design_events.collect_sources.run_collect import run_collect
from design_events.collect_sources.usgs_streamgages import (
    active_streamgage_candidate_artifact_ready,
    build_reviewed_streamgage_decisions,
    collect_usgs_streamflow_records,
    write_reviewed_streamgage_network,
)
from design_events.config import build_paths, load_runtime
from study_location import define_location
from wflow_runs.notebook import exists_table


source_artifacts = {
    "cora": ("cora", "boundary_water_level"),
    "usgs_streamgages": ("usgs_streamgages", "active_candidates"),
    "nwm": ("nwm", "retrospective_hydrologic_state"),
    "aorc_sst": ("aorc_sst", "rainfall_catalog"),
    "era5_waves": ("era5", "snapwave_boundary_forcing"),
}


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


def load_collect_sources_notebook_runtime(
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

    if streamgage_review_settings:
        collection["usgs_streamgages"].update(streamgage_review_settings)
    if wflow_domain_review_required is not None:
        runtime_config["wflow"]["domain_set"]["review_required"] = bool(wflow_domain_review_required)
    runtime_paths = build_paths(runtime_config)

    usgs_streamgages = collection["usgs_streamgages"]
    candidate_path = _ensure_location_parent(location_root, usgs_streamgages["candidate_output"])
    reviewed_network_path = _ensure_location_parent(location_root, usgs_streamgages["reviewed_network"])
    streamflow_records_cfg = usgs_streamgages["streamflow_records"]
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
        wflow_config={"wflow": runtime_config["wflow"]},
        runtime_paths=runtime_paths,
        collection=collection,
        usgs_streamgages=usgs_streamgages,
        candidate_path=candidate_path,
        reviewed_network_path=reviewed_network_path,
        streamflow_records_cfg=streamflow_records_cfg,
        streamflow_records_path=streamflow_records_path,
    )


def source_record_location_table(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    collection = runtime.collection
    sources = runtime.data_sources["event_catalog"]["forcing_members"]
    records = {
        "active USGS streamgage candidates": runtime.usgs_streamgages["candidate_output"],
        "reviewed streamgage network": runtime.usgs_streamgages["reviewed_network"],
        "reviewed discharge records": runtime.usgs_streamgages["streamflow_records"]["output"],
        "AORC rainfall members": sources["rainfall"],
        "NWM soil-moisture members": sources["soil_moisture"],
        "NHDPlus river geometry": collection["national_hydrography"]["river_geometry"],
        "NHDPlus catchments": collection["national_hydrography"]["catchments"],
        "Wflow soil parameters": collection["national_hydrography"]["wflow_soil_parameters"],
    }
    return _location_record_table(runtime.location_root, records)


def source_role_labels() -> dict[str, str]:
    return {
        "usgs_streamgages": "active records for POT, validation, and handoff",
        "national_hydrography": "USA hydrography and SSURGO pedology for HydroMT-Wflow build sources",
        "aorc_sst": "direct rainfall members shared by Wflow and SFINCS",
        "nwm": "antecedent soil-moisture context",
    }


def source_collection_plan_table(collection_plan, source_roles: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(collection_plan.summary_rows()).assign(
        role=lambda frame: frame["source"].map(source_roles)
    )


def streamgage_review_summary(runtime: CollectSourcesNotebookRuntime) -> pd.Series:
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


def streamgage_review_source_table(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
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
    return streamgage_review_source_table(runtime)


def streamgage_review_policy_summary(runtime: CollectSourcesNotebookRuntime) -> pd.Series:
    return streamgage_review_summary(runtime)


def collect_configured_source_artifacts(
    runtime: CollectSourcesNotebookRuntime,
    collection_plan,
    *,
    skip_existing: bool = True,
    stop_on_error: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """Collect missing configured artifacts and return one audit table."""
    collectable_readiness = source_collection_readiness(runtime)
    prerequisite_result = prepare_collection_prerequisites(runtime.runtime_config, runtime.runtime_paths)
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


def source_collection_readiness(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    collection = runtime.collection
    usgs_streamgages = runtime.usgs_streamgages
    national_hydrography = collection["national_hydrography"]
    outputs = {
        "streamgage candidates": usgs_streamgages["candidate_output"],
        "wflow HydroMT hydrography basemap": national_hydrography["hydromt_basemap"],
        "wflow US hydrography river geometry": national_hydrography["river_geometry"],
        "wflow US hydrography catchments": national_hydrography["catchments"],
        "wflow SSURGO soil parameters": national_hydrography["wflow_soil_parameters"],
        "rainfall members": runtime.data_sources["event_catalog"]["forcing_members"]["rainfall"],
        "soil moisture": runtime.data_sources["event_catalog"]["forcing_members"]["soil_moisture"],
    }
    readiness = exists_table(runtime.location_root, outputs)
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


def streamgage_source_readiness(runtime: CollectSourcesNotebookRuntime) -> pd.DataFrame:
    return source_collection_readiness(runtime).loc[
        lambda frame: frame["artifact"].isin(
            ["streamgage candidates", "reviewed streamgage network", "soil moisture"]
        )
    ]


def build_or_write_reviewed_streamgage_network(
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


def streamgage_review_area_layer_specs(runtime: CollectSourcesNotebookRuntime) -> list[dict]:
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


def plot_streamgage_review_regions(
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


def collect_or_reuse_reviewed_streamflow_records(
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
    readiness = exists_table(runtime.location_root, outputs)
    readiness["ready_for_catalog"] = readiness["exists"]
    mask = readiness["artifact"].eq("reviewed streamgage network")
    readiness.loc[mask, "ready_for_catalog"] = readiness.loc[mask, "exists"] | bool(
        runtime.usgs_streamgages["accept_unreviewed_streamgage_network"]
    )
    return readiness


def collected_data_overview() -> pd.DataFrame:
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


def source_collection_plan_with_reuse_table(
    plan,
    paths: dict,
    *,
    source_skip_existing: bool = True,
) -> pd.DataFrame:
    plan_table = pd.DataFrame(plan.summary_rows())
    will_reuse = [_will_reuse_source(step, paths, source_skip_existing=source_skip_existing) for step in plan.steps]
    plan_table["will_reuse_existing"] = will_reuse
    plan_table["action"] = plan_table["will_reuse_existing"].map(
        {True: "reuse complete artifact", False: "collect or repair"}
    )
    return plan_table


def source_collection_runtime_summary(config: dict, paths: dict) -> pd.Series:
    return pd.Series(
        {
            "location": paths["location_name"],
            "sources_root": str(paths["location_data_root"] / "sources"),
            "collection_start": config["collection"]["start"],
            "collection_end": config["collection"]["end"],
        },
        name="source_collection",
    )


def source_collection_readiness_report(config: dict, paths: dict) -> tuple[pd.Series, pd.DataFrame]:
    from design_events.readiness import write_data_acquisition_readiness

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


def nwm_soil_moisture_source_summary(config: dict, paths: dict) -> pd.Series:
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


def usgs_streamgage_discovery_summary(runtime: CollectSourcesNotebookRuntime) -> tuple[pd.Series, pd.Series]:
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
    footprint_file = config["grid_footprint"]["source"]
    region_path = _source_location_path(paths, region_file)
    footprint_path = _source_location_path(paths, footprint_file)

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
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    study_area = gpd.read_file(_source_location_path(paths, config["grid_footprint"]["source"])).to_crs("EPSG:4326")
    sst_region = gpd.read_file(
        _source_location_path(paths, config["collection"]["aorc_sst"]["transposition_region"]["geometry_file"])
    ).to_crs("EPSG:4326")
    rainfall = _read_csv(paths["aorc_sst_rainfall_members_csv"], parse_dates=["storm_start", "storm_end"])

    lon_column = _first_column(rainfall, ["centroid_lon", "transposed_centroid_lon"])
    lat_column = _first_column(rainfall, ["centroid_lat", "transposed_centroid_lat"])
    value_column = _first_column(rainfall, ["max_precip_in", "mean_precip_in", "max", "mean"])

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    sst_region.plot(ax=ax, facecolor="#f4a26133", edgecolor="#d95f02", linewidth=1.8)
    study_area.boundary.plot(ax=ax, color="black", linewidth=1.2)

    if not rainfall.empty and lon_column and lat_column:
        rainfall_points = gpd.GeoDataFrame(
            rainfall.dropna(subset=[lon_column, lat_column]),
            geometry=gpd.points_from_xy(rainfall[lon_column], rainfall[lat_column]),
            crs="EPSG:4326",
        )
        rainfall_points.plot(
            ax=ax,
            column=value_column,
            cmap="inferno_r",
            markersize=24,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.15,
            legend=value_column is not None,
            legend_kwds={"label": "72h rainfall magnitude", "shrink": 0.62},
        )

    ax.legend(
        handles=[
            Patch(facecolor="#f4a26133", edgecolor="#d95f02", label="AORC SST region"),
            Patch(facecolor="none", edgecolor="black", label="study area"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#c51b7d",
                markersize=8,
                label="rainfall transposition targets",
            ),
        ],
        loc="best",
    )
    ax.set_title("Collected source geography")
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


def plot_usgs_streamgage_network(runtime: CollectSourcesNotebookRuntime):
    area_specs = streamgage_review_area_layer_specs(runtime)
    qa = plot_streamgage_review_regions(runtime, area_specs)
    return qa.figure, qa.artifact_summary, qa.gage_domain_summary


def _will_reuse_source(step, paths: dict, *, source_skip_existing: bool) -> bool:
    if not source_skip_existing:
        return False
    if step.name == "national_hydrography":
        manifest = paths["source_artifacts_root"] / "national_hydrography_wflow_sources.json"
        return manifest.exists()
    if step.name not in source_artifacts:
        return False
    source, kind = source_artifacts[step.name]
    manifest_covers = source_artifacts_module.source_artifact_covers(paths, source, kind, step.start, step.end)
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
    for _, gage in gages.iterrows():
        site_no = str(gage.get("site_no", ""))
        if site_no:
            ax.annotate(site_no, (gage.geometry.x, gage.geometry.y), xytext=(4, 4), textcoords="offset points", fontsize=8)


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
