from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import pandas as pd

from fiat_runs import diagnostics, risk, risk_native, validate
from fiat_runs._env import env_ready
from fiat_runs.build_model import apply_ground, build_model, model_ready
from fiat_runs.hazard import WaterLevelRasterizer
from fiat_runs.run import run_event
from sfincs_runs import diagnostics as sfincs_diagnostics


@dataclass(frozen=True)
class RiskPreflight:
    inputs: pd.Series
    completed_by_scenario: pd.Series
    probability: pd.Series
    runs: pd.DataFrame
    selected: pd.DataFrame
    weights: pd.DataFrame
    total_rate: float


@dataclass(frozen=True)
class ExposureReview:
    summary: pd.Series
    by_occupancy: pd.DataFrame
    buildings: object
    exposure_csv: Path
    vulnerability_csv: Path


@dataclass(frozen=True)
class FiatEventRun:
    rasterizer: WaterLevelRasterizer
    per_event_damage: pd.DataFrame
    event_outcomes: pd.DataFrame
    coverage: pd.Series
    status_counts: pd.Series
    elapsed_seconds: float


@dataclass(frozen=True)
class EventDamageReview:
    damage_audit: pd.DataFrame
    damage_by_depth: pd.DataFrame
    damage_by_use: pd.DataFrame
    top_assets: pd.DataFrame
    negative_depth_curves: pd.DataFrame
    review_event: pd.Series


@dataclass(frozen=True)
class BuildingRiskReview:
    building_risk: object
    top_neighborhoods: pd.DataFrame
    building_risk_csv: Path
    building_risk_gpkg: Path


@dataclass(frozen=True)
class EadReview:
    audit: dict
    ead_table: pd.DataFrame
    exceedance: pd.DataFrame


def risk_preflight(
    paths: dict,
    *,
    catalog_csv,
    metadata_json,
    event_limit: int | None = None,
    rerun: bool = False,
) -> RiskPreflight:
    """Validate FIAT risk inputs and select completed SFINCS events for this notebook run."""
    inputs = pd.Series(
        {
            "env_ready": env_ready(),
            "fiat_model_built": model_ready(paths),
            "event_catalog": Path(catalog_csv).exists(),
            "catalog_risk_metadata": Path(metadata_json).exists(),
            "storage_root": str(paths["storage_root"]),
        },
        name="stage_inputs",
    )
    if not inputs["env_ready"]:
        raise RuntimeError(
            "conda 'fiat' env is missing hydromt_fiat/delft_fiat. Create it: "
            "mamba create -n fiat -c conda-forge python=3.12 gdal delft_fiat && "
            "conda run -n fiat pip install hydromt_fiat"
        )
    if not (inputs["event_catalog"] and inputs["catalog_risk_metadata"]):
        raise FileNotFoundError(
            "Missing event_catalog.csv or catalog_risk_metadata.json. Run 03_build_event_catalog "
            "(including the catalog_risk_metadata export cell) first."
        )

    runs = sfincs_diagnostics.completed_sfincs_runs(paths["storage_root"])
    if runs.empty:
        raise RuntimeError("No completed SFINCS run_outputs found. Run 05 + the cluster job first.")

    weights = risk.load_catalog_weights(catalog_csv)
    total_rate = risk.total_rate_from_metadata(metadata_json)
    selected = runs if event_limit is None else runs.groupby("design_scenario", group_keys=False).head(event_limit)
    coverage_preview = selected[["event_id", "design_scenario"]].merge(weights, on="event_id", how="inner")
    probability = pd.Series(
        {
            "event_limit": "all_completed" if event_limit is None else event_limit,
            "selected_completed_runs": len(selected),
            "weighted_catalog_events": len(weights),
            "selected_probability_weight": float(coverage_preview["probability_weight"].sum()),
            "total_rate_per_year": total_rate,
            "rerun": rerun,
        },
        name="catalog_probability_preflight",
    )
    return RiskPreflight(
        inputs=inputs,
        completed_by_scenario=runs.groupby("design_scenario").size().rename("completed_sfincs_runs"),
        probability=probability,
        runs=runs,
        selected=selected,
        weights=weights,
        total_rate=total_rate,
    )


def build_or_reuse_model(config: dict, paths: dict) -> pd.Series:
    """Build the FIAT model if needed and re-ground exposure to the SFINCS DEM."""
    model_root = Path(paths["fiat_model_root"])
    receipt = build_model(config, paths) if not model_ready(paths) else {"reused": True, "model_root": str(model_root)}
    ground = apply_ground(config, paths)
    return pd.Series(
        {
            "model_root": str(model_root),
            "n_exposure_assets": receipt.get("n_exposure_assets"),
            "structures_grounded": ground["structures_grounded"],
            "ground_ft_median": round(ground["ground_ft_median"], 2),
            "exposure": "NSI",
            "vulnerability": "HAZUS IWR curves",
        },
        name="fiat_model",
    )


def exposure_review(model_root, *, show: bool = True) -> ExposureReview:
    """Summarize and map the FIAT exposure/vulnerability model."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    model_root = Path(model_root)
    exposure_csv = model_root / "exposure" / "exposure.csv"
    vulnerability_csv = model_root / "vulnerability" / "vulnerability_curves.csv"
    buildings = diagnostics.load_exposure_buildings(model_root)
    summary = diagnostics.exposure_summary(buildings)
    by_occupancy = (
        buildings.groupby("primary_object_type", dropna=False)
        .agg(
            n_buildings=("object_id", "size"),
            max_structure_damage=("max_damage_structure", "sum"),
            max_content_damage=("max_damage_content", "sum"),
        )
        .sort_values("max_structure_damage", ascending=False)
        .head(12)
    )
    if show:
        display(summary)
        display(by_occupancy)
        fig, ax = plt.subplots(figsize=(8, 7))
        diagnostics.plot_building_exposure(
            buildings,
            ax=ax,
            basemap_style="osm",
            title="HydroMT-FIAT NSI building exposure (points)",
        )
        plt.tight_layout()
        display(fig)
        plt.close(fig)
    return ExposureReview(summary, by_occupancy, buildings, exposure_csv, vulnerability_csv)


def run_fiat_events(
    model_root,
    paths: dict,
    selected: pd.DataFrame,
    *,
    catalog_csv,
    weights: pd.DataFrame,
    total_rate: float,
    rerun: bool = False,
) -> FiatEventRun:
    """Export water-level rasters, run/reuse FIAT event damages, and join outcomes."""
    if selected.empty:
        raise RuntimeError("No completed SFINCS runs were selected for FIAT.")
    rasterizer = WaterLevelRasterizer(selected.iloc[0]["map_path"])
    rows = []
    t0 = time.time()
    for row in selected.itertuples(index=False):
        haz = paths["fiat_hazard_root"] / row.design_scenario / f"{row.event_id}.tif"
        out_dir = paths["fiat_risk_root"] / row.design_scenario / row.event_id
        gpkg = out_dir / "spatial.gpkg"
        if rerun or not gpkg.exists():
            rasterizer.export(row.map_path, haz)
            result = run_event(model_root, haz, out_dir, event_id=row.event_id)
            status = "ran"
        else:
            result = _reused_event_damage(paths["fiat_risk_root"], row.event_id, row.design_scenario, gpkg)
            status = "reused"
        result["design_scenario"] = row.design_scenario
        result["fiat_status"] = status
        rows.append(result)

    per_event_damage = pd.DataFrame(rows)
    outcomes = sfincs_diagnostics.event_outcome_table(selected, catalog_csv, weights, total_rate, per_event_damage)
    coverage = sfincs_diagnostics.outcome_coverage(outcomes, weights)
    return FiatEventRun(
        rasterizer=rasterizer,
        per_event_damage=per_event_damage,
        event_outcomes=outcomes,
        coverage=coverage,
        status_counts=per_event_damage["fiat_status"].value_counts().rename("fiat_event_status"),
        elapsed_seconds=time.time() - t0,
    )


def show_event_run_summary(run: FiatEventRun, *, top_n: int = 15) -> None:
    from IPython.display import display

    print(f"processed {len(run.per_event_damage)} FIAT events in {run.elapsed_seconds:.0f}s")
    display(run.coverage)
    display(run.status_counts)
    display(
        run.event_outcomes.sort_values("total_damage", ascending=False)[
            [
                "event_id",
                "design_scenario",
                "storm_type",
                "severity_band",
                "probability_weight",
                "annual_rate",
                "sample_rp_years",
                "total_damage",
                "n_assets_damaged",
            ]
        ].head(top_n)
    )


def event_damage_review(
    paths: dict,
    selected: pd.DataFrame,
    per_event_damage: pd.DataFrame,
    *,
    exposure_csv,
    vulnerability_csv,
    show: bool = True,
) -> EventDamageReview:
    """Audit the highest-damage FIAT events and plot the top event."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    review = per_event_damage.sort_values("total_damage", ascending=False).iloc[0]
    review_run = selected[
        (selected["event_id"] == review.event_id) & (selected["design_scenario"] == review.design_scenario)
    ].iloc[0]
    review_damage = diagnostics.event_damage(
        paths["fiat_risk_root"], review.event_id, scenario=review.design_scenario, exposure_csv=exposure_csv
    )
    audit_rows = []
    for row in per_event_damage.sort_values("total_damage", ascending=False).head(12).itertuples(index=False):
        damage = diagnostics.event_damage(paths["fiat_risk_root"], row.event_id, scenario=row.design_scenario, exposure_csv=exposure_csv)
        audit_rows.append({"event_id": row.event_id, "design_scenario": row.design_scenario, **diagnostics.damage_summary(damage)})
    damage_audit = pd.DataFrame(audit_rows)
    depth = diagnostics.damage_by_depth(review_damage)
    use = diagnostics.damage_by_use(review_damage)
    top = diagnostics.top_assets(review_damage, n=12)
    negative_curves = diagnostics.nonzero_negative_depth_curves(vulnerability_csv).head(12)
    if show:
        display(
            damage_audit[
                [
                    "event_id",
                    "design_scenario",
                    "total_damage",
                    "n_assets_damaged",
                    "top10_damage_share_pct",
                    "median_inun_depth_ft",
                    "p95_inun_depth_ft",
                    "low_depth_damage",
                    "low_depth_damage_pct",
                    "median_loss_ratio",
                    "p95_loss_ratio",
                ]
            ]
        )
        display(depth)
        display(use.head(8))
        display(top)
        display(negative_curves)
        _plot_top_event_review(review, review_run, review_damage, depth, use)
    return EventDamageReview(damage_audit, depth, use, top, negative_curves, pd.Series(review))


def building_risk_review(paths: dict, event_outcomes: pd.DataFrame, *, exposure_csv, show: bool = True) -> BuildingRiskReview:
    """Build annualized per-building risk outputs and maps for the base scenario."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    building_risk = diagnostics.building_risk(paths["fiat_risk_root"], event_outcomes, scenario="base", exposure_csv=exposure_csv)
    top_neighborhoods = diagnostics.top_neighborhoods(building_risk, n=4)
    risk_root = Path(paths["fiat_risk_root"])
    risk_root.mkdir(parents=True, exist_ok=True)
    building_risk_csv = risk_root / "building_annualized_risk.csv"
    building_risk_gpkg = risk_root / "building_annualized_risk.gpkg"
    building_risk.drop(columns="geometry").to_csv(building_risk_csv, index=False)
    building_risk.to_file(building_risk_gpkg, driver="GPKG")
    if show:
        display(top_neighborhoods)
        display(
            building_risk.sort_values("annual_damage", ascending=False)[
                [
                    "object_id",
                    "primary_object_type",
                    "annual_damage",
                    "damage_aep",
                    "max_event_damage",
                    "weighted_mean_inun_depth_ft",
                    "max_inun_depth_ft",
                    "aggregation_label:Census Blockgroup",
                ]
            ].head(15)
        )
        _plot_building_risk(building_risk, top_neighborhoods)
        plt.close("all")
    return BuildingRiskReview(building_risk, top_neighborhoods, building_risk_csv, building_risk_gpkg)


def ead_review(per_event_damage: pd.DataFrame, weights: pd.DataFrame, total_rate: float, *, show: bool = True) -> EadReview:
    """Compute weighted-event EAD and the base-scenario exceedance curve."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    audit = risk.ead_audit(per_event_damage, weights, total_rate, expected_event_count=len(weights))
    ead_table = risk.ead_by_scenario(per_event_damage, weights, total_rate)
    base_damage = per_event_damage[per_event_damage["design_scenario"] == "base"]
    exceedance = risk.exceedance(base_damage, weights, total_rate)
    if show:
        display(pd.Series({k: v for k, v in audit.items() if k != "ead_by_scenario"}, name="ead_audit"))
        display(ead_table)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(exceedance["exceedance_rate_per_year"], exceedance["total_damage"] / 1e6, marker=".", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("annual exceedance rate [1/yr]")
        ax.set_ylabel("event damage [$M]")
        ax.set_title("Damage vs annual exceedance (base) - area under curve approximates EAD")
        ax.grid(True, alpha=0.3)
        display(fig)
        plt.close(fig)
    return EadReview(audit, ead_table, exceedance)


def native_rp_crosscheck(model_root, rasterizer, paths: dict, catalog_csv, ead_table: pd.DataFrame) -> tuple[dict, pd.Series | None]:
    """Run FIAT-native return-period risk and compare it with weighted-event EAD."""
    rp_events = risk_native.select_rp(catalog_csv)
    native = risk_native.run_rp_risk(
        model_root,
        rasterizer,
        paths["storage_root"],
        rp_events,
        paths["fiat_risk_root"] / "_native_rp",
        paths["fiat_hazard_root"] / "_native_rp",
    )
    base_ead = ead_table.loc[ead_table["design_scenario"] == "base", "ead"]
    crosscheck = None
    if len(base_ead):
        crosscheck = pd.Series(
            {
                "weighted_event_ead_base": float(base_ead.iloc[0]),
                "native_rp_ead": native["ead"],
                "ratio_native_over_weighted": native["ead"] / float(base_ead.iloc[0]) if base_ead.iloc[0] else None,
            },
            name="ead_crosscheck",
        )
    return native, crosscheck


def historical_validation(model_root, rasterizer, paths: dict, catalog_csv) -> dict:
    """Run historical-tail FIAT validation where historical SFINCS maps exist."""
    return validate.validate_history(
        model_root,
        rasterizer,
        paths["storage_root"],
        catalog_csv,
        paths["fiat_risk_root"] / "historical",
        paths["fiat_hazard_root"] / "historical",
    )


def write_risk_outputs(
    paths: dict,
    *,
    per_event_damage: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    ead_table: pd.DataFrame,
    exceedance: pd.DataFrame,
    audit: dict,
    native: dict,
    hist: dict,
    coverage: pd.Series,
    total_rate: float,
    event_limit: int | None,
    rerun: bool,
    building_risk_csv,
    building_risk_gpkg,
) -> pd.Series:
    """Write notebook risk products and a compact summary receipt."""
    risk_root = Path(paths["fiat_risk_root"])
    risk_root.mkdir(parents=True, exist_ok=True)
    per_event_damage.to_csv(risk_root / "per_event_damage.csv", index=False)
    event_outcomes.to_csv(risk_root / "fiat_event_outcomes_with_catalog_weights.csv", index=False)
    ead_table.to_csv(risk_root / "ead_by_scenario.csv", index=False)
    exceedance.to_csv(risk_root / "damage_exceedance_base.csv", index=False)
    (risk_root / "ead_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (risk_root / "ead_crosscheck.json").write_text(json.dumps(native, indent=2, default=str), encoding="utf-8")
    if hist["n_run"]:
        hist["damages"].to_csv(risk_root / "historical_validation.csv", index=False)
    summary = {
        "event_count": int(len(per_event_damage)),
        "event_limit": event_limit,
        "rerun": rerun,
        "scenarios": sorted(per_event_damage["design_scenario"].unique().tolist()),
        "total_rate_per_year": total_rate,
        "synthetic_weight_sum": audit["synthetic_weight_sum"],
        "probability_weight_coverage": float(coverage["weight_coverage"]),
        "covered_probability_weight": float(coverage["covered_probability_weight"]),
        "weighted_ead_by_scenario": ead_table.to_dict("records"),
        "native_rp_ead": native.get("ead"),
        "historical_runs": hist["n_run"],
        "building_annualized_risk_csv": str(building_risk_csv),
        "building_annualized_risk_gpkg": str(building_risk_gpkg),
    }
    (risk_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return pd.Series(
        {
            "per_event_damage": str(risk_root / "per_event_damage.csv"),
            "fiat_event_outcomes": str(risk_root / "fiat_event_outcomes_with_catalog_weights.csv"),
            "building_annualized_risk": str(building_risk_gpkg),
            "ead_by_scenario": str(risk_root / "ead_by_scenario.csv"),
            "summary": str(risk_root / "summary.json"),
        },
        name="risk_outputs",
    )


def _reused_event_damage(risk_root, event_id: str, scenario: str, gpkg: Path) -> dict:
    event_damage = diagnostics.event_damage(risk_root, event_id, scenario=scenario)
    total = pd.to_numeric(event_damage.get("total_damage"), errors="coerce").fillna(0.0)
    return {
        "event_id": event_id,
        "total_damage": float(total.sum()),
        "n_assets": int(len(event_damage)),
        "n_assets_damaged": int((total > 0).sum()),
        "output_gpkg": str(gpkg),
    }


def _plot_top_event_review(review, review_run, review_damage, depth, use) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display

    fig, ax = plt.subplots(figsize=(8, 7))
    ax, mesh = diagnostics.plot_flood_depth(
        review_run.map_path,
        ax=ax,
        basemap_style="dark",
        land_min_elev_m=-0.5,
        title=f"{review.event_id}: land-focused masked SFINCS flood depth",
    )
    fig.colorbar(mesh, ax=ax, shrink=0.9).set_label("Flood Depth (feet)")
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    diagnostics.plot_damaged_assets(
        review_damage,
        ax=ax,
        basemap_style="dark",
        title=f"{review.event_id}: damaged FIAT building assets",
    )
    plt.tight_layout()
    display(fig)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(depth["depth_band_ft"].astype(str), depth["damage"] / 1e6, color="#2b8cbe")
    axes[0].set_ylabel("Damage ($M)")
    axes[0].set_title("Damage by FIAT inundation depth")
    axes[0].tick_params(axis="x", rotation=45)
    occupancy = use.head(6)
    axes[1].bar(occupancy["primary_object_type"].astype(str), occupancy["damage"] / 1e6, color="#f03b20")
    axes[1].set_ylabel("Damage ($M)")
    axes[1].set_title("Damage by NSI occupancy")
    plt.tight_layout()
    display(fig)
    plt.close(fig)


def _plot_building_risk(building_risk, top_neighborhoods: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import display

    for metric, title in [
        ("annual_damage", "Annualized building damage from completed weighted catalog"),
        ("damage_aep", "Annual probability of nonzero building damage"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 7))
        diagnostics.plot_risk(building_risk, metric=metric, ax=ax, basemap_style="dark", title=title)
        plt.tight_layout()
        display(fig)
        plt.close(fig)

    if not top_neighborhoods.empty:
        count = min(2, len(top_neighborhoods))
        fig, axes = plt.subplots(1, count, figsize=(7 * count, 6), squeeze=False)
        for ax, neighborhood in zip(axes[0], top_neighborhoods["aggregation_label:Census Blockgroup"].head(2)):
            diagnostics.plot_risk(
                building_risk,
                metric="annual_damage",
                ax=ax,
                basemap_style="osm",
                neighborhood=neighborhood,
                title=f"Affected neighborhood proxy: {neighborhood}",
            )
        plt.tight_layout()
        display(fig)
        plt.close(fig)
