"""HydroMT-style case façade for the distribution artifact pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .core import CasePaths, read_json, write_json


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("YAML config requires PyYAML, or provide a JSON config") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _coerce_utc(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass
class DistributionCase:
    """One auditable distribution-grid workspace.

    The method names match HydroMT setup/update style. Each method reads and
    writes explicit artifacts below :class:`CasePaths`; the return value is a
    small report or manifest suitable for notebook display and CI gates.
    """

    paths: CasePaths
    location_id: str = "case"
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DistributionCase":
        config = _load_config(config_path)
        root = Path(config.get("root") or Path(config_path).parent).resolve()
        project = config.get("project") or {}
        grid = config.get("grid") or {}
        location_id = str(project.get("name") or config.get("location_id") or root.name)
        paths = CasePaths.from_root(root, power_grid=grid.get("power_grid_root", "data/power_grid"))
        return cls(paths=paths, location_id=location_id, config=config)

    # HydroMT-style setup aliases -------------------------------------------------

    def setup_asset_registry(self, **kwargs: Any) -> dict[str, int]:
        return self.build_registry(**kwargs)

    def setup_grid_dataset(self, **kwargs: Any) -> dict[str, Any]:
        return self.export_grid_dataset(**kwargs)

    def setup_event_states(self, **kwargs: Any) -> dict[str, Any]:
        return self.add_event_states(**kwargs)

    def setup_resilience(self, **kwargs: Any) -> dict[str, Any]:
        return self.add_resilience_layers(**kwargs)

    def setup_switches(self, **kwargs: Any) -> dict[str, Any]:
        return self.place_switches(**kwargs)

    def setup_onm_export(self, **kwargs: Any) -> dict[str, Any]:
        return self.export_onm(**kwargs)

    def setup_audit(self, **kwargs: Any) -> dict[str, Any]:
        return self.audit(**kwargs)

    # Pipeline stages ------------------------------------------------------------

    def build_baseline(
        self,
        *,
        parcels: Sequence[Any] | None = None,
        source_anchor: Mapping[str, Any] | None = None,
        equipment_catalog_path: str | Path | None = None,
        gdm_json: str | Path | None = None,
        opendss_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Build a baseline native SHIFT/GDM system and export it.

        Data pull and graph construction use SHIFT native APIs.  If ``parcels``
        is not supplied, parcels are fetched through ``shift.parcels_from_location``
        using the configured source-area geometry.
        """

        from .baseline import (
            build_shift_distribution_system,
            export_baseline_system,
            fetch_parcels_native,
            load_equipment_catalog,
            source_anchors,
            source_area,
        )

        area = source_area(self.config)
        anchors = [source_anchor] if source_anchor is not None else source_anchors(
            self.config,
            location_root=self.paths.root,
            source_area_geometry=area.geometry,
        )
        if not anchors:
            raise ValueError("at least one reviewed source anchor is required")
        anchor = anchors[0]
        parcel_rows = list(parcels) if parcels is not None else list(fetch_parcels_native(area.geometry) or [])
        catalog = load_equipment_catalog(equipment_catalog_path)
        system = build_shift_distribution_system(
            name=self.location_id,
            parcels=parcel_rows,
            source_longitude=float(anchor["lon"]),
            source_latitude=float(anchor["lat"]),
            equipment_catalog=catalog,
            **kwargs,
        )
        return export_baseline_system(
            system,
            gdm_json=gdm_json or self.paths.power_grid / "baseline_gdm.json",
            opendss_dir=opendss_dir or self.paths.opendss / self.location_id,
        )


    def build_registry(self, *, opendss_dir: str | Path | None = None, output_dir: str | Path | None = None) -> dict[str, int]:
        from .registry import build_registry

        return build_registry(Path(opendss_dir or self.paths.opendss), Path(output_dir or self.paths.registry))

    def control_registry(
        self,
        *,
        raw_registry_dir: str | Path,
        output_dir: str | Path | None = None,
        max_tie_distance_m: float = 100.0,
        min_tie_bus_line_degree: int = 2,
    ) -> dict[str, int]:
        from . import dataset

        dataset.location_id = self.location_id
        return dataset.control_registry(
            raw_registry_dir,
            output_dir or self.paths.registry,
            max_tie_distance_m=max_tie_distance_m,
            min_tie_bus_line_degree=min_tie_bus_line_degree,
        )

    def export_grid_dataset(self, *, registry_dir: str | Path | None = None, output_dir: str | Path | None = None, debug_csv: bool = False) -> dict[str, Any]:
        from . import dataset

        dataset.location_id = self.location_id
        return dataset.export_base(Path(registry_dir or self.paths.registry), Path(output_dir or self.paths.augmented), debug_csv=debug_csv)

    def add_event_states(self, *, output_dir: str | Path | None = None, **kwargs: Any) -> dict[str, Any]:
        from . import dataset

        dataset.location_id = self.location_id
        return dataset.export_stage_a2(Path(output_dir or self.paths.augmented), **kwargs)

    def add_resilience_layers(
        self,
        *,
        critical_facilities_path: str | Path | None = None,
        augmented_dir: str | Path | None = None,
        registry_dir: str | Path | None = None,
        use_oedi_load_profiles: bool = False,
        assign_nearest_when_outside_radius: bool = False,
    ) -> dict[str, Any]:
        """Build facilities, load matches, load profiles, and Layer-1 DER rows."""

        from .facilities import (
            build_load_matches,
            load_bus_electrical_metadata,
            load_critical_facilities,
            validate_load_matches,
            write_critical_facilities_artifact,
            write_load_matches,
        )
        from .profiles import load_inputs
        from .der import build_der_inventory, validate_der_assignments, write_der_inventory

        augmented = Path(augmented_dir or self.paths.augmented)
        registry = Path(registry_dir or self.paths.registry)
        facility_path = Path(critical_facilities_path or (self.config.get("resilience") or {}).get("critical_facilities", ""))
        if not facility_path.exists():
            raise FileNotFoundError(f"critical facility source is missing: {facility_path}")

        facilities = load_critical_facilities(facility_path, location_name=self.location_id)
        write_critical_facilities_artifact(facilities, augmented / "critical_facilities.parquet")
        facility_rows = pd.DataFrame(facilities.drop(columns=["geometry"], errors="ignore")).to_dict("records")
        assets = pd.read_parquet(augmented / "assets.parquet").to_dict("records")
        control_units = pd.read_parquet(augmented / "control_units.parquet").to_dict("records")
        loads = pd.read_csv(registry / "loads.csv", keep_default_na=False)
        matches = build_load_matches(
            facility_rows,
            asset_rows=assets,
            control_unit_rows=control_units,
            load_bus_electrical_metadata=load_bus_electrical_metadata(loads),
            location_id=self.location_id,
            assign_nearest_when_outside_radius=assign_nearest_when_outside_radius,
        )
        validate_load_matches(matches, facility_ids=[r["facility_id"] for r in facility_rows], asset_ids=[r["asset_id"] for r in assets], control_unit_ids=[r["control_unit_id"] for r in control_units])
        write_load_matches(matches, augmented / "critical_load_assignments.parquet")
        load_by_facility = {str(row["facility_id"]): row for row in matches if row.get("assignment_status") == "assigned"}
        profile_inputs = load_inputs(
            facility_rows,
            load_by_facility,
            load_profile_assignments_path=augmented / "load_profile_assignments.parquet",
            oedi_profile_cache_dir=augmented / "oedi_profiles",
            use_oedi_load_profiles=use_oedi_load_profiles,
        )
        ders = build_der_inventory(facility_rows, location_id=self.location_id, load_matches=matches)
        validate_der_assignments(ders, valid_buses=loads["bus"].astype(str).unique())
        write_der_inventory(ders, augmented / "der_inventory.parquet")
        report = {
            "critical_facilities": len(facility_rows),
            "load_matches": len(matches),
            "assigned_load_matches": sum(1 for row in matches if row.get("assignment_status") == "assigned"),
            "load_profile_assignments": len(profile_inputs.assignment_rows),
            "der_inventory_rows": len(ders),
        }
        write_json(augmented / "resilience_layer_report.json", report)
        return report

    def place_switches(
        self,
        *,
        registry_dir: str | Path | None = None,
        augmented_dir: str | Path | None = None,
        max_switches: int = 12,
        min_marginal_benefit: float = 0.0,
        exposure_mode: str = "homogeneous",
        max_candidate_edges: int | None = 250,
        automated_count: int | None = None,
    ) -> dict[str, Any]:
        from .ssap import (
            SsapCandidatePolicy,
            classify_two_tier_switches,
            physical_lines_only,
            physical_switch_candidate_edges,
            select_global_marginal_benefit_budget,
            solve_rpop_ready_ssap_frontier,
            switch_inputs,
            write_switches,
        )
        from .blocks import build_blocks
        from .onm import render_onm_settings

        registry = Path(registry_dir or self.paths.registry)
        augmented = Path(augmented_dir or self.paths.augmented)
        buses = pd.read_csv(registry / "buses.csv", keep_default_na=False)
        lines = pd.read_csv(registry / "lines.csv", keep_default_na=False)
        loads = pd.read_csv(registry / "loads.csv", keep_default_na=False)
        sources = pd.read_csv(registry / "sources.csv", keep_default_na=False)
        transformers = pd.read_csv(registry / "transformers.csv", keep_default_na=False)
        physical = physical_lines_only(lines)
        feeders, meta = switch_inputs(buses, physical, sources, exposure_mode=exposure_mode, transformers=transformers)
        policy = SsapCandidatePolicy(max_candidate_edges=max_candidate_edges)
        frontiers = {
            component_id: solve_rpop_ready_ssap_frontier(
                feeder,
                max_switches=max_switches,
                policy=policy,
                eligible_edges=physical_switch_candidate_edges(feeder, physical),
            )
            for component_id, feeder in feeders.items()
        }
        selection = select_global_marginal_benefit_budget(frontiers, min_marginal_benefit=min_marginal_benefit, max_total_switches=max_switches)
        switches, diagnostics = write_switches(selection, frontiers, meta, physical, location_id=self.location_id, exposure_mode=exposure_mode, candidate_policy=policy, ssap_budget=max_switches)
        if automated_count is not None and len(switches):
            switches = pd.DataFrame(classify_two_tier_switches(switches.to_dict("records"), automated_count=automated_count))
        augmented.mkdir(parents=True, exist_ok=True)
        switches.to_parquet(augmented / "controllable_switches.parquet", index=False)
        diagnostics.to_parquet(augmented / "switch_placement_diagnostics.parquet", index=False)
        der_inventory = pd.read_parquet(augmented / "der_inventory.parquet") if (augmented / "der_inventory.parquet").exists() else None
        blocks, block_report = build_blocks(buses=buses, lines=lines, loads=loads, sources=sources, switches=switches, transformers=transformers, der_inventory=der_inventory, location_id=self.location_id)
        blocks.to_parquet(augmented / "switch_bounded_load_blocks.parquet", index=False)
        settings = render_onm_settings(switches)
        settings.setdefault("settings", {})["microgrid"] = {
            row.block_id: {"buses": json.loads(row.buses_json), "load_kw": float(row.load_kw), "voltage_source_reachability": str(row.voltage_source_reachability)}
            for row in blocks.itertuples(index=False)
        }
        write_json(augmented / "onm_settings.json", settings)
        report = {
            "selected_switch_count": int(len(switches)),
            "component_count": len(frontiers),
            "block_report": block_report,
        }
        write_json(augmented / "switch_layer_report.json", report)
        return report

    def export_onm(
        self,
        *,
        opendss_root: str | Path | None = None,
        augmented_dir: str | Path | None = None,
        registry_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        feeder_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        from .onm import build_powermodels_onm_export

        export = build_powermodels_onm_export(
            feeder_opendss_dir=Path(opendss_root or self.paths.opendss),
            smart_ds_compat_dir=Path(augmented_dir or self.paths.augmented),
            asset_registry_dir=Path(registry_dir or self.paths.registry),
            output_dir=Path(output_dir or self.paths.onm),
            feeder_ids=feeder_ids,
        )
        return read_json(export.manifest_path, {})

    def materialize_run_bundle(
        self,
        *,
        event_id: str,
        mc_draw: int,
        event_start: str | datetime,
        export_dir: str | Path | None = None,
        augmented_dir: str | Path | None = None,
        horizon_hours: int = 72,
        uncertainty_band: float = 0.20,
    ) -> dict[str, Any]:
        from .onm import materialize_onm_run_bundle

        augmented = Path(augmented_dir or self.paths.augmented)
        states_path = augmented / "asset_states.parquet"
        states = pd.read_parquet(states_path) if states_path.exists() else None
        bundle = materialize_onm_run_bundle(
            export_dir=Path(export_dir or self.paths.onm),
            smart_ds_compat_dir=augmented,
            event_id=event_id,
            mc_draw=mc_draw,
            event_start=_coerce_utc(event_start),
            horizon_hours=horizon_hours,
            uncertainty_band=uncertainty_band,
            asset_states=states,
        )
        return read_json(bundle.run_manifest_path, {})

    def audit(self, *, registry_dir: str | Path | None = None, augmented_dir: str | Path | None = None, output_dir: str | Path | None = None, grid_network_dir: str | Path | None = None) -> dict[str, Any]:
        from . import audit as audit_mod

        audit_mod.location_id = self.location_id
        report = audit_mod.build_synthetic_validation_report(
            registry_dir=Path(registry_dir or self.paths.registry),
            smart_ds_compat_dir=Path(augmented_dir or self.paths.augmented),
            grid_network_dir=Path(grid_network_dir or self.paths.root),
        )
        paths = audit_mod.write_synthetic_validation_report(report, Path(output_dir or self.paths.reports))
        gate = audit_mod.build_validation_compliance_gate(report)
        audit_mod.write_validation_compliance_gate(gate, Path(output_dir or self.paths.reports))
        return {"report": report, "gate": gate, "paths": {k: str(v) for k, v in paths.items()}}
