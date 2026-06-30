# NLR Distribution Case

This is a compact, artifact-first rebuild of the Marshfield distribution-grid workflow.  The code now treats the NLR Distribution Suite packages as the native implementation layer:

```text
SHIFT data pull + feeder graph + mappers
  -> GDM DistributionSystem
  -> DiTTo OpenDSS export/import
  -> asset_registry/*.csv
  -> augmented/{assets,control_units}.parquet
  -> ERAD-native event failure probabilities + telemetry
  -> facilities, load matches, profiles, DER, switches, blocks
  -> PowerModelsONM / DynaGrid run bundle
  -> validation and readiness reports
```

Local code owns only the study-specific artifacts, probability/audit notation, restoration sidecars, and stakeholder-facing reports.  It no longer carries a local SHIFT equipment-catalog builder, a custom OpenDSS parser as the default registry path, or a hand-coded ERAD lognormal CDF.  Those domains are delegated to native APIs.

## Install

For artifact-only development:

```bash
python -m pip install -e ".[dev]"
```

For the full NLR suite path:

```bash
python -m pip install -e ".[suite,parquet,geo,solvers]"
```

The `suite` extra installs the native packages used at runtime: `grid-data-models`, `NREL-ditto`, `nrel-shift`, `NREL-erad`, and `gdmloader`.

## Public workflow

```python
from power import DistributionCase

case = DistributionCase.from_config("case.yml")

case.build_baseline()          # SHIFT -> GDM -> DiTTo OpenDSS
case.build_registry()          # DiTTo OpenDSS Reader -> GDM -> registry CSVs
case.export_grid_dataset()     # registry -> SMART-DS-like assets/control units
case.add_event_states()        # ERAD-native fragility probabilities -> states
case.add_resilience_layers()   # facilities, profiles, DER evidence
case.place_switches()          # exact SSAP + block invariants
case.export_onm()              # PMONM/DynaGrid-facing OpenDSS bundle
case.materialize_run_bundle(
    event_id="event_001",
    mc_draw=0,
    event_start="2026-01-01T00:00:00Z",
)
case.audit()
```

## Native API boundary

| Domain | Native package/API | Local responsibility |
| --- | --- | --- |
| Parcel and road data pull | `shift.parcels_from_location`, `shift.get_road_network` | case geometry and reviewed source-anchor gate |
| Feeder graph and baseline system | `shift.PRSG`, `BalancedPhaseMapper`, `TransformerVoltageMapper`, `EdgeEquipmentMapper`, `DistributionSystemBuilder` | build configuration and artifact paths |
| Equipment catalog | `gdm.DistributionSystem.from_json` | selecting the reviewed catalog file |
| OpenDSS conversion | `ditto.readers.opendss.reader.Reader`, `ditto.writers.opendss.write.Writer` | normalized registry rows and provenance |
| Model validation/serialization | GDM Pydantic/Infrasys models, `DistributionSystem.to_json/from_json` | stable artifact IDs and CSV/Parquet contracts |
| Hazard and fragility | ERAD `FragilityCurve`, `ProbabilityFunction`, `AssetSystem.from_gdm`, `HazardSimulator` | mapping local asset labels to ERAD `AssetTypes`, Monte Carlo draw bookkeeping |