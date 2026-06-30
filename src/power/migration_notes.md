# Migration notes

## Removed or demoted local reimplementations

- `baseline.build_shift_equipment_catalog` was removed.  Use a reviewed GDM catalog via `gdm.DistributionSystem.from_json`, or the SHIFT example catalog through `baseline.load_equipment_catalog(None)`.
- The default `registry.build_registry` path no longer parses DSS text with local regexes.  It uses `ditto.readers.opendss.reader.Reader(...).get_system()` and exports rows from native GDM components.
- `impact.failure_probability` no longer evaluates a local lognormal CDF.  It delegates to ERAD `FragilityCurve` / `ProbabilityFunction` probability models.
- OSM parcel pulls now use `shift.parcels_from_location` instead of local OSMnx `features_from_polygon` + `parcels_from_geodataframe` plumbing.

## New native adapter module

`power.native` centralizes lazy imports and native calls:

- `read_gdm_json`, `write_gdm_json`, `load_gdm_dataset`
- `read_opendss`, `write_opendss`
- `shift_distance`, `shift_geo_location`, `load_shift_test_catalog`
- `asset_system_from_gdm`, `run_erad_hazard`

The module raises `NativeDependencyError` with an install hint when an optional suite package is missing.

## Registry compatibility

The registry CSV contract is unchanged:

```text
buses.csv
lines.csv
transformers.csv
sources.csv
loads.csv
load_buses.csv
feeders.csv
summary.json
```

The provenance method changed from `deterministic extraction from generated OpenDSS named properties` to `DiTTo OpenDSS Reader -> native GDM DistributionSystem -> normalized Asset Registry`.

## Event-state compatibility

`dataset.build_asset_states` accepts an optional `fragility_model` implementing:

```python
class FragilityModel:
    def probability(self, local_asset_type: str, depth_m: float | None) -> float: ...
```

Production defaults use ERAD.  Tests and notebooks may inject a small deterministic model for traceability.
