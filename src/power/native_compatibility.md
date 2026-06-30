# Native NLR Distribution Suite compatibility

This rebuild replaces local reimplementations with native package calls wherever the capability belongs to SHIFT, DiTTo, GDM, or ERAD.

## SHIFT

Used directly:

```python
shift.parcels_from_location(location, Distance(...))
shift.get_road_network(location, Distance(...))
shift.get_kmeans_clusters(k, parcel_points)
shift.PRSG(groups=..., source_location=GeoLocation(...)).get_distribution_graph()
shift.BalancedPhaseMapper(...)
shift.TransformerVoltageMapper(...)
shift.EdgeEquipmentMapper(...)
shift.DistributionSystemBuilder(...).get_system()
```

Local code no longer builds a bespoke example equipment catalog.  It loads a reviewed GDM catalog with `DistributionSystem.from_json` or, by default, the catalog packaged with SHIFT examples.

## DiTTo

Used directly:

```python
from ditto.readers.opendss.reader import Reader
from ditto.writers.opendss.write import Writer

system = Reader("Master.dss").get_system()
Writer(system).write(output_path="derived_opendss")
```

The Asset Registry is now `DiTTo Reader -> GDM DistributionSystem -> CSV artifact rows`.  The custom line-oriented OpenDSS parser from the first rebuild is removed from the default path.

## GDM

Used directly:

```python
from gdm import DistributionSystem

system = DistributionSystem.from_json("catalog_or_model.json")
system.to_json("baseline_gdm.json", overwrite=True)
```

GDM component accessors such as `get_buses`, `get_lines`, `get_loads`, `get_transformers`, and `get_voltage_sources` are preferred when available.  Generic component introspection is only a compatibility adapter that turns validated GDM objects into the small registry CSV contract.

## ERAD

Used directly:

```python
from erad.models.fragility_curve import FragilityCurve, ProbabilityFunction
from erad.systems import AssetSystem
from erad.runner import HazardSimulator

asset_system = AssetSystem.from_gdm(distribution_system)
HazardSimulator(asset_system=asset_system).run(hazard_system=hazard_system)
```

For Stage A2 Monte Carlo states, the local package maps Grid Dataset asset labels to ERAD `AssetTypes`, constructs ERAD `FragilityCurve` / `ProbabilityFunction` objects from reviewed depth-curve CSVs when supplied, or binds to ERAD default curves.  Probability evaluation goes through ERAD's native `prob_model.probability(...)` path.

## What remains local

The following remain local because they are study contracts rather than suite-library internals:

- Stable artifact IDs, manifests, and CSV/Parquet schemas.
- SMART-DS-like static/event artifact assembly.
- Critical-facility public-evidence matching to synthetic load buses.
- Exact SSAP switch-placement objective and switch-bounded block invariants.
- PowerModelsONM/DynaGrid sidecars, run bundles, and readiness gates.
- Stakeholder-facing audit wording and non-claim language.
