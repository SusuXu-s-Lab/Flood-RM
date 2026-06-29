# Migration notes: old SFINCS runner to native HydroMT-SFINCS runtime

## Delete or move out of the SFINCS runtime core

The new core assumes `sf.rivers.create_river_inflow(...)` is the default and only production source-point generator. Move these to `compat/legacy_handoffs.py` if still needed for review evidence:

- reviewed-gage SFINCS source placement
- snap-to-domain-boundary source placement
- stream-boundary intersection source placement
- Wflow-native river/SFINCS-boundary reconstruction
- source-point deduplication and tolerance tuning
- plotting shims that mutate HydroMT-SFINCS private component data

## Replace manual writers

| Old pattern | New native pattern |
|---|---|
| manual `sfincs.dis` writer | `sf.discharge_points.create(timeseries=...)` |
| manual `sfincs.bzs` writer | `sf.water_level.create(timeseries=...)` |
| manual netampr pointers | `sf.precipitation.create(precip=..., aggregate=False)` |
| manual spatial ini writer | `sf.initial_conditions.create(ini=...)` |
| custom source placement | `sf.rivers.create_river_inflow(...)` |

## Keep as scientific code

- Coastal tide/NTR decomposition and scaling.
- AORC field-preserving rainfall scaling.
- SSURGO/HSG/Ksat conditioning where local pedology is required.
- Event probability weights and Poisson AEP products.
- Small audit receipts.

## SnapWave note

HydroMT-SFINCS should build the SnapWave mask and boundary points. This package writes the four event-varying SnapWave forcing tables (`snapwave.bhs`, `snapwave.btp`, `snapwave.bwd`, `snapwave.bds`) as a thin SFINCS-table adapter until the installed HydroMT-SFINCS version exposes a stable public DataFrame API for arbitrary point-wave time series.
