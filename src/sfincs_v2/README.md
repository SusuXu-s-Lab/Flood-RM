# `sfincs_native_runtime`

Minimal HydroMT-SFINCS event runtime for stochastic design flood scenarios.

The package is designed around one ownership rule:

> HydroMT-SFINCS owns SFINCS model files. HydroMT-Wflow only supplies upstream discharge hydrographs at SFINCS-native `src.name`. SnapWave event forcing is staged on native SFINCS/SnapWave boundary points.

## What was kept

- Native SFINCS source contract from `SfincsModel.rivers.create_river_inflow(...)`.
- Native event forcing calls:
  - `sf.discharge_points.create(timeseries=...)`
  - `sf.water_level.create(timeseries=...)`
  - `sf.precipitation.create(precip=..., aggregate=False)`
  - `sf.initial_conditions.create(ini=...)`
  - native infiltration and structure components during base build/update.
- Field-preserving stochastic realizations:
  - coastal: `eta = MSL + tide + K * NTR + SLR`
  - rainfall: `P_e(x,y,t) = K_e P_analog(x,y,t)`
- Poisson annual exceedance probability:
  - `lambda_e = Lambda * w_e`
  - `p_d(x) = 1 - exp(-sum_e lambda_e I[D_e(x) > d])`

## What moved out

- Legacy reviewed-gage, stream-boundary, and snap-to-boundary handoff placement.
- Notebook plotting and animation.
- Import alias compatibility layers.
- Manual `sfincs.bzs` / `sfincs.dis` writers except the small SnapWave table writer, which remains because arbitrary event-varying SnapWave point time-series are not yet exposed as a stable HydroMT-SFINCS public component in the same way as water level, discharge, and precipitation.

## Minimal inland flow

```python
from pathlib import Path
from sfincs_native_runtime.io import copy_base_model
from sfincs_native_runtime.forcing import stage_inland_event_forcing
from sfincs_native_runtime.audit import audit_run_folder
from sfincs_native_runtime.solver import run_sfincs

run_root = copy_base_model(
    Path("data/sfincs/domains/main/base"),
    Path("data/sfincs/scenarios/evt_0001/main"),
    force=True,
)
manifest = stage_inland_event_forcing(
    run_root,
    event_id="evt_0001",
    wflow_discharge_nc="data/wflow/events/evt_0001/sfincs_discharge.nc",
    precip_nc="data/wflow/events/evt_0001/precip.nc",
    direct_rainfall=True,
)
audit = audit_run_folder(run_root)
assert audit.passed, audit.to_dict()
run_sfincs(run_root, storage_dir="data/sfincs/run_outputs/evt_0001/main", sfincs_bin="sfincs")
```

## Native Wflow source contract

```python
from sfincs_native_runtime.schema import NativeSourceConfig
from sfincs_native_runtime.sources import create_wflow_source_contract

src = create_wflow_source_contract(
    "data/sfincs/domains/main/base",
    sfincs_domain_id="main",
    output="data/sfincs/domains/main/base/gis/wflow_handoff_sources.geojson",
    source_config=NativeSourceConfig(
        hydrography="us_hydrography_basemap",
        river_upa_km2=5.0,
        river_len_m=500.0,
        buffer_m=200.0,
        river_width_m=0.0,
    ),
)
```

Wflow should then write:

```text
sfincs_discharge.nc
  discharge(time, index)
  name(index) == src["name"]
```

## Minimal coastal flow

```python
import pandas as pd
from sfincs_native_runtime.coastal import build_coastal_hydrograph_from_analog, stage_coastal_event_forcing

components = pd.read_csv("coastal_components.csv", parse_dates=["time"]).set_index("time")
eta = build_coastal_hydrograph_from_analog(
    components,
    peak_time="2018-03-02T15:00:00",
    scale_factor=1.35,
    msl_offset_m=0.30,
)
stage_coastal_event_forcing(
    "data/sfincs/scenarios/evt_0120",
    event_id="evt_0120",
    eta=eta,
)
```

## CLI examples

```bash
sfincs-native-runtime create-sources \
  --config locations/example/config.yaml \
  --sfincs-domain-id main \
  --hydrography us_hydrography_basemap

sfincs-native-runtime stage-inland \
  --run-root data/sfincs/scenarios/evt_0001/main \
  --event-id evt_0001 \
  --wflow-discharge-nc data/wflow/events/evt_0001/sfincs_discharge.nc \
  --precip-nc data/wflow/events/evt_0001/precip.nc

sfincs-native-runtime run \
  --scenarios-root data/sfincs/scenarios \
  --scenario-catalog data/sfincs/scenarios/scenario_catalog.csv \
  --storage-root data/sfincs/run_outputs \
  --run-root data/sfincs/run_stage \
  --sfincs-bin sfincs \
  --workers 4
```
