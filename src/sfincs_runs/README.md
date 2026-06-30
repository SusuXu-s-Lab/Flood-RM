# `sfincs_runs`

Minimal HydroMT-SFINCS event runtime for stochastic design flood scenarios.

The package is designed around one ownership rule:

> HydroMT-SFINCS owns SFINCS model files. HydroMT-Wflow only supplies upstream discharge hydrographs at SFINCS-native `src.name`. SnapWave event forcing is staged on native SFINCS/SnapWave boundary points.

## What was kept

- Native SFINCS source contract from `SfincsModel.rivers.create_river_inflow(...)`.
- Native event forcing calls:
  - `sf.discharge_points.create(timeseries=...)`
  - `sf.precipitation.create(precip=..., aggregate=False)`
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
from hydromt_sfincs import SfincsModel
from sfincs_runs.io import copy_base_model
from sfincs_runs.forcing import read_wflow_discharge, stage_gridded_precipitation, stage_wflow_discharge_points
from sfincs_runs.solver import run_sfincs

run_root = copy_base_model(
    Path("data/sfincs/domains/main/base"),
    Path("data/sfincs/scenarios/evt_0001/main"),
    force=True,
)

sf = SfincsModel(root=str(run_root), mode="r+")
sf.read()
q = read_wflow_discharge("data/wflow/events/evt_0001/sfincs_discharge.nc")
stage_wflow_discharge_points(sf, q, event_id="evt_0001", model_root=run_root)
stage_gridded_precipitation(sf, "data/wflow/events/evt_0001/precip.nc")
sf.write()

run_sfincs(run_root, storage_dir="data/sfincs/run_outputs/evt_0001/main", sfincs_bin="sfincs")
```

## Native Wflow source contract

```python
from sfincs_runs.schema import NativeSourceConfig
from sfincs_runs.sources import create_wflow_source_contract

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
from sfincs_runs.coastal import build_coastal_hydrograph_from_analog, stage_coastal_event_forcing

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
