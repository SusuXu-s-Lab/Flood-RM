# Reorganized stochastic boundary-condition source package

This regeneration splits source acquisition from stochastic design-flood science in one shallow package.

```text
collect_sources/
  aorc.py                   # AORC source access and raw subset fetch
  cora.py                   # CORA water-level boundary CSV
  era5.py                   # ERA5 wave NetCDF from CDS or Earth Data Hub
  nwm.py                    # NWM streamflow / soil-state CSVs
  usgs.py                   # USGS NWIS gage discovery and discharge records
  hurdat2.py                # HURDAT2 track download
  lcra_hydromet.py          # supplemental Hydromet site discovery
  ssurgo.py                 # SSURGO polygons and tabular attributes
  stream_geo.py             # STREAM-geo table cache
  national_hydrography.py   # NHDPlus HR layer fetcher
  audit.py                  # Artifact contract and JSON manifests
  workflow.py               # small registry runner
  gridded.py                # small xarray coordinate/bbox helpers
  rainfall.py               # R_d(t,c), POT declustering, SST members
  meteo.py                  # AORC meteo -> Wflow temp/PET fields
  hydrology.py              # NWM soil saturation and point derivation
  static.py                 # stream/reservoir/SSURGO transformations
  members.py                # B_i member table and empirical weights
```

## Minimal usage

```python
import pandas as pd
from collect_sources import plan, run
from collect_sources.members import event_members, empirical_measure
from collect_sources.rainfall import build_aorc_sst

collection_plan = plan(config, paths)
audit = run(collection_plan, skip_existing=True)

# Science is explicit and outside collect_sources/.
aorc_step = next(step for step in collection_plan.steps if step.name in {"aorc", "aorc_sst"})
rainfall_artifact = build_aorc_sst(aorc_step.settings(config, paths), skip_existing=True)

rainfall = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
waterlevel = pd.read_csv(paths["waterlevel_csv"])
soil = pd.read_csv(paths["nwm_soil_moisture_csv"])
members = event_members(rainfall, waterlevel=waterlevel, soil=soil, output_csv=paths["event_members_csv"])
weights = empirical_measure(members)
```

## Scientific contract

\[
R_d(t,c)=\frac{1}{|A|}\int_A\sum_{h=0}^{d-1}X_R(t-h,a+c)\,da,
\quad
\mathcal S=\operatorname{Decluster}_\tau\{R_d(t,c)\ge u\}.
\]

\[
B_i=(P_i,\Theta_i,\eta_i,W_i,M_i,Q_i,G),
\quad
\widehat{\mathbb P}_n=\frac{1}{n}\sum_i\delta_{B_i}.
\]

## Design decisions

* Source-specific I/O adapters and stochastic-boundary helpers live in the shallow package root.
* Probability/stochastic logic remains explicit in the rainfall, meteo, hydrology, static, and members modules.
* There is one audit object: `Artifact`.
* There is one manifest format: source, kind, status, start/end, artifacts, metadata.
* The package uses ordinary Python, NumPy, SciPy, pandas, xarray, GeoPandas, requests, and pyproj. Optional service clients such as `cdsapi` are imported only inside the source function that needs them.
* Plotting, repair routines, readiness dashboards, and notebook-only review tables are intentionally not in the production package.

The regenerated source tree is about 1,550 lines of Python, compared with roughly 9,900 lines in the uploaded collection files.
