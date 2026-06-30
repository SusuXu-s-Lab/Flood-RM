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