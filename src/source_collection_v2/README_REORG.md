# Source collection reorganization

This package keeps data acquisition and stochastic-boundary science separate.

## Layout

```text
design_events/
  collect_sources/
    aorc.py                 # AORC source subset download only
    cora.py                 # CORA boundary water-level extraction only
    era5.py                 # ERA5 wave fetch only
    nwm.py                  # NWM streamflow/soil sampling only
    usgs.py                 # NWIS site/discharge fetch only
    hurdat2.py              # HURDAT2 download/parse only
    lcra_hydromet.py        # supplemental current-flow-site discovery only
    ssurgo.py               # SSURGO polygon/attribute fetch only
    stream_geo.py           # STREAM-geo table cache only
    national_hydrography.py # NHDPlus HR layer fetch only

  stochastic_boundary/
    audit.py                # Artifact manifest, path resolution, tiny checks
    workflow.py             # source plan/run registry
    gridded.py              # small xarray coordinate/bbox helpers
    rainfall.py             # R_d(t,c), POT declustering, transposition, rainfall members
    meteo.py                # AORC -> Wflow temp/PET conversion
    hydrology.py            # SOILSAT_TOP and NWM sampling-point derivation
    static.py               # static river/reservoir/SSURGO transformations
    members.py              # B_i event-member assembly and empirical weights
```

`collect_sources/` modules should not contain plotting, notebook review, repair routines, or stochastic design-event logic. They only fetch or subset a named external source and return an `Artifact`.

## Use

```python
from design_events.stochastic_boundary import plan, run
from design_events.stochastic_boundary.rainfall import build_aorc_sst
from design_events.stochastic_boundary.members import event_members, empirical_measure

collection_plan = plan(config, paths)
audit = run(collection_plan, skip_existing=True)

# Science is explicit, auditable, and outside collect_sources/.
aorc_step = next(step for step in collection_plan.steps if step.name in {"aorc", "aorc_sst"})
rainfall_artifact = build_aorc_sst(aorc_step.settings(config, paths), skip_existing=True)

rainfall = pandas.read_csv(paths["aorc_sst_rainfall_members_csv"])
soil = pandas.read_csv(paths["nwm_soil_moisture_csv"])
waterlevel = pandas.read_csv(paths["waterlevel_csv"])
members = event_members(rainfall, soil=soil, waterlevel=waterlevel)
weights = empirical_measure(members)
```

## Scientific contract

Rainfall event selection is represented as

\[
R_d(t,c)=|A|^{-1}\int_A\sum_{h=0}^{d-1}X_R(t-h,a+c)\,da,
\]

followed by

\[
\mathcal S=\operatorname{Decluster}_\tau\{(t,c):R_d(t,c)\ge u\}.
\]

The event-member table represents

\[
B_i=(P_i,\Theta_i,\eta_i,W_i,M_i,Q_i,G),
\qquad
\widehat{\mathbb P}_n=n^{-1}\sum_i\delta_{B_i}.
\]

The code intentionally avoids a large class hierarchy. The only common object is `Artifact`, and every source/science output writes the same manifest shape.
