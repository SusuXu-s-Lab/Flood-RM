# HydroMT-Wflow Reference

> Source: Official HydroMT-Wflow documentation, especially the stable
> `WflowSbmModel` API page:
> <https://deltares.github.io/hydromt_wflow/stable/api/sbm-model.html>,
> plus the official build examples for current `steps:` YAML shape:
> <https://deltares.github.io/hydromt_wflow/latest/user_guide/wflow_build.html>.
> Reviewed 2026-06-03.
>
> Use this file as a compact syntax guardrail for Flood-RM inland/fluvial
> coupling. Do not invent HydroMT-Wflow step names or parameters from the older
> `coupling/hydromt-wflow-sfincs` example without checking the official docs.

## Scope

Flood-RM should use HydroMT-Wflow as the Wflow Hydrologic Bridge for inland and
fluvial Study Locations. The Event Catalog owns streamflow frequency
provenance; Wflow owns routed event hydrographs that become SFINCS discharge
forcing.

The relevant model class is `hydromt_wflow.WflowSbmModel`, described by the
official docs as the main SBM hydrological model implementation. It inherits
base model build, update, read, write, and component behavior.

## Model Entry Points

Allowed model-level operations:

```python
from hydromt_wflow import WflowSbmModel

model = WflowSbmModel(root=model_root, mode="w")
model.build(steps=build_steps)
model.write()

model = WflowSbmModel(root=model_root, mode="r+")
model.update(model_out=event_model_root, steps=event_steps)
model.write()
```

Preferred CLI shape:

```bash
hydromt build wflow_sbm <model_root> -i <wflow_build.yml> -d <data_catalog.yml> -vvv
hydromt update wflow_sbm <model_root> -i <wflow_update_forcing.yml> -d <data_catalog.yml> -o <event_model_root> -vvv
```

Use `wflow_sbm` for the plugin name unless the installed HydroMT-Wflow version
proves a different registered model name.

## Components

The official `WflowSbmModel` component surface is:

| Component | Use in Flood-RM |
| --- | --- |
| `model.config` | TOML configuration values, output sections, run windows |
| `model.staticmaps` | DEM, flow direction, river mask, soil, landcover, parameters |
| `model.geoms` | basins, rivers, gauges, subcatchments |
| `model.forcing` | event precipitation, temperature, PET |
| `model.states` | cold or warm hydrologic states |
| `model.tables` | parameter and mapping tables |
| `model.output_grid` | gridded outputs when needed |
| `model.output_scalar` | scalar outputs when configured |
| `model.output_csv` | gauge time-series outputs for SFINCS handoff |

## Minimal Build YAML

This is the smallest official-method-oriented shape to start an Austin or
Greensboro baseline. Parameter values are placeholders; method names and
grouping are the important part.

```yaml
steps:
  - setup_config:
      data:
        time.starttime: "2000-01-01T00:00:00"
        time.endtime: "2000-01-02T00:00:00"
        time.timestepsecs: 3600
        input.path_static: staticmaps.nc
        input.path_forcing: inmaps.nc
        output.csv.path: output.csv
        output.csv.header: [river_q, precip]
        output.csv.params:
          - river_water__volume_flow_rate
          - atmosphere_water__precipitation_volume_flux

  - setup_basemaps:
      region:
        subbasin: [-97.7431, 30.2672]
        uparea: 10
        bounds: [-98.2, 29.9, -97.3, 30.6]
      hydrography_fn: merit_hydro
      basin_index_fn: merit_hydro_index
      res: 0.008333333333333333
      upscale_method: ihu

  - setup_rivers:
      hydrography_fn: merit_hydro
      river_geom_fn: hydro_rivers_lin
      river_upa: 30
      rivdph_method: powlaw
      min_rivwth: 30
      slope_len: 2000
      smooth_len: 5000

  - setup_lulcmaps:
      lulc_fn: globcover

  - setup_laimaps:
      lai_fn: modis_lai

  - setup_soilmaps:
      soil_fn: soilgrids
      ptf_ksatver: brakensiek

  - setup_constant_pars:
      "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio": 100
      "soil_water_saturated_zone_bottom__max_leakage_volume_flux": 0

  - setup_gauges:
      gauges_fn: usgs_streamgages
      index_col: site_no
      snap_to_river: true
      snap_uparea: true
      max_dist: 10000
      toml_output: csv
      gauge_toml_header:
        - river_q
        - precip
      gauge_toml_param:
        - river_water__volume_flow_rate
        - atmosphere_water__precipitation_volume_flux
```

Notes:

- Put the model `region` under `setup_basemaps`; the current docs say the
  region argument moved there for HydroMT 1.x.
- Prefer `basin` or `subbasin` regions for Wflow. Use `bbox` only when the
  watershed boundary is intentionally supplied or preprocessed.
- Keep `setup_rivers.hydrography_fn` consistent with `setup_basemaps` so river
  parameters are derived from the same hydrography basis.
- Use `setup_gauges` for USGS gage output points. When drainage area is known,
  `snap_uparea: true` is the defensible default; otherwise use river snapping
  and record the limitation in the Source Artifact.

## Minimal Event-Forcing Update YAML

The event update should change dynamic forcing and run configuration without
rebuilding static maps.

```yaml
steps:
  - setup_config:
      data:
        time.starttime: "2018-10-15T00:00:00"
        time.endtime: "2018-10-18T00:00:00"
        time.timestepsecs: 3600
        input.path_forcing: inmaps-event.nc
        dir_output: run_event

  - setup_precip_forcing:
      precip_fn: event_precip

  - setup_temp_pet_forcing:
      temp_pet_fn: event_temp_pet
      press_correction: true
      temp_correction: true
      pet_method: debruin
```

If the installed HydroMT-Wflow version rejects `write_forcing` or
`write_config`, use the component API directly:

```python
model.forcing.write()
model.config.write()
```

## SFINCS Handoff Rule

Flood-RM should not pass the raw USGS gage hydrograph directly to SFINCS as the
default inland path. The preferred handoff is:

1. Event Catalog row records the USGS Streamgage Network source, selected
   Frequency-Basis Streamgages, POT thresholds, declustering, fitted marginals,
   Streamflow Driver Return Periods, and selected event/design member. Multiple
   active frequency-basis gages are allowed; provenance is gage-specific, but
   the streamflow member must remain a coherent network event rather than an
   independent mix of unrelated gage peaks.
2. Wflow event run produces gauge or outlet discharge time series at configured
   calibration, validation, and SFINCS handoff points, typically `river_q` from
   `river_water__volume_flow_rate`.
3. SFINCS scenario staging converts the Wflow discharge series into official
   SFINCS discharge forcing files and records both USGS and Wflow provenance in
   `forcing_manifest.json`.

Use the Marshfield workflow shape as the baseline: the Event Catalog row owns
the selected forcing members, pairing policy, and timing descriptors before any
model run folder is staged. For Austin and Greensboro, the streamflow-network
event replaces Marshfield's coastal water-level analog as the primary reference
driver, and the same rainfall member/time window must feed both Wflow forcing
and direct SFINCS rainfall.

For inland timing, set `event_reference_time` to the dominant streamgage-network
peak. Rainfall, antecedent soil moisture, Wflow forcing, and SFINCS direct
rainfall should preserve their offsets relative to that streamflow reference,
not be forced to peak at the same hour.

For rainfall pairing, prefer the same storm window as the selected
streamgage-network analog when rainfall data exist. If that coherent rainfall
member is unavailable, fall back to Marshfield-style seasonal-window
permutation and record the fallback policy in the Event Catalog row.

For antecedent soil moisture, follow the selected rainfall member when rainfall
is same-storm/coherent. If coherent rainfall is unavailable, pair antecedent
soil moisture relative to the dominant streamgage-network peak.

Direct USGS-to-SFINCS discharge forcing may be kept as a validation or
sensitivity path, but not as the primary Wflow-coupled production path.

## Austin And Greensboro Defaults To Resolve

- `first location`: implement Greensboro first as the first Wflow-coupled Study
  Location; use the same convention for Austin once reviewed handoff outlets
  exist.
- `first implementation slice`: implement active USGS streamgage discovery and
  reviewed streamgage-network artifact generation before Wflow domain-set
  planning.
- `reviewed streamgage schema`: require at least `site_no`, `site_name`,
  `status`, geometry, `drainage_area_sqmi`, `period_start`, `period_end`,
  `record_years`, `completeness_score`, `roles`, `frequency_basis`,
  `wflow_submodel_id`, `sfincs_domain_id`, `sfincs_handoff_id`,
  `review_status`, and `review_notes`.
- `roles`: keep `roles` as the canonical list field. Any role-specific boolean
  columns are derived conveniences, not the production artifact contract.
- `frequency_basis`: store a named frequency-basis group string, not a boolean,
  so multiple active gages can share one coherent POT/provenance grouping.
- `review_status`: use one of `candidate`, `accepted`,
  `accepted_with_warning`, or `rejected`.
- `notebooks`: keep the same top-level Flood Notebook Workflow shape as
  Marshfield. Add Wflow-coupled sections for streamgage review, Wflow
  readiness, and Wflow-SFINCS handoff inside the relevant notebooks before
  creating a separate inland sequence.
- `region`: use Wflow watershed/subbasin definitions tied to reviewed handoff
  outlet gages. Use SFINCS coverage boxes around SMART-DS evaluation areas;
  do not force SFINCS boxes to be watersheds.
- `SMART-DS`: treat SMART-DS artifacts as the evaluation footprint and minimum
  required SFINCS flood coverage, not as the full Wflow watershed. Expand
  Wflow domains as needed for upstream hydrology and place SFINCS discharge
  sources where Wflow-native rivers enter the coverage box.
- `submodels`: allow multiple Wflow submodels per Study Location when the
  active streamgage network or SFINCS handoff reaches span hydrologically
  separate watersheds. Use one model only when the domain is hydrologically
  coherent.
- `SFINCS domains`: default to coverage boxes around SMART-DS evaluation
  footprints/components. Multiple SFINCS domains are allowed when evaluation
  coverage or hydraulic review requires separate boxes, but the boxes are not
  outlet-delineated watersheds.
- `event catalog`: one Event Catalog row defines the event across the full
  Wflow-SFINCS domain set. Submodels and SFINCS domains may write
  domain-specific run manifests and forcing files, but not independent catalogs.
- `evaluation merge`: combine multiple SFINCS domain outputs into one
  asset-level Evaluation Layer by taking max depth per SMART-DS asset across
  domains, retaining source-domain IDs and overlap diagnostics.
- `gauges_fn`: write a USGS Streamgage Network GeoDataFrame source with
  only active-record candidate gages, plus `site_no`, geometry, drainage area
  where available, and explicit roles such as `frequency`, `calibration`,
  `validation`, and `sfincs_handoff`.
- `frequency`: allow multiple active frequency-basis gages, each with its own
  POT fit and return-period provenance.
- `network events`: preserve coherent historical/design timing across active
  streamgages; do not independently sample gage-specific design peaks inside
  one streamflow member.
- `design events`: for events beyond the observed record, select a coherent
  historical streamgage-network analog and scale it toward target gage-specific
  return-period magnitudes with documented bounds.
- `readiness validation`: default HydroMT-Wflow may support prototypes, but
  production flood layers require historical event replay checks against active
  streamgages: outlet placement, peak-flow magnitude, peak timing, and
  hydrograph volume. Full calibration is a future upgrade if these diagnostics
  show material bias. Summarize these diagnostics in a reviewed pass/warn/fail
  report rather than enforcing universal numeric thresholds at first.
- `SFINCS forcing`: Wflow supplies routed river inflows; SFINCS still receives
  direct rainfall over the flood grid for local pluvial flooding in the same
  Event Catalog row.
- `rainfall coherence`: use the same rainfall member and event window for
  Wflow precipitation forcing and SFINCS direct rainfall, following the
  Marshfield catalog-to-staging pattern.
- `rainfall pairing`: prefer same-storm rainfall for historical
  streamgage-network analogs; use seasonal-window permutation only as a
  documented fallback.
- `soil moisture pairing`: pair antecedent wetness to coherent rainfall when
  available, otherwise to the dominant streamgage-network peak.
- `event reference`: use the dominant streamgage-network peak as the inland
  `Event Reference Time`; keep rainfall lead/lag offsets relative to it.
- `review gate`: write candidate active-record gages first, then require a
  reviewed streamgage-network artifact before production Wflow-coupled runs.
- `inactive gages`: exclude discontinued or historical-only gages entirely from
  the production candidate and reviewed streamgage-network artifacts.
- `output`: configure at least `river_q` at Wflow-SFINCS handoff points.
- `static data`: keep HydroMT data catalog entries separate from Location
  Configuration, matching the existing HydroMT Data Catalog boundary.
- `frequency`: store streamflow POT settings under Event Catalog/source
  collection configuration, not in Wflow model setup.

## Inland Artifact Chain

Greensboro uses a concrete inland artifact chain before Wflow or SFINCS model
runs are opened:

1. `discover_active_streamgage_candidates` writes active USGS discharge gage
   candidates and a reviewed streamgage-network target path.
2. `build_usgs_streamflow_event_members` reads reviewed USGS discharge records,
   extracts POT peaks per active gage, declusters them into coherent
   streamgage-network events, and writes `streamflow_members.csv`. USGS
   `site_no` values are treated as strings so leading zeros remain stable.
3. `build_inland_event_artifacts` pairs streamflow members with rainfall and
   antecedent soil-moisture members, then writes:
   - `data/event_catalog/catalog/probability_catalog.parquet`
   - `data/event_catalog/catalog/probability_catalog.csv`
   - `data/event_catalog/catalog/wflow_replay_set.parquet`
   - `data/event_catalog/catalog/wflow_replay_set.csv`
   - `data/event_catalog/event_manifest.yaml`
   - `data/event_catalog/catalog/event_catalog_audit.json`
4. `write_wflow_sfincs_handoff_manifest` writes
   `data/wflow/domain_set_handoff.yaml` from the Event Catalog rows, Wflow
   submodel set, SFINCS domain set, and direct-rainfall setting.

Notebook gates should call these APIs in order. Network downloads, HydroMT
model builds, Wflow runs, and SFINCS runs remain explicit review-gated actions.
