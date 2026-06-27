# Notebook To Backend Trace

Generated from notebook JSON and Python AST inspection. Detailed call-site evidence
lives in `docs/audit/notebook_backend_trace.json`.

## Assumptions

- A notebook import from `src/` is a public workflow contract unless documented otherwise.
- Artifact paths below are representative strings found in code or Markdown cells, not a
  guarantee that the notebook always writes them in every run.
- Hidden dependencies include environment variables, `sys.path` mutation, SFINCS/Wflow
  executables, external credentials, or generated files that are not obvious from the
  function name.

## Marshfield

| Notebook | Stage | Backend entry points | Artifacts and hidden dependencies |
| --- | --- | --- | --- |
| `01_grid/01_base_network.ipynb` | Baseline Network | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.baseline_network.source_inputs`, `power.baseline_network.shift_equipment`, `power.baseline_network.build_asset_registry.build_registry`, `power.exports.export_base`, `power.exports.control_registry` | writes/reads `buses.csv`, `lines.csv`, `loads.csv`, `feeders.csv`, `assets.parquet`, `control_units.parquet`; mutates `sys.path` |
| `01_grid/02_augment_network/01_der_inventory.ipynb` | DER inventory and critical loads | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.resilience` facade for facilities, load matches, DER sizing, validation | `critical_facilities.geojson`, `critical_facilities.parquet`, `critical_load_assignments.parquet`, `der_inventory.parquet`; mutates `sys.path` |
| `01_grid/02_augment_network/02_load_profiles.ipynb` | Load profiles | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.resilience.load_inputs`, `power.resilience.load_profile_assignments_schema_version` | `load_profile_assignments.parquet`, critical facility/load assignment inputs; mutates `sys.path` |
| `01_grid/02_augment_network/03_switch_synthesis.ipynb` | Controllable Switches and SSAP | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.resilience` switch facade, `power.plotting.plot_switches` | `controllable_switches.parquet`, `fuses.parquet`, registry CSVs; mutates `sys.path` |
| `01_grid/02_augment_network/04_load_blocks.ipynb` | Switch-Bounded Load Blocks | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.resilience.build_blocks`, `power.plotting.block_detail`, `power.plotting.block_overview` | `switch_bounded_load_blocks.parquet`, `block_invariant_report.json`; mutates `sys.path` |
| `01_grid/02_augment_network/05_onm_export.ipynb` | ONM/RPOP export | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.exports.build_event_window_bundle`, `power.exports.export_powermodels_onm`, `power.plotting.plot_switches` | `onm_export`, `load_profile_assignments.parquet`, `der_inventory.parquet`, switch/block artifacts; mutates `sys.path` |
| `01_grid/03_audit_network.ipynb` | Synthetic Validation Audit | `sfincs_runs.config.load_runtime`, `sfincs_runs.config.build_grid_paths`, `power.audit.synthetic_validation` functions, `power.exports.default_output_dir` | `audit_summary.json`, `marshfield_synthetic_validation_audit.json`, validation figures; mutates `sys.path` |
| `01_grid/psds_plot.ipynb` | legacy/inspection plot | no `src/` imports found | reads SMART-DS-compatible artifacts and figures |
| `02_flood/01_region_setup.ipynb` | flood region setup | `sfincs_runs.build_base.region_notebook` | `bbox.geojson`; SFINCS files; mutates `sys.path` |
| `02_flood/02_collect_sources.ipynb` | source collection | `design_events.collect_sources.workflow`, `collect_aorc_sst` | `config.yaml`, `storm-stats.csv`; mutates `sys.path` |
| `02_flood/03_build_event_catalog.ipynb` | coastal design Event Catalog | probability catalog builders, `design_events.build_events.workflow`, `fit_history` modules, `sfincs_runs.scenarios.coastal_realization`, `design_events.plotting` | `aorc_sst_rainfall_catalog.json`, `cora_boundary_water_level.json`, `era5_snapwave_boundary_forcing.json`, `catalog_risk_metadata.json`; SFINCS files; mutates `sys.path` |
| `02_flood/04/a_build_waves.ipynb` | SFINCS-SnapWave base build | `sfincs_runs.config.load_sfincs_runtime`, `sfincs_runs.build_base.static_intake.build_region_setup`, `structures`, `hydrology`, `snapwave_setup` | `sfincs.inp`, `manning_subgrid_lev0.tif`, `coastal_region.geojson`; SFINCS files and environment variables; mutates `sys.path` |
| `02_flood/04/b_example_waves.ipynb` | wave-enabled example run | `sfincs_runs.config.load_sfincs_runtime`, `sfincs_runs.config.parse_sfincs_inp`, `sfincs_runs.diagnostics`, `sfincs_runs.scenarios.event_forcing` | `sfincs.inp`, `usgs_3dep_13_arcsec.tif`, forcing/run outputs; SFINCS files and environment variables; mutates `sys.path` |
| `02_flood/05_create_scenarios.ipynb` | coastal scenario creation | `sfincs_runs.config.load_sfincs_runtime`, `sfincs_runs.scenarios.create_events.build_scenarios` | `scenario_catalog.csv`, `forcing_manifest.json`, `sfincs_netampr.nc`; SFINCS files; mutates `sys.path` |
| `02_flood/06_evaluate.ipynb` | flood evaluation | `sfincs_runs.config.load_sfincs_runtime`, `sfincs_runs.scenarios` facade, `sfincs_runs.diagnostics` | `scenario_stats.csv`, `scenario_stats_notebook.csv`, `run_metadata.json`, `sfincs.inp`; SFINCS files; mutates `sys.path` |
| `02_flood/07_risk_fiat.ipynb` | FIAT risk | `fiat_runs.load_notebook_runtime`, `fiat_runs.risk_workflow` | `fiat_config.yml`, `catalog_risk_metadata.json`, `preflight.inp`; mutates `sys.path` |
| `03_ops/overlay.ipynb` | operations overlay | no `src/` imports found | reads grid CSV/parquet and `/sfincs_map.nc`; SFINCS files |

## Austin

| Notebook | Stage | Backend entry points | Artifacts and hidden dependencies |
| --- | --- | --- | --- |
| `01_grid/sds_plot.ipynb` | SMART-DS inspection | no `src/` imports found | generated SMART-DS artifacts |
| `02_flood/01_region_setup.ipynb` | flood region setup | `sfincs_runs.build_base.region_notebook` | SFINCS files; mutates `sys.path` |
| `02_flood/02_collect_sources.ipynb` | source collection | `design_events.collect_sources.workflow`, `collect_aorc_sst` | `config.yaml`, `storm-stats.csv`; mutates `sys.path` |
| `02_flood/03_build_event_catalog.ipynb` | inland design Event Catalog | `design_events.build_events.inland_timing`, `design_events.build_events.probability`, `design_events.build_events.workflow`, `design_events.plotting` | `inland_design_event_catalog.csv`, `NWM soil_moisture.csv`; mutates `sys.path` |
| `02_flood/04/a_build_coupled_model.ipynb` | coupled Wflow-SFINCS base build | direct `yaml.safe_dump` for extracted model YAML, `sfincs_runs.build_base` facade, `wflow_runs` facade, `wflow_runs.notebook` | `data/static/data_catalogue.yaml`, `data_catalog.yml`, `gis/wflow_handoff_sources.geojson`, `staticgeoms/rivers.geojson`, `sfincs.inp`; environment variables and SFINCS files |
| `02_flood/04/b_prepare_wflow_dynamic_handoff.ipynb` | Wflow dynamic handoff prep | `design_events.collect_sources.collect_warmup`, `sfincs_runs.scenarios.plan_example`, `wflow_runs` facade, `wflow_runs.notebook` | `probability_catalog.csv`, `sfincs_discharge.nc`, `staticmaps.nc`, `instates.nc`; environment variables and SFINCS files |
| `02_flood/04/c_run_example.ipynb` | inland example run | `sfincs_runs.diagnostics`, `sfincs_runs.scenarios.event_forcing.run_model`, `sfincs_runs.scenarios`, `wflow_runs` facade | `forcing_manifest.json`, `precip.nc`, `sfincs_discharge.nc`, dynamic handoff JSON; environment variables and SFINCS files |
| `02_flood/05_create_scenarios.ipynb` | scenario creation | `wflow_runs.notebook.load_runtime` | `scenario_catalog.csv`, `forcing_manifest.json`, `sfincs_netampr.nc`, `cluster/instructions.txt`; SFINCS files; mutates `sys.path` |
| `02_flood/05b_calibrate_wshed.ipynb` | Wflow calibration | `wflow_runs.calibration`, `wflow_runs.streamflow_realization`, `wflow_runs.notebook` | `instates.nc`; SFINCS files; mutates `sys.path` |
| `02_flood/05c_ship_calibrated.ipynb` | calibration artifact promotion | no `src/` imports found | `scenario_catalog.calibrated.csv`, `wflow_calibration_patch.json`, `wflow_calibration_patch.yaml` |
| `02_flood/06_evaluate.ipynb` | evaluation | `wflow_runs.notebook`, `wflow_runs.plot_event_precipitation_peak_discharge` | `scenario_catalog.csv`, `flood_event_outcome_catalogue.csv`, `scenario_build_report.csv`; SFINCS files; mutates `sys.path` |
| `03_ops/overlay.ipynb` | operations overlay | `power.impact.fragility` | grid CSVs and `/sfincs_map.nc`; SFINCS files |

## Greensboro

Greensboro follows the same inland workflow shape as Austin. The main differences found
in the notebooks are an extra direct call to
`design_events.collect_sources.usgs_streamgages.collect_usgs_streamflow_records` during
Event Catalog construction and a direct import of
`wflow_runs.build_plan.validate_staticmaps` during dynamic handoff preparation.

| Notebook | Stage | Backend entry points | Artifacts and hidden dependencies |
| --- | --- | --- | --- |
| `01_grid/sds_plot.ipynb` | SMART-DS inspection | no `src/` imports found | generated SMART-DS artifacts |
| `02_flood/01_region_setup.ipynb` | flood region setup | `sfincs_runs.build_base.region_notebook` | SFINCS files; mutates `sys.path` |
| `02_flood/02_collect_sources.ipynb` | source collection | `design_events.collect_sources.workflow`, `collect_aorc_sst` | `config.yaml`, `storm-stats.csv`; mutates `sys.path` |
| `02_flood/03_build_event_catalog.ipynb` | inland design Event Catalog | `design_events.build_events.inland_timing`, `design_events.build_events.probability`, `design_events.build_events.workflow`, `design_events.collect_sources.usgs_streamgages`, `design_events.plotting` | `inland_design_event_catalog.csv`, `NWM soil_moisture.csv`; mutates `sys.path` |
| `02_flood/04/a_build_coupled_model.ipynb` | coupled Wflow-SFINCS base build | direct `yaml.safe_dump` for extracted model YAML, `sfincs_runs.build_base` facade, `wflow_runs` facade, `wflow_runs.notebook` | Wflow/SFINCS domain artifacts, `sfincs.inp`; environment variables and SFINCS files |
| `02_flood/04/b_prepare_wflow_dynamic_handoff.ipynb` | Wflow dynamic handoff prep | `design_events.collect_sources`, `sfincs_runs.scenarios`, `wflow_runs`, `wflow_runs.build_plan.validate_staticmaps` | `probability_catalog.csv`, `sfincs_discharge.nc`, `staticmaps.nc`, `instates.nc`; environment variables and SFINCS files |
| `02_flood/04/c_run_example.ipynb` | inland example run | `sfincs_runs.diagnostics`, `sfincs_runs.scenarios`, `sfincs_runs.scenarios.event_forcing`, `wflow_runs` facade | `forcing_manifest.json`, `precip.nc`, `sfincs_discharge.nc`, dynamic handoff JSON; environment variables and SFINCS files |
| `02_flood/05_create_scenarios.ipynb` | scenario creation | `wflow_runs.notebook.load_runtime` | `scenario_catalog.csv`, `forcing_manifest.json`, `sfincs_netampr.nc`, `cluster/instructions.txt`; SFINCS files; mutates `sys.path` |
| `02_flood/05b_calibrate_wshed.ipynb` | Wflow calibration | `wflow_runs.calibration`, `wflow_runs.streamflow_realization`, `wflow_runs.notebook` | `instates.nc`; SFINCS files; mutates `sys.path` |
| `02_flood/05c_ship_calibrated.ipynb` | calibration artifact promotion | no `src/` imports found | `scenario_catalog.calibrated.csv`, `wflow_calibration_patch.json`, `wflow_calibration_patch.yaml` |
| `02_flood/06_evaluate.ipynb` | evaluation | `wflow_runs.notebook`, `wflow_runs.plot_event_precipitation_peak_discharge` | `scenario_catalog.csv`, `flood_event_outcome_catalogue.csv`, `scenario_build_report.csv`; SFINCS files; mutates `sys.path` |
| `03_ops/overlay.ipynb` | operations overlay | `power.impact.fragility` | fragility CSV, `/sfincs_map.nc`; SFINCS files |

## SFO

| Notebook | Stage | Backend entry points | Artifacts and hidden dependencies |
| --- | --- | --- | --- |
| `01_grid/sds_plot.ipynb` | SMART-DS inspection | no `src/` imports found | generated SMART-DS artifacts |
| `03_ops/overlay.ipynb` | operations overlay | `power.impact.fragility` | grid CSVs and `/sfincs_map.nc`; SFINCS files |

## Reader Entry Points

- Baseline Network: start at `locations/marshfield/01_grid/01_base_network.ipynb`,
  then `src/sfincs_runs/config.py`, `src/power/baseline_network/source_inputs.py`, and
  `src/power/baseline_network/build_asset_registry.py`.
- Augmented Network: start at `locations/marshfield/01_grid/02_augment_network/*`,
  then `src/power/resilience/__init__.py`, `src/power/resilience/der.py`,
  `src/power/resilience/profiles.py`, and `src/power/resilience/switches.py`.
- Event Catalogs: start at `02_flood/03_build_event_catalog.ipynb`, then
  `src/design_events/build_events/workflow.py`, probability modules, and
  `src/design_events/fit_history/*`.
- Source collection: start at `02_flood/02_collect_sources.ipynb`, then
  `src/design_events/collect_sources/workflow.py` and source-specific modules.
- Coastal SFINCS-SnapWave: start at Marshfield `02_flood/04/a_build_waves.ipynb`,
  then `src/sfincs_runs/build_base/*`, `src/sfincs_runs/snapwave_setup.py`, and
  `src/sfincs_runs/scenarios/event_forcing.py`.
- Inland Wflow-SFINCS: start at Austin/Greensboro `02_flood/04/*`, then
  `src/wflow_runs/notebook.py`, `src/wflow_runs/build_plan.py`,
  `src/wflow_runs/replay.py`, and `src/sfincs_runs/build_base/inland_base.py`.
- FIAT risk: start at Marshfield `02_flood/07_risk_fiat.ipynb`, then
  `src/fiat_runs/config.py` for `load_notebook_runtime` and
  `src/fiat_runs/risk_workflow.py`.
