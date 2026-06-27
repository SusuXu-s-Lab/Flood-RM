# Refactor Roadmap

This roadmap is for reducing, splitting, simplifying, and documenting `src/` without
changing scientific behavior. Detailed candidates live in
`docs/audit/reduction_candidates.json`.

## Guardrails

- Preserve formulas, event-selection semantics, flood-model setup semantics, grid
  topology logic, switch-allocation logic, and file-format semantics unless a change is
  explicitly justified and validated.
- Treat notebooks under `locations/<name>/` as the public workflow interface.
- First add maps, inventories, and characterization tests. Then split files. Then remove
  dead or redundant code.
- Prefer standard Python and scientific-library idioms, but never silently replace domain
  logic with a library call unless equivalence is demonstrated.
- Treat helper-module extraction as transitional. Once behavior is characterized, remove,
  inline, or replace thin wrappers that do not encode a named Flood-RM concept. See
  `docs/plumbing_reduction_directive.md`.
- Apply this rule across all of `src/`, not only `power`. Repeated `_location_path`,
  `_repo_path`, `load_config`, `load_runtime`, source-artifact, manifest, parser, hash,
  and compatibility helpers are review targets unless they protect a notebook-facing
  contract or a named domain artifact.
- Treat `REDUNDANT_OR_REPLACEABLE` as a review label, not a deletion label.
- Treat no-notebook-call-site evidence as incomplete until scripts, tests, package
  facades, dynamic imports, and artifact-only notebooks have also been checked.

## Core Concepts That Constrain Refactors

Future work packages must explicitly state whether they touch any of these concepts:

- `Driver Probability Index`
- `Field-Preserving Realization`
- `Probability Catalog`
- `Event Reference Time`
- `Forcing Support Window`
- `Event Timing Descriptor`
- `Static Intake`
- `Model Recipe Configuration`
- `Source Artifact`
- `Hydrodynamic Truth Set`
- `Subgrid Flood Layer`
- `Stable Grid ID`
- `SMART-DS-Compatible Interface`
- `Block Invariant Contract`
- `Synthetic Validation Audit`

If a proposed edit changes one of these concepts, it needs focused characterization tests
and a migration note before implementation.

## Refactor Gate

Before editing source code:

1. Confirm the file is owned by exactly one active work package.
2. Confirm the public notebook imports, call order, artifact paths, and expected side
   effects that must be preserved.
3. Add or identify characterization tests for schema columns, path conventions,
   manifests, and public facades.
4. Split oversized files behind existing facades before simplifying behavior.
5. For deletion, prove the code is not reachable from notebooks, facades, scripts, tests,
   dynamic imports, or artifact-only workflows.

## Stage 0: Freeze The Map

Target outcome:

- Keep `docs/codebase_map.md`, `docs/notebook_backend_trace.md`,
  `docs/refactor_roadmap.md`, `docs/reduction_candidates.md`, and
  `docs/agent_work_packages.md` current.
- Keep JSON inventories in `docs/audit/` as review evidence.
- Add a lightweight notebook-import smoke test that checks every notebook `src/` import
  still resolves.
- Add an artifact-only notebook trace for notebooks that do not import `src/` but still
  read or write public workflow artifacts.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest`
- `FLOOD_RM_LOCATION=marshfield uv run python -m pytest`
- A no-execution notebook import smoke check for `locations/*/**/*.ipynb`.

## Stage 1: Characterize Notebook Interfaces

Target outcome:

- Add characterization tests for `load_runtime` helpers and public facades:
  `power.resilience`, `power.exports`, `sfincs_runs.build_base`,
  `sfincs_runs.scenarios`, and `wflow_runs`.
- Snapshot expected artifact paths and manifest keys for one tiny fixture per workflow.
- Document any intentional notebook migration before changing imports.

Risk:

- Medium. This touches public workflow contracts but should not change implementation.

## Validation Harness

Added lightweight smoke coverage in `tests/test_architecture_smoke.py` and shared
collection setup in `tests/conftest.py`.

Protected now:

- All Python modules under `src/` import under the Reference Study Location
  (`FLOOD_RM_LOCATION=marshfield`) without executing notebooks, downloads, external
  solvers, or credential-dependent workflows.
- Local imports found in notebooks under `locations/` resolve to importable modules and
  symbols.
- Selected notebook-facing facades keep key signature parameters, including grid,
  source-collection, Event Catalog, SFINCS, Wflow, FIAT, and ONM/export entry points.
- Known duplicate `__all__` exports in public facades are captured as a baseline so
  future refactors do not introduce new duplicate public names silently.
- Small pure utilities are covered for generated YAML notices and stable grid-artifact
  value parsing/token helpers.

Still unprotected:

- Full notebook execution order, cell outputs, and side effects.
- Scientific numerical equivalence for hydrology, probability fitting, event selection,
  SSAP, SFINCS/Wflow setup, and FIAT risk.
- External tools and services: SFINCS, Wflow CLI, Julia/PowerModelsONM, DynaGrid,
  EarthDataHub, CDS, OEDI downloads, ArcGIS services, and credentials.
- Large local datasets and generated Location Workspace artifacts.
- Artifact-only notebooks with no `src` imports beyond import/path smoke coverage.

Validation commands:

- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m compileall src`
- `uv run python -m pytest`

## Stage 2: Split Notebook-Facing Orchestrators

These files have high leverage because they are both large and notebook-facing.

| Current file | Proposed modules | Keep public interface | Move internal implementation | Risk | Expected reduction | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| `src/sfincs_runs/build_base/region_notebook.py` | `region_runtime.py`, `region_collect.py`, `region_plotting.py`, `coverage_preflight.py`, `static_input_qa.py` | Existing notebook imports from `region_notebook.py`; re-export stable calls during migration | `_collect_*`, `_plot_*`, coverage writers, static raster checks | High | 2200-line facade reduced to roughly 200-400 lines | region notebook smoke tests, `tests/sfincs_runs/build_base/region_notebook_test.py`, compileall |
| `src/design_events/collect_sources/workflow.py` | `runtime.py`, `plan.py`, `readiness.py`, `source_plots.py`, `streamgage_review.py` | Existing notebook calls from `02_collect_sources.ipynb` | plotting helpers, streamgage QA, source summaries | Medium | 2000-line notebook module reduced by 50-70 percent | collect-source tests, notebook import smoke |
| `src/wflow_runs/build_plan.py` | `domain_plan.py`, `source_strategy.py`, `staticmaps_qa.py`, `reservoirs.py`, `hydromt_build.py`, `repairs.py` | Keep facade exports through `wflow_runs.__init__` and direct `validate_staticmaps` import | NHD/WBD planning, static map repair, reservoir QA, HydroMT build steps | High | 3500-line module split into focused modules | `tests/wflow_runs/wflow_build_plan_test.py`, Wflow import smoke |
| `src/sfincs_runs/diagnostics.py` | `diagnostic_tables.py`, `depth_metrics.py`, `forcing_plots.py`, `animations.py`, `driver_response.py` | Keep plotting function names used by notebooks | stats tables, NetCDF readers, animation helpers | Medium | 2500-line plotting/stat file split by product | diagnostics tests, sample NetCDF fixture tests |

Completed SFINCS plumbing cleanup: `src/sfincs_runs/notebook.py`,
`src/sfincs_runs/scenarios/io.py`, `src/sfincs_runs/build_base/region_setup.py`, and
`src/sfincs_runs/single_use_case/` were deleted after caller migration. The remaining
SFINCS refactors should focus on oversized domain files (`region_notebook.py`,
`diagnostics.py`, `inland_base.py`, `scenarios/inland_coupled.py`) and should not
recreate generic IO/notebook facades.

## Stage 3: Split Grid Dataset Modules

| Current file | Proposed modules | Keep public interface | Move internal implementation | Risk | Expected reduction | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| Former `src/power/artifacts.py` | Deleted after caller migration; also deleted `src/power/_artifact_paths.py`, `src/power/_artifact_values.py`, `src/power/_artifact_ids.py`, and `src/power/_artifact_io.py` | No notebook-facing import depended on `power.artifacts`; runtime callers now use `src/paths.py`, `study_location.py`, or local helpers | Schema-aware writes, hashes, manifests, and Stage A1 Stable Grid IDs live in `power.exports.smart_ds_grid`; scalar parsing lives at the small number of callers that need it | Low for deleted helper; Medium for broader src-wide path/runtime consolidation | removed generic artifact/path facade | `tests/flood_rm/power_artifacts_test.py`, `tests/flood_rm/power_restoration_test.py`, architecture smoke |
| Former `src/power/notebook.py` | Deleted after Marshfield grid notebooks were migrated to direct runtime/path imports | Grid notebooks now import `sfincs_runs.config.load_runtime` and `build_grid_paths` directly | Output-directory creation remains in the notebook setup cells | Low after notebook import smoke | removed shallow notebook facade | notebook import smoke, `tests/flood_rm/notebook_runtime_helpers_test.py` |
| `src/power/resilience/switches.py` | `ssap.py`, `switch_candidates.py`, `fuses.py`, `load_blocks.py`, `block_validation.py`, `topology.py` | `power.resilience` facade names used by notebooks | DP solver internals, candidate metrics, block invariant checks, ONM injection | High | 2000-line mixed module split into science and artifact layers | switch/load-block tests, notebook import smoke |
| `src/power/audit/synthetic_validation.py` | `smartds_reference.py`, `power_flow.py`, `short_circuit.py`, `compliance_gate.py`, `audit_plots.py` | `audit_summary`, `plot_audit`, `run_stats`, `run_ops` | OpenDSS operational checks, reference-set statistics, Markdown rendering | Medium | 1900-line report module split into validation products | audit notebook smoke, existing power tests |
| `src/power/exports/smart_ds_grid.py` | `asset_export.py`, `event_samples.py`, `telemetry.py`, `schemas.py`, `manifest.py` | `power.exports` facade | synthetic/SFINCS event sampling, schemas, validation | Medium | 1580-line export module split by artifact type | export schema tests, fixture parquet/csv tests |
| `src/power/exports/restoration.py` | `onm_render.py`, `event_windows.py`, `run_bundle.py`, `julia_smoke.py`, `dss_text.py` | `build_event_window_bundle`, `export_powermodels_onm`, `materialize_onm_run_bundle` | DSS text manipulation, Julia subprocess setup, run bundle materialization | Medium | 1590-line module split by output product | ONM runnable asset tests, no live Julia by default |
| `src/power/plotting.py` | `network_plots.py`, `switch_plots.py`, `block_plots.py`, `export_plots.py` | functions used by grid notebooks | repeated Matplotlib setup | Low | 1500-line plotting file split by notebook stage | plot smoke tests using Agg backend |

Note: `smart_ds_grid.py`, `restoration.py`, `der.py`, and `profiles.py` must be treated
as Grid Dataset semantics until tests prove a slice is pure plumbing. They define
artifact schemas, Stable Grid IDs, event windows, load profiles, DER assignment, ONM
settings, and telemetry-facing outputs.

Power artifact cleanup rule: preserve the Stable Grid ID and published artifact schema
contracts, but do not preserve one-line ID helpers, generic parquet/hash wrappers,
permissive value parsers, or power-specific path mini-systems merely because they exist.
The first pass removed the generic IO/hash and one-off Stage A1 ID helpers from the
public artifact surface. Continue migrating remaining scalar parsers toward direct
pandas/vectorized validation where call sites make that safe.

## Stage 4: Split Event And Source Modules

| Current file | Proposed modules | Keep public interface | Move internal implementation | Risk | Expected reduction | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| `src/design_events/plotting.py` | `return_period_plots.py`, `pairing_plots.py`, `catalog_plots.py`, `streamflow_plots.py`, `copula_plots.py` | existing plot function names, possibly via re-export | repeated plot styling and table shaping | Low to Medium | 2240-line file split by diagnostic family | plotting tests with Agg backend |
| `src/design_events/collect_sources/national_hydrography.py` | `nhd_fetch.py`, `wbd_fetch.py`, `river_geometry.py`, `hydromt_basemap.py`, `ssurgo_wflow.py` | source-collection facade functions | NLDI lookup, attribute transfer, NetCDF writes | High | 1800-line source module split by external source/product | national hydrography tests, no network by default |
| `src/design_events/collect_sources/aorc_sst.py` | `sst_catalog.py`, `sst_windows.py`, `sst_download.py`, `sst_manifest.py` | `collect_aorc_sst` | path/manifest and collection windows | Medium | 1300-line module split by source product | AORC SST tests, no download by default |
| `src/design_events/build_events/selection.py` | `selection_policy.py`, `scenario_catalog.py`, `training_selection.py` | notebook-facing workflow calls only after tests | dataframe ranking/filtering helpers | High | 900-line science file split only after characterization | selection tests, catalog fixture tests |
| `src/design_events/build_events/coastal.py` | `coastal_policy.py`, `coastal_members.py`, `coastal_catalog.py` | existing workflow entry points | catalog materialization vs driver semantics | High | modest line reduction; better locality | design-event tests and notebook import smoke |

## Stage 5: Study Location And Shared Plumbing

| Current file | Proposed modules | Keep public interface | Move internal implementation | Risk | Expected reduction | Validation |
| --- | --- | --- | --- | --- | --- | --- |
| `src/study_location.py`, `src/aoi.py`, and `src/paths.py` | Shared path module created and simplified; `_study_location_paths.py`, `_study_location_templates.py`, `_study_location_recipes.py`, and `src/data.yaml` deleted; AOI helper renamed to short `aoi.py`; resolved-config writer, stage-order facade, and recipe inheritance removed | `define_location`, `load_location_config`, `resolve_study_location`, `build_study_area`, `study_area_bbox` | `src/paths.py` owns minimal repo/location path resolution and GeoJSON IO; Study Location identity and small YAML include merge live in the facade because callers use it at runtime; defaults are no longer resolved here | Medium remaining | explicit location YAML, fewer path conventions, no hidden methodology-default layer | shared-infrastructure tests, architecture smoke, location definition tests |
| repeated `load_runtime` helpers | shared runtime conventions, not necessarily one module | preserve per-domain `load_runtime` imports | common path summary/render helpers only | Medium | fewer wrappers, clearer notebook runtime contracts | notebook runtime helper tests |
| repeated `write_manifest` / config loaders | tiny shared utilities only where semantics match | preserve domain-specific manifests | date/path JSON helpers | Low to Medium | removes duplication without creating a framework | focused pure-function tests |

Src-wide plumbing inventory from the artifact/path review:

- `src/design_events/utils.py` was deleted. Design-event runtime paths now live in
  `src/design_events/runtime.py`; collect-source readiness lives in
  `design_events.collect_sources.workflow`; source manifest writes/checks live in the
  adapters that own the artifacts.
- Path helpers still repeat in `design_events.collect_sources.*`,
  `sfincs_runs.config`, `sfincs_runs.build_base.*`, `sfincs_runs.scenarios.*`,
  `wflow_runs.*`, `fiat_runs.notebook`, and `power.*`.
- Source artifact helpers should not be restored as a generic module unless adapter-local
  duplication proves a real shared seam.
- Notebook-facing `load_runtime` helpers should remain public seams, but their internals
  should converge on `src/paths.py` plus domain-owned artifact keys where possible.
- Scalar parser and slug/token helpers should live near their artifact owner or be
  replaced by direct pandas/numpy operations.

## Stage 6: Remove Or Archive Stale Paths

Candidates needing call-site confirmation:

- Notebooks with no `src/` imports that duplicate plotting or artifact inspection:
  `sds_plot.ipynb`, `psds_plot.ipynb`, and some `03_ops/overlay.ipynb` notebooks.
- Duplicate command/path conventions between `scripts/onm_export.py` and
  `power.exports.restoration`.
- Exact duplicate function names such as `main`, `parse_args`, `load_config`,
  `load_runtime`, and `write_manifest`. Some are legitimate local conventions, so use
  the deletion test before merging.

Validation:

- Search for notebook, script, and test call sites before deleting.
- Search package facades (`__init__.py`), CLI scripts, string/dynamic imports, and
  artifact-only notebooks before deleting.
- Run compileall and pytest.
- Keep migration notes for any notebook-facing import changes.

Do not delete or archive these paths from the current evidence alone. Reclassify them as
`UNKNOWN_NEEDS_REVIEW` until artifact dependencies and external workflow use are mapped.

## Highest-Priority Targets

1. `src/wflow_runs/build_plan.py`: largest file, notebook-facing, mixes domain planning,
   external source readiness, HydroMT build steps, staticmap QA, repairs, and reservoirs.
2. `src/sfincs_runs/build_base/region_notebook.py`: public notebook interface mixing
   coastal/inland region setup, static data collection, plotting, and coverage planning.
3. `src/power/resilience/switches.py`: core topology/switch-allocation semantics mixed
   with artifact rows and block validation.
4. `src/design_events/collect_sources/workflow.py`: public source-collection notebook
   interface mixing planning, readiness, plotting, streamgage review, and source IO.
5. `src/study_location.py`: shared Location Configuration interface with path,
   geospatial, merge, and template responsibilities.

## Validation Baseline

- `uv run python -m compileall src`: passed.
- `uv run python -m pytest`: failed during collection when no Flood-RM location was
  selected. The import-time location resolver raised: "No Flood-RM location is selected.
  Set FLOOD_RM_LOCATION_CONFIG, set FLOOD_RM_LOCATION, or run from a
  locations/<name> workspace."
- `FLOOD_RM_LOCATION=marshfield uv run python -m pytest`: completed with 338 passed,
  34 failed, and 8 warnings in 173.45 seconds.

Failure themes in the location-scoped run:

- Missing or outdated cluster helper expectations, including
  `cluster/run_sfincs_dsai_inland_coupled.slurm` and wave-coupled SLURM defaults.
- Real-location fixture drift for Greensboro and Austin notebooks/artifacts, including
  selected Austin domain expectations and missing Greensboro gauge GeoJSON.
- Notebook contract tests expecting older inland setup text or clean execution counts.
- Wflow/SFINCS dynamic handoff readiness tests where accepted/blocked event status did
  not match fixture expectations.
- One NetCDF overwrite/permission failure in
  `tests/wflow_runs/wflow_build_plan_test.py::test_build_meteo_writes_scaled_precip_and_neutral_pet`.
