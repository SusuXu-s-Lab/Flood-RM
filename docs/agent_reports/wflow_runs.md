# Wflow Runs Task 4d Report

## Scope

Ownership followed:

- Edited only `src/wflow_runs/__init__.py`, `src/wflow_runs/replay.py`, and this report.
- Did not edit notebooks, `src/design_events/**`, `src/power/**`, `src/sfincs_runs/**`, `src/fiat_runs/**`, or shared docs.
- Observed pre-existing Wflow worktree modifications before this pass in `src/wflow_runs/build_plan.py`, `src/wflow_runs/notebook.py`, `src/wflow_runs/replay.py`, and `src/wflow_runs/states.py`; those were treated as parallel/user changes and not reverted.

Guarded concepts touched:

- `Event Reference Time`: read-only preservation through `resolve_event_window`; no logic change.
- `Forcing Support Window`: read-only preservation in `build_meteo`; no timestamp/window change.
- `Wflow Hydrologic Bridge`: only duplicate private streamflow path/scalar helpers were removed from `replay.py`.
- `Wflow Readiness Validation`: no acceptance, readiness, or QA status logic changed.
- `Model Recipe Configuration`: no recipe semantics changed by this pass.

## Call-Site Evidence

`rg` evidence before editing found Wflow is live from:

- Notebooks: Austin and Greensboro `04/a_build_coupled_model`, `04/b_prepare_wflow_dynamic_handoff`, `04/c_run_example`, `05_create_scenarios`, `05b_calibrate_wshed`, and `06_evaluate`.
- Package facades/tests: `tests/test_architecture_smoke.py` imports `wflow_runs` facade names and `wflow_runs.notebook.load_runtime`.
- Cross-domain source callers: `src/sfincs_runs/build_base/region_notebook.py`, `src/sfincs_runs/build_base/inland_base.py`, `src/sfincs_runs/scenarios/inland_coupled.py`, `src/sfincs_runs/scenarios/run_inland_coupled_events.py`, and `src/design_events/*/workflow.py`.
- Wflow tests: `tests/wflow_runs/**` and `tests/flood_rm/wflow_event_window_test.py`.

Deletion-test decisions:

- `src/wflow_runs/__init__.py` duplicate `__all__` entries: `REDUNDANT_OR_REPLACEABLE`, `NOTEBOOK_API`. Removed duplicate names only; public objects remain imported once.
- `src/wflow_runs/replay.py` duplicated `_streamflow_records_path`, `_streamflow_members_path`, `_split_site_list`, and `_finite_float`: `REDUNDANT_OR_REPLACEABLE`, `PLUMBING_IO`, with streamflow semantics guarded as `CORE_SCIENCE`. Reused the identical helpers from `streamflow_realization.py`, which `replay.py` already imports for Same-Frequency Amplification.
- `src/wflow_runs/notebook.py` lazy wrappers: kept. They are notebook-facing and facade-facing call sites still depend on them.
- `src/wflow_runs/build_plan.py`: kept. It remains large because it owns Wflow domain planning, source strategy, staticmap QA, HydroMT build steps, gauge/handoff placement, and reservoir checks. Splitting/removal needs deeper characterization.

## Changed Files

- `src/wflow_runs/__init__.py`: removed duplicate `__all__` names. Unique public facade names and imports are preserved.
- `src/wflow_runs/replay.py`: removed duplicate private streamflow path/list/scalar helpers and imported the same implementations from `streamflow_realization.py`. Replay-specific member metadata validation remains local.
- `docs/agent_reports/wflow_runs.md`: added this report.

Line counts for files edited in this pass:

| File | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `src/wflow_runs/__init__.py` | 170 | 160 | -10 |
| `src/wflow_runs/replay.py` | 1191 | 1165 | -26 |
| Total | 1361 | 1325 | -36 |

## Preserved APIs And Contracts

- Preserved `wflow_runs` facade imports and unique `__all__` entries.
- Preserved notebook imports from `wflow_runs.notebook`.
- Preserved `build_meteo`, `resolve_event_window`, `configured_event_window_hours`, `replay_inland_domain_set`, `write_event_streamflow_handoff_discharge`, and all Wflow handoff public function names.
- No changes to SFINCS handoff NetCDF/JSON file names, discharge units, timestamps, routing, event-window semantics, scenario folder conventions, streamflow realization semantics, or Wflow staticmap assumptions.
- Did not run Wflow, SFINCS, large downloads, or credential-dependent workflows.

## Needs Evidence

- `build_plan.py` still needs a facade-preserving characterization/split pass before reducing domain planning, staticmap QA, reservoir readiness, and HydroMT build execution.
- `dynamic_handoff_batch_worklist` currently fails the accepted/blocked event fixture; this is a Wflow/SFINCS readiness contract issue needing coordination with Task 4c.
- `build_meteo` still has the known NetCDF overwrite/permission failure when an opened dataset is rewritten; fixing it would touch file-write behavior and should be handled with a focused test.
- `notebook.py` wrappers are shallow but notebook/public-facade evidence keeps them live. Migrate only with a notebook import migration note.
- `visualize.py`, `calibration.py`, and `streamflow_realization.py` remain high-value candidates, but no safe deletion evidence was produced in this pass.

## Validation

Commands used `UV_CACHE_DIR=/tmp/uv-cache` because the default uv cache under the home directory is read-only in this sandbox.

- `env UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall src`: passed.
- `env UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest tests/wflow_runs tests/flood_rm/wflow_event_window_test.py`: failed with 2 failures, 102 passed, 1 warning.
  - `tests/wflow_runs/dynamic_handoff_batch_test.py::test_dynamic_handoff_batch_worklist_selects_blocked_events`
  - `tests/wflow_runs/wflow_build_plan_test.py::test_build_meteo_writes_scaled_precip_and_neutral_pet`
- `env UV_CACHE_DIR=/tmp/uv-cache uv run python -m pytest`: failed with 31 failures, 775 passed, 8 warnings in 178.85s.
  - Failure themes: missing/changed cluster SLURM helpers, real-location notebook/artifact drift, Austin/Greensboro domain expectations, grid-source input contracts, SFINCS/Wflow coupled readiness, the dynamic handoff worklist fixture, and the Wflow NetCDF overwrite case.

## Cross-Domain TODOs

- Coordinate with SFINCS on accepted dynamic handoff readiness status and `audit_inland_coupled_batch_readiness`.
- Coordinate with integration lane on cluster SLURM helper expectations and notebook drift tests.
- Coordinate with shared infrastructure/location lane on Austin selected-domain drift (`austin_p4u` versus `austin_p5u`) and Greensboro missing Wflow gauge artifacts.

## Coordinator Follow-Up: Handoff Locations Module

User-requested cross-domain follow-up after Tasks 4c/4d:

- Folded the former `src/wflow_runs/coupled_handoff/__init__.py` package into the flatter
  `src/wflow_runs/handoff_locations.py` module.
- Updated Wflow imports in `build_plan.py`, `dynamic_handoff.py`, and `replay.py`.
- Kept the shared SFINCS/Wflow handoff contract intact: mode constants, config mode
  interpretation, SFINCS handoff source construction, and generated handoff artifact
  readers.
- Made the old `candidate_handoff_source_paths()` helper private as
  `_candidate_handoff_source_paths()` because no external source or test call sites used it.
- Removed the old `src/wflow_runs/coupled_handoff/` source path.

Validation:

- `UV_CACHE_DIR=/tmp/uv-cache uv run python -m compileall src`: passed.
- `uv run python -m pytest tests/wflow_runs/test_domain_set_crossings.py tests/wflow_runs/wflow_build_plan_test.py tests/sfincs_runs/test_stream_boundary_crossings.py tests/sfincs_runs/build_base/inland_base_test.py`: 101 passed, 1 failed, 4 warnings. The failure is the pre-existing Wflow `build_meteo` NetCDF overwrite permission case, not the handoff-location move.
