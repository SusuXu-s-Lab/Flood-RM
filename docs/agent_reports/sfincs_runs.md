# SFINCS/SnapWave Task 4c Report

## Scope

Owned files edited in this pass:

- `src/sfincs_runs/scenarios/scenarios.py`
- `src/sfincs_runs/scenarios/create_events.py`
- `src/sfincs_runs/scenarios/run_events.py`
- `src/sfincs_runs/scenarios/inland_coupled.py`
- `tests/sfincs_runs/scenarios/inland_coupled_test.py`

The working tree already contained broader SFINCS deletion-bridge edits before this
pass, including removal of `sfincs_runs.notebook`, `sfincs_runs.scenarios.io`,
`sfincs_runs.build_base.region_setup`, and `sfincs_runs.single_use_case`. I did not
recreate those facades.

Guarded concepts touched: `Forcing Support Window`, `Hydrodynamic Truth Set`,
`Wave-Coupled Truth Set`, `Static Intake`, `Subgrid Flood Layer`, and Wflow-SFINCS
handoff readiness. This pass did not change SFINCS input-file keys, coordinate handling,
water-level forcing, wave forcing, precipitation forcing, subgrid outputs, scenario
folder structure, or handoff geometry.

## Call-Site Evidence

- `rg` for deleted facades found no live source, test, script, or notebook imports outside
  stale audit JSON for:
  - `sfincs_runs.notebook`
  - `sfincs_runs.scenarios.io`
  - `sfincs_runs.build_base.region_setup`
  - `sfincs_runs.single_use_case`
- `rg` found `scenario_static_files`, `read_design_inputs`, and `ensure_static_files`
  only in `src/sfincs_runs/scenarios/scenarios.py`; no notebook, source, script, or
  SFINCS test callers.
- `rg` found `default_design_outputs`, `default_base_model`, `default_scenarios`,
  `default_scenarios_root`, `default_storage_root`, and `default_run_root` only as
  import-time globals in `create_events.py` and `run_events.py`.
- Notebook-facing modules remain live: `region_notebook`, `diagnostics`,
  `event_forcing`, `create_events`, and the `sfincs_runs.scenarios` inland-coupled
  facade have notebook/test call sites.

## Reductions And Fixes

- `REDUNDANT_OR_REPLACEABLE` / `STALE_OR_DANGEROUS`: removed unused scenario-static
  file list plus stale `read_design_inputs()` and `ensure_static_files()` from
  `scenarios.py`. Current staging semantics remain owned by `event_forcing.stage_run()`
  and inland coupled staging.
- `PLUMBING_IO`: removed unused import-time runtime path globals from `create_events.py`
  and `run_events.py`. Both CLIs still resolve defaults inside `parse_args()` through
  `load_runtime()`.
- `VALIDATION_QA` / `CORE_SCIENCE`: fixed `handoff_readiness()` to pass its
  `catalog_path` through to Wflow's `require_handoff()`, preserving the current dynamic
  handoff window check instead of falling through to compatibility diagnostics.
- `VALIDATION_QA`: updated the inland-coupled readiness fixture to use a tiny
  time-indexed NetCDF discharge artifact plus a catalog reference time, matching the
  current accepted dynamic handoff contract.

## Preserved Public Interfaces

- `sfincs_runs.scenarios.create_events.build_scenarios`
- `sfincs_runs.scenarios.create_events.parse_args`
- `sfincs_runs.scenarios.event_forcing.build_event`
- `sfincs_runs.scenarios.event_forcing.stage_run`
- `sfincs_runs.scenarios.event_forcing.stage_precip`
- `sfincs_runs.scenarios.event_forcing.run_model`
- `sfincs_runs.scenarios.plan_example`
- `sfincs_runs.scenarios.stage_inland_coupled_example_forcing`
- `sfincs_runs.scenarios.stage_inland_coupled_scenario_forcing`
- `sfincs_runs.scenarios.audit_inland_coupled_batch_readiness`
- `sfincs_runs.scenarios.scenarios.build_event_timeseries`
- `sfincs_runs.scenarios.scenarios.select_zsini_from_series`
- `sfincs_runs.scenarios.scenarios.assert_event_catalog_audit`

## Line Counts

Measured with `wc -l`.

| File | Before | After | Delta |
| --- | ---: | ---: | ---: |
| `src/sfincs_runs/scenarios/scenarios.py` | 98 | 43 | -55 |
| `src/sfincs_runs/scenarios/create_events.py` | 487 | 482 | -5 |
| `src/sfincs_runs/scenarios/run_events.py` | 274 | 269 | -5 |
| `src/sfincs_runs/scenarios/inland_coupled.py` | 901 | 901 | 0 |
| `tests/sfincs_runs/scenarios/inland_coupled_test.py` | 688 | 696 | +8 |

Net source reduction in files edited by this pass: 65 lines.

## Needs-Evidence Candidates

- `src/sfincs_runs/build_base/region_notebook.py`: still mixes notebook orchestration,
  static intake, coverage preflight, and plotting. It is `CORE_SCIENCE` and
  `NOTEBOOK_API`; split or reduce only with notebook artifact evidence.
- `src/sfincs_runs/build_base/inland_base.py`: Wflow-native river handoff placement and
  SFINCS domain planning remain `CORE_SCIENCE`; do not simplify without focused
  characterization tests.
- `src/sfincs_runs/diagnostics.py`: large `PLOTTING_REPORTING` plus outcome-stat module;
  keep notebook-facing names stable if split later.
- `src/sfincs_runs/scenarios/inland_coupled.py`: scenario staging overlaps with Wflow
  batch tooling and cluster scripts; further reduction needs cross-domain ownership.
- `src/sfincs_runs/scenarios/scenario_stats.py`: import-time defaults are still used by
  the module's notebook-facing summary helpers; needs separate characterization before
  removing.

## Validation

- `uv run python -m compileall src`: passed after final source edits.
- `uv run python -m pytest tests/sfincs_runs`: passed, 65 passed, 2 warnings.
- `uv run python -m pytest`: failed, 779 passed, 30 failed, 8 warnings in 181.13s.

Full-suite failure themes were outside this pass's edit surface:

- Missing or outdated cluster helpers, especially
  `cluster/run_sfincs_dsai_inland_coupled.slurm` and wave-coupled sync/default
  expectations.
- Cross-domain notebook/location drift in Greensboro and Austin coverage-box,
  region-setup, coupled-model notebook, and clean-notebook expectations.
- Power/grid source-input expectations after concurrent grid/location edits.
- Wflow dynamic handoff batch fixture still uses a non-NetCDF discharge fixture for an
  accepted handoff; this mirrors the SFINCS fixture issue fixed here but belongs to
  Task 4d ownership.
- Wflow `build_meteo` NetCDF overwrite permission failure remains in
  `tests/wflow_runs/wflow_build_plan_test.py`.

## Cross-Domain TODOs

- Task 4d should update Wflow dynamic handoff tests to use current time-indexed NetCDF
  acceptance fixtures and pass catalog paths where the handoff-window validator needs
  event reference time.
- Integration lane should reconcile cluster helper expectations and generated
  `cluster/run_*` paths before treating full-suite failures as SFINCS regressions.
- Integration lane should refresh shared docs/audit JSON after all domain reports land;
  stale audit JSON still references deleted SFINCS facades.

## Coordinator Follow-Up: Wflow Handoff Locations Import

User-requested cross-domain follow-up after Tasks 4c/4d:

- Updated `src/sfincs_runs/build_base/inland_base.py` to import the shared Wflow/SFINCS
  handoff contract from `wflow_runs.handoff_locations` instead of the deleted
  `wflow_runs.coupled_handoff` package.
- No SFINCS handoff placement, source artifact schema, SFINCS domain planning, or
  Wflow-native river inflow behavior changed.

Validation:

- Covered by the coordinated handoff test command recorded in
  `docs/agent_reports/wflow_runs.md`.
