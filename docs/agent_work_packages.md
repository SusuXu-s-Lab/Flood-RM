# Agent Work Packages

This file is the source-of-truth assignment map for the active refactor pass. It is
aligned to the task briefs in `docs/tasks/4a.md` through `docs/tasks/4e.md`.

The purpose of this split is to let agents work in parallel without editing the same
source files, tests, notebooks, or reports.

## Dispatch Order

Recommended execution:

1. Run the validation baseline and keep `tests/test_architecture_smoke.py` green.
2. Start Task 4e first or keep it audit-only while other agents work; shared path/config
   changes can affect every package.
3. Run Tasks 4a, 4b, 4c, and 4d in parallel only if each agent stays inside its assigned
   source and test ownership.
4. Do a separate integration pass after all domain reports exist.

Do not treat this as permission for broad refactors. Each task must preserve notebook
imports, artifact paths, scientific behavior, and file-format semantics.

## Global Coordination Rules

- No two active agents may edit the same source file.
- No two active agents may edit the same test file.
- No agent may edit notebooks during this pass unless the user explicitly assigns a
  notebook migration.
- Domain agents may write only their own report under `docs/agent_reports/`.
- Cross-domain docs, test harness files, and unresolved ownership questions belong to the
  integration lane, not to a domain agent.
- Plotting/reporting files belong to their owning domain. There is no separate plotting
  implementation agent in this pass.
- Repeated names such as `load_runtime`, `write_manifest`, `main`, and `parse_args` are
  review triggers, not automatic consolidation targets.
- Code labeled `REDUNDANT_OR_REPLACEABLE` still needs call-site and artifact evidence
  before deletion or replacement.
- Helper modules and compatibility facades are also review triggers, not automatic
  keepers. Read `docs/plumbing_reduction_directive.md` before preserving or expanding
  artifact, path, manifest, value-coercion, or runtime-config wrappers.

Every report must say whether the work touches any of these guarded concepts:

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

## Task 4a: Power/Grid Construction And Audit

Prompt:

- `docs/tasks/4a.md`

Owns source:

- `src/power/**`

Owns tests:

- Power-specific tests under `tests/flood_rm/`, currently including:
  - `tests/flood_rm/grid_dataset_workflow_test.py`
  - `tests/flood_rm/grid_source_inputs_test.py`
  - `tests/flood_rm/power_restoration_test.py`
  - `tests/flood_rm/power_artifacts_test.py`
- Any new tests placed under a future `tests/power/` directory.

Owns report:

- `docs/agent_reports/power.md`

Primary workload:

- Map and simplify module boundaries for grid construction, baseline network inputs,
  network augmentation, switch synthesis, load blocks, impact/resilience, audit, plotting,
  SMART-DS-compatible exports, and ONM/restoration exports.
- Treat `src/power/resilience/switches.py`, `src/power/exports/smart_ds_grid.py`,
  `src/power/exports/restoration.py`, `src/power/audit/synthetic_validation.py`, and
  `src/power/plotting.py` as high-leverage split candidates.
- Do not change SSAP, topology, switch allocation, Stable Grid ID, Block Invariant
  Contract, SMART-DS-compatible schema, DER sizing, or ONM event-window behavior without
  characterization tests.

Must coordinate with:

- Task 4e for location/path/config conventions.
- Integration lane for cross-domain notebook tests and architecture smoke tests.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest tests/flood_rm/grid_dataset_workflow_test.py tests/flood_rm/grid_source_inputs_test.py tests/flood_rm/power_restoration_test.py`

## Task 4b: Design Events And Source Collection

Prompt:

- `docs/tasks/4b.md`

Owns source:

- `src/design_events/**`

Owns tests:

- `tests/design_events/**`

Owns report:

- `docs/agent_reports/design_events.md`

Primary workload:

- Map and simplify source collection, historical fitting, stochastic event generation,
  probability catalogs, timing, selection, plotting, and utility logic.
- Treat `src/design_events/collect_sources/workflow.py`,
  `src/design_events/collect_sources/national_hydrography.py`,
  `src/design_events/collect_sources/aorc_sst.py`,
  `src/design_events/plotting.py`, `src/design_events/build_events/selection.py`, and
  `src/design_events/build_events/coastal.py` as high-leverage split candidates.
- Do not alter tail sampling, probability weighting, dependence modeling, historical
  analog pairing, driver records, event reference time, timing descriptors, or
  event-member semantics without explicit equivalence evidence.

Must coordinate with:

- Task 4d on streamflow event members and Wflow handoff catalog semantics.
- Task 4c on SFINCS scenario forcing expectations.
- Integration lane for artifact-only notebook trace gaps.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest tests/design_events`

## Task 4c: SFINCS/SnapWave Setup And Scenarios

Prompt:

- `docs/tasks/4c.md`

Owns source:

- `src/sfincs_runs/**`

Owns tests:

- `tests/sfincs_runs/**`

Owns report:

- `docs/agent_reports/sfincs_runs.md`

Primary workload:

- Map and simplify SFINCS base setup, static intake, structures, hydrology helpers,
  SnapWave setup, tide gauges, scenario forcing, diagnostics, outcome catalogs, and
  notebook helpers.
- Treat `src/sfincs_runs/build_base/region_notebook.py`,
  `src/sfincs_runs/build_base/inland_base.py`, `src/sfincs_runs/diagnostics.py`, and
  `src/sfincs_runs/scenarios/inland_coupled.py` as high-leverage split candidates.
- Do not alter SFINCS input-file semantics, coordinate handling, water-level forcing,
  wave forcing, subgrid flood layers, static-intake assumptions, scenario folder
  structure, or Wflow-SFINCS handoff geometry without characterization tests.

Must coordinate with:

- Task 4d on dynamic handoff file formats, source locations, timestamps, units, and
  routing semantics.
- Task 4e on HydroMT recipe and Location Configuration behavior.
- Integration lane for cluster SLURM and cross-domain notebook tests.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest tests/sfincs_runs`

## Task 4d: Wflow Coupling And Handoff

Prompt:

- `docs/tasks/4d.md`

Owns source:

- `src/wflow_runs/**`

Owns tests:

- `tests/wflow_runs/**`
- `tests/flood_rm/wflow_event_window_test.py`

Owns report:

- `docs/agent_reports/wflow_runs.md`

Primary workload:

- Map and simplify Wflow domain planning, staticmap QA, source strategies, calibration,
  streamflow realization, replay, state handling, dynamic handoff, coupled handoff,
  river geometry, QA, visualization, and notebook helpers.
- Treat `src/wflow_runs/build_plan.py`, `src/wflow_runs/replay.py`,
  `src/wflow_runs/streamflow_realization.py`, `src/wflow_runs/calibration.py`, and
  `src/wflow_runs/visualize.py` as high-leverage split candidates.
- Do not alter SFINCS handoff file formats, timestamps, units, routing semantics,
  scenario folder conventions, streamflow realization semantics, or Wflow staticmap
  assumptions without characterization tests.

Must coordinate with:

- Task 4b on streamflow event members and design-event catalog semantics.
- Task 4c on dynamic handoff and SFINCS boundary inputs.
- Integration lane for cluster SLURM and cross-domain notebook tests.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest tests/wflow_runs tests/flood_rm/wflow_event_window_test.py`

## Task 4e: Shared Infrastructure

Prompt:

- `docs/tasks/4e.md`

Owns source:

- `src/study_location.py`
- `src/paths.py`
- `src/_study_location_*.py`
- `src/generated_artifact.py` is no longer owned because it was deleted; keep generated
  YAML notices local to writer modules.
- `scripts/run_pipeline.py`, if present and relevant to location/runtime plumbing.
- `pyproject.toml`, only for packaging metadata required by shared-infrastructure helper
  modules.

Owns tests:

- `tests/flood_rm/location_definition_test.py`
- `tests/flood_rm/model_recipe_contract_test.py`
- `tests/flood_rm/notebook_runtime_helpers_test.py`
- Any new tests for `src/study_location.py` or shared path/AOI helpers.

Owns report:

- `docs/agent_reports/shared_infrastructure.md`

Primary workload:

- Clarify Location Configuration, Location Detail Configuration, HydroMT recipe includes,
  path resolution, generated-artifact metadata, notebook runtime conventions, and
  repeated manifest/config helpers.
- Treat `src/study_location.py` as a high-risk shared facade: split behind stable public
  imports before simplification.
- Keep private helper modules packageable when the top-level `study_location.py` facade
  depends on them.
- Prefer documenting cross-domain helper opportunities over forcing migrations during
  domain-agent work.

Must coordinate with:

- Every domain task before changing path, config, recipe, or generated-artifact behavior.
- Integration lane before touching cross-domain test harness files.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest tests/flood_rm/location_definition_test.py tests/flood_rm/model_recipe_contract_test.py tests/flood_rm/notebook_runtime_helpers_test.py`

## Reserved Integration Lane

This lane is not one of the five active domain task briefs. Use it for final coordination,
documentation consolidation, baseline updates, and cross-domain validation only.

Current next-pass decision: run an integration-led deletion bridge before assigning broad
domain-led big-file splits. The bridge should preserve notebook behavior while proving
call sites and deleting shallow compatibility wrappers or redundant plumbing after
caller migration. Do not use the bridge to change guarded science or artifact semantics.

Owns docs and audit artifacts:

- `AGENTS.md`
- `docs/codebase_map.md`
- `docs/notebook_backend_trace.md`
- `docs/refactor_roadmap.md`
- `docs/reduction_candidates.md`
- `docs/agent_work_packages.md`
- `docs/audit/**`
- `docs/tasks/**`

Owns cross-domain tests:

- `tests/conftest.py`
- `tests/test_architecture_smoke.py`
- `tests/cluster/**`
- `tests/flood_rm/coverage_box_domain_convention_test.py`
- `tests/flood_rm/greensboro_notebook_setup_test.py`
- Cross-domain notebook, artifact, or architecture tests added later.

Reserved, not assigned in Tasks 4a-4e:

- `src/fiat_runs/**`
- `tests/fiat_runs/**`
- `tests/test_fiat_diagnostics.py`
- Notebooks under `locations/**`
- Generated figures and location workspace artifacts.

Integration responsibilities:

- Merge domain reports into the roadmap.
- Resolve conflicts between domain proposals.
- Update notebook-backend traces, especially artifact-only notebooks.
- Own deletion-bridge decisions that cross domain boundaries or notebook setup cells.
- Decide whether FIAT needs a separate task brief before any FIAT source edits.
- Run full validation and document current failures.

Validation:

- `uv run python -m compileall src`
- `uv run python -m pytest tests/test_architecture_smoke.py`
- `uv run python -m pytest`

## Current Full-Suite Baseline

The validation harness pass established that `uv run python -m pytest` now collects and
runs with `FLOOD_RM_LOCATION` defaulted to `marshfield` in `tests/conftest.py`.

Current baseline from the latest run:

- `33 failed`
- `773 passed`
- `8 warnings`

Known failure themes include cluster SLURM expectations, real-location notebook/artifact
drift, Greensboro/Austin setup expectations, grid-source input contracts, SFINCS/Wflow
coupled scenario readiness, and one Wflow build-plan behavior check.

Domain agents should not fix unrelated baseline failures outside their owned files. If a
domain change changes the failure count, the report must explain why.
