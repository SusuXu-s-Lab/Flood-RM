# Grill Review: Refactor Roadmap Stress Test

This is an adversarial review of the codebase-mapping pass. It reviews:

- `docs/codebase_map.md`
- `docs/notebook_backend_trace.md`
- `docs/refactor_roadmap.md`
- `docs/reduction_candidates.md`
- `docs/agent_work_packages.md`
- `docs/audit/file_inventory.json`
- `docs/audit/notebook_backend_trace.json`
- `docs/audit/reduction_candidates.json`

## Assumptions

- This review does not authorize source edits.
- AST/notebook inventories are useful evidence, but they are not complete call graphs.
- A notebook without `src/` imports can still be a public workflow contract if it reads or
  writes Grid Dataset, Event Catalog, SFINCS, Wflow, FIAT, or evaluation artifacts.
- `REDUNDANT_OR_REPLACEABLE` means "review for simplification," not "safe to delete."

## Findings

### BLOCKER: Call-site evidence is incomplete for deletion decisions

Evidence:

- The inventory scans notebooks and Python files, but the human docs currently lean on
  "no `src/` imports" as a stale-code signal for several notebooks.
- Notebook-local artifact logic exists in `sds_plot.ipynb`, `psds_plot.ipynb`,
  `03_ops/overlay.ipynb`, and `05c_ship_calibrated.ipynb` even when no backend import is
  present.
- CLI scripts, tests, package facades, external user imports, and string/dynamic imports
  can keep code live even when notebook call sites are absent.

Decision:

- No deletion or archival is allowed from the current inventory alone.
- Before deletion, require at least: `rg` call-site search, tests import search, notebook
  JSON search, package `__all__` review, script entry-point review, and artifact
  dependency review.

### BLOCKER: Some artifact/schema modules were classified too weakly

Evidence:

- `src/power/exports/smart_ds_grid.py` was classified only as
  `REDUNDANT_OR_REPLACEABLE`, but it defines SMART-DS-compatible assets, control units,
  asset states, telemetry, schemas, stable seeds, and manifests.
- `src/power/exports/restoration.py` was classified only as
  `REDUNDANT_OR_REPLACEABLE`, but it defines ONM settings, event windows, DSS rendering,
  asset-state events, and run bundles.
- `src/power/resilience/der.py` and `src/power/resilience/profiles.py` were not
  classified as core Grid Dataset semantics in the first pass.

Decision:

- Treat these modules as `CORE_SCIENCE` plus `NOTEBOOK_API` or `PLUMBING_IO` where
  applicable until tests prove a slice is pure plumbing.
- Splitting is allowed before simplification; deletion is not.

### HIGH: Missing core scientific concepts in the roadmap

The roadmap mentions Event Catalogs, Wflow/SFINCS handoff, and switch allocation, but it
does not yet foreground several glossary concepts that must constrain refactors:

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

Decision:

- Every package report must state which of these concepts it touches.
- Refactors that alter any of these concepts require explicit characterization tests and
  a migration note.

### HIGH: Notebook-to-backend trace misses artifact-only notebooks

Evidence:

- The JSON trace reports no backend entry points for several notebooks, including
  `locations/*/01_grid/sds_plot.ipynb`,
  `locations/marshfield/01_grid/psds_plot.ipynb`,
  `locations/*/03_ops/overlay.ipynb`, and
  `locations/austin|greensboro/02_flood/05c_ship_calibrated.ipynb`.
- These notebooks still read/write artifacts that are part of the public workflow.

Decision:

- The trace must classify these as `artifact-facing notebooks`, not stale notebooks.
- A future trace pass should record dataframe/file operations even when no `src/` import
  exists.

### HIGH: Split before simplify for oversized science files

Evidence:

- `src/wflow_runs/build_plan.py`, `src/sfincs_runs/build_base/region_notebook.py`,
  `src/power/resilience/switches.py`, `src/design_events/collect_sources/workflow.py`,
  and `src/study_location.py` mix multiple responsibilities.
- The risk is not just line count; it is that public interfaces expose too many
  responsibilities and make invariants hard to locate.

Decision:

- First pass for these files must be a facade-preserving split with tests.
- Do not combine the split with behavior simplification.

### HIGH: Parallel ownership conflicts need tightening

Evidence:

- Package 2 originally owned all `src/design_events/*`, while Package 6 also owned
  `src/design_events/plotting.py`.
- Package 3 owned all `src/sfincs_runs/*`, while Package 6 wanted
  `src/sfincs_runs/diagnostics.py`.
- Package 4 owned all `src/wflow_runs/*`, while Package 6 wanted
  `src/wflow_runs/visualize.py`.
- Package 7 owned broad `tests/*`, overlapping with domain test ownership.

Decision:

- Plotting/reporting is an audit-only package until a specific domain package delegates
  files to it.
- Tests/docs/integration may add cross-domain tests but must not edit domain-owned tests
  in parallel.
- No source file may be co-owned by two active packages.

### MEDIUM: Duplicate names are over-easy to misread as duplicate behavior

Evidence:

- `load_runtime`, `load_config`, `write_manifest`, `parse_args`, and `main` repeat
  across domains.
- Some repeated names are intentional local interfaces or CLI conventions.

Decision:

- Repeated name is only a review trigger.
- Consolidation requires matching schemas, call order, error behavior, path semantics,
  and side effects.

### MEDIUM: Competing implementations need evidence before removal

Review these as "needs evidence," not deletion candidates:

- Source collection readiness paths in `design_events.collect_sources.workflow` versus
  source-specific modules.
- Static input collection in `sfincs_runs.build_base.region_notebook` versus
  `sfincs_runs.build_base.static_intake`.
- Wflow domain planning variants in `wflow_runs.build_plan`.
- SFINCS event staging paths in `sfincs_runs.scenarios.create_events`,
  `run_events`, `run_inland_coupled_events`, and `inland_coupled`.
- ONM export behavior split between `scripts/onm_export.py` and
  `power.exports.restoration`.
- Former power artifact/notebook location resolution versus `study_location.py` and
  `sfincs_runs.config`; the power facades have since been deleted after caller
  migration.

Decision:

- For each pair, document callers and artifacts first. Remove only after one path is
  proven stale or becomes a compatibility shim with a migration note.

### MEDIUM: Validation needs location-aware commands

Evidence:

- `uv run python -m pytest` failed during collection without a selected location.
- `FLOOD_RM_LOCATION=marshfield uv run python -m pytest` ran the suite and exposed the
  current baseline failures.

Decision:

- Refactor validation should include both the no-location import expectation and a
  location-scoped pytest command.
- Package-level commands should set `FLOOD_RM_LOCATION=marshfield` unless the package is
  explicitly validating no-location behavior.

### LOW: Docs still need a "new reader path"

The docs map modules, but a new user still needs a one-page reading path:

1. Grid Notebook Workflow.
2. Flood Notebook Workflow.
3. Location Configuration and Location Detail Configuration.
4. Source Artifacts to Event Catalog.
5. Hydrodynamic Truth Set to Evaluation Layers.
6. Grid Dataset to operation/export artifacts.

Decision:

- Add this to a future `docs/codebase_map.md` update or README cleanup.

## Updated Refactor Gate

Before any source-code refactor:

1. Confirm the source file is owned by exactly one active work package.
2. Identify whether the edit touches any core concept listed above.
3. Add or identify characterization tests for notebook imports, artifact paths, schema
   columns, and expected side effects.
4. Run package-specific tests with `FLOOD_RM_LOCATION=marshfield`.
5. Run `uv run python -m compileall src`.
6. For deletion, prove the code is not reachable from notebooks, facades, scripts, tests,
   dynamic imports, or artifact-only workflows.

## Next Refactor Pass Order

1. Add tests and trace improvements, especially notebook import and artifact-only
   notebook smoke checks.
2. Split `src/wflow_runs/build_plan.py` behind existing facades.
3. Split `src/sfincs_runs/build_base/region_notebook.py` behind existing notebook calls.
4. Split `src/power/resilience/switches.py` only after SSAP and Block Invariant Contract
   characterization tests.
5. Split `src/design_events/collect_sources/workflow.py` after source-artifact path and
   readiness table characterization.
6. Split `src/study_location.py` after Location Definition and Model Recipe
   Configuration tests are green.
