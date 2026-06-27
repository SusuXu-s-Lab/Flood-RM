# Power/Grid Construction And Audit Refactor Report

## Scope And Assumptions

Assumption: this pass continued Task 4a with a narrow plumbing-reduction slice. The
goal was to delete shallow power facades while preserving Grid Dataset artifact
contracts, notebook call order, and scientific behavior.

Edited source and notebook files:

- `src/power/artifacts.py` deleted.
- `src/power/notebook.py` deleted.
- `src/power/exports/smart_ds_grid.py`
- `src/power/audit/synthetic_validation.py`
- `src/power/baseline_network/build_asset_registry.py`
- `src/power/impact/analysis.py`
- `src/power/impact/fragility.py`
- `src/power/plotting.py`
- `src/power/resilience/der.py`
- `src/power/resilience/facilities.py`
- `src/power/exports/restoration.py`
- Marshfield `locations/marshfield/01_grid/**/*.ipynb` setup cells.

Files intentionally not edited:

- SSAP and topology internals in `src/power/resilience/switches.py`.
- SMART-DS-compatible schemas beyond local helper placement in
  `src/power/exports/smart_ds_grid.py`.
- ONM event-window semantics in `src/power/exports/restoration.py`.
- Non-power science modules outside the shared path/runtime imports already touched.

## Public Interface Changes

No notebook-facing scientific function names were removed. The Marshfield grid
notebooks no longer import `power.notebook.load_runtime`; they now import
`sfincs_runs.config.load_runtime` and `sfincs_runs.config.build_grid_paths` directly in
their setup cells, then create the same Grid Dataset output directories.

`power.artifacts` is no longer a public seam. Source callers were migrated to:

- `src/paths.py` and `study_location.define_location()` for location/config roots.
- Module-local scalar parsing where the parser is only a caller convenience.
- Private ID helpers in `power.exports.smart_ds_grid` where Stable Grid ID strings are
  actually emitted.

## Changes Made

- Deleted the temporary `_artifact_*` helper modules and the final
  `src/power/artifacts.py` compatibility surface.
- Deleted `src/power/notebook.py` after migrating all Marshfield grid notebooks off it.
- Kept schema-aware parquet writes, file hashing, manifests, and Stage A1 Stable Grid ID
  construction beside the SMART-DS-compatible export rows they describe.
- Replaced broad public scalar/presence helpers with small local helpers at the few
  power call sites that still need them.
- Renamed the former power path test to `tests/flood_rm/power_restoration_test.py`
  because it now protects ONM/restoration behavior, not path plumbing.

## Guarded Concepts Touched

- `Stable Grid ID`: preserved by focused export tests that assert generated strings.
- `SMART-DS-Compatible Interface`: schema and manifest behavior stayed local to the
  export module.
- `Grid Notebook Workflow`: setup cells still derive `config`, `paths`, `grid`,
  `location_root`, `location_name`, and `repo_root`, and still create the expected output
  directories.

Not touched:

- `Block Invariant Contract`
- SSAP or switch allocation
- grid topology logic
- DER sizing policy
- ONM event-window behavior
- Synthetic Validation Audit scoring

## Deferred TODOs

- Apply the same deletion test to other `*.notebook.py` runtime facades only after
  notebook call sites are mapped and migrated.
- Split `src/power/resilience/switches.py` only after focused characterization around
  SSAP candidate selection, transformer-bridged feeder islands, Fuse Proxy handling, and
  Block Invariant Contract outputs.
- Split `src/power/exports/smart_ds_grid.py` by artifact/schema family only after fixture
  tests cover Stable Grid IDs, schema columns, manifests, and validation reports.
- Split `src/power/exports/restoration.py` only after ONM event windows, DSS text
  rendering, run bundles, and Julia/DynaGrid smoke setup are characterized without live
  Julia execution.

## Validation

- `uv run python -m compileall src`: passed.
- `uv run python -m pytest tests/flood_rm/power_artifacts_test.py
  tests/flood_rm/power_restoration_test.py
  tests/flood_rm/notebook_runtime_helpers_test.py tests/test_architecture_smoke.py`:
  passed with `452 passed, 1 warning`.
- `uv run python -m pytest`: completed with `33 failed, 785 passed, 8 warnings`.
  Failures are the existing baseline clusters in cluster SLURM expectations, real
  location/notebook drift, grid-source input contracts, inland coupled readiness,
  dynamic handoff batch selection, and Wflow NetCDF overwrite behavior.
