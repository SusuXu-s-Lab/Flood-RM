# Shared Infrastructure Refactor Report

## Scope And Assumptions

Assumption: this pass executed Task 4e, Shared Infrastructure, because
`docs/agent_work_packages.md` recommends starting there before parallel domain work.

Edited files:

- `src/study_location.py`
- `src/paths.py`
- `src/aoi.py`
- `src/_study_location_recipes.py` deleted.
- `src/_study_location_templates.py` deleted.
- `pyproject.toml`
- `tests/flood_rm/study_location_shared_infrastructure_test.py`
- `docs/codebase_map.md`
- `docs/refactor_roadmap.md`
- `docs/reduction_candidates.md`
- `docs/agent_work_packages.md`
- `docs/tasks/README.md`
- `docs/agent_reports/shared_infrastructure.md`

Packaging assumption: `pyproject.toml` was edited only to ship the new top-level
`paths.py` and `aoi.py` helpers with the existing top-level `study_location.py`
facade. No dependency change was made.

Files intentionally not edited:

- `src/power/**`
- `src/design_events/**`
- `src/sfincs_runs/**`
- `src/wflow_runs/**`
- `src/fiat_runs/**`
- notebooks under `locations/**`
- cross-domain tests such as `tests/test_architecture_smoke.py`
- `scripts/run_pipeline.py`

## Public Interface Preserved

The public import seam remains `study_location`. Existing notebook and backend imports
continue to resolve through that facade, including:

- `find_repo_root`
- `default_location_config_path`
- `resolve_repo_path`
- `load_location_config`
- `build_study_area`
- `study_area_bbox`
- `StudyLocation`
- `LocationDefinition`
- `define_location`
- `resolve_study_location`

The older AOI helper exports and resolved-config writer were removed from the runtime
facade. `tests/flood_rm/show_resolved_config.py` owns resolved-config file writing.

## Changes Made

- Moved repository/path resolution and GeoJSON IO into a small shared `src/paths.py`.
- Deleted `src/_study_location_paths.py`.
- Kept AOI construction in a short `src/aoi.py` module and updated callers to match its
  dict-style return metadata.
- Collapsed YAML detail loading and recursive config merge back into `src/study_location.py`
  after removing recipe inheritance.
- Deleted `src/_study_location_templates.py` after review. Runtime Study Location
  identity resolution now lives in `src/study_location.py`; starter-template/listing and
  `coastal_waves_enabled` helpers were removed because no source or notebook callers
  used them.
- Kept `src/study_location.py` as the stable facade around `define_location`,
  `LocationDefinition`, explicit location YAML merging, and Study Location identity.
- Deleted `src/data.yaml`; source/model defaults should live in owning domain modules
  or explicit location YAML, not behind `define_location()`.
- Removed the stage-order list, validator method, and resolved-config writer from
  `study_location.py`; the facade now stays closer to runtime loading only.
- Deleted `src/_study_location_recipes.py`; Wflow and SnapWave HydroMT recipes are now
  explicit in location YAML rather than inherited through Python.
- Added fast synthetic tests for the new shared helper modules and public facade exports.
- Updated the codebase map, roadmap, and reduction candidates to record the partial split.

## Removed Or Replaced Code

The broad runtime facade was reduced. Deleted `study_location.py` exports include the
old stage-order method, validator method, resolved-config writer, and private AOI helper
re-exports that no source callers used after the `aoi.py` rename.

The methodology-default dictionaries were removed from `study_location.py`, and
`src/data.yaml` was deleted. Values that are real location choices now need to be
explicit in `locations/<name>/*.yaml`; source/provider constants should move into the
domain modules that own the collection or model setup.

## Post-Review Architecture Correction

The `_study_location_*` split is a transitional organization step, not a requirement to
keep every helper module permanently. The path portion has already been flattened into
`src/paths.py`, AOI was flattened into `src/aoi.py`, and `study_location.py`,
`sfincs_runs.config`, `design_events.runtime`, and power runtime/export modules now share
the path helper.

Do not replace many local path helpers with a larger framework. The target is boring
shared conventions for repository root, Location Workspace root, and location-relative
path resolution, with domain modules still owning their artifact keys and side effects.
The current `paths.py` is intentionally minimal; explicit relative paths are
repo-relative rather than cwd-probing.

See `docs/plumbing_reduction_directive.md`.

## Guarded Concepts Touched

- `Model Recipe Configuration`: touched by moving HydroMT recipe loading/merging without
  changing merge semantics.
- `Location Configuration`: hidden default values were removed from `study_location.py`
  and the needed runtime contracts were made explicit in location YAML.
- `Source Artifact`: indirectly touched through Study Area and generated path helpers,
  but no artifact path conventions were intentionally changed.

No formulas, flood-driver generation code, SFINCS/Wflow coupling code, grid topology
logic, switch-allocation logic, SMART-DS schema, or file-format semantics were
intentionally changed.

## Deferred TODOs

- Fully deleting `study_location.py` requires a separate caller migration because
  `define_location`, `resolve_study_location`, `build_study_area`, and
  `study_area_bbox` are still imported by source modules and tests.
- Keep repeated domain `load_runtime` helpers as public local seams unless an integration
  pass proves matching call order, error behavior, path semantics, and side effects.
- Do not consolidate domain `write_manifest` helpers until each manifest schema and
  artifact side effect is characterized.

## Validation

- `uv run python -m compileall src` passed.
- Focused validation after the `aoi.py` rename and explicit-YAML replacement:
  `uv run python -m pytest tests/design_events/fit_history/driver_records_test.py::test_real_greensboro_records_load_and_pair
  tests/flood_rm/coverage_box_domain_convention_test.py::test_greensboro_plans_one_enclosing_wflow_watershed_for_selected_sfincs_box
  tests/flood_rm/coverage_box_domain_convention_test.py::test_austin_uses_one_selected_sfincs_box_and_one_enclosing_wflow_watershed
  tests/wflow_runs/wflow_build_plan_test.py::test_location_wflow_configs_use_installed_runner_command
  tests/wflow_runs/wflow_build_plan_test.py::test_locations_default_to_dynamic_wflow_handoff
  tests/wflow_runs/wflow_build_plan_test.py::test_greensboro_wflow_build_resolution_matches_collected_hydrography_resolution`
  completed with `2 failed, 4 passed, 1 warning`. The passing checks confirm the
  removed hidden defaults are now explicit YAML; the two failures are existing Austin
  `p5u` vs expected `p4u` and Greensboro domain-readiness drift.
- `uv run python -m pytest tests/flood_rm/study_location_shared_infrastructure_test.py
  tests/flood_rm/location_definition_test.py tests/flood_rm/model_recipe_contract_test.py
  tests/flood_rm/notebook_runtime_helpers_test.py tests/test_architecture_smoke.py`
  completed with `6 failed, 469 passed, 1 warning`. The failures are the known
  Greensboro notebook contract/cleanliness checks in `location_definition_test.py`.
- `uv run python -m pytest` completed with `32 failed, 785 passed, 8 warnings`.

The remaining full-suite failures are in cluster SLURM expectations, real-location
notebook/artifact drift, grid-source input contracts, SFINCS/Wflow coupled scenario
readiness, dynamic-handoff worklist behavior, and the existing Wflow NetCDF overwrite
behavior.
