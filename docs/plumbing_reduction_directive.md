# Plumbing Reduction Directive

This note records the June 27, 2026 architecture clarification for future refactor
passes. It overrides any interpretation that helper-module splits are the desired final
state.

## Direction

Flood-RM should become smaller, flatter, and more idiomatic scientific Python. Splitting
a large file behind a stable facade is useful when it makes behavior visible and safer to
test, but it is not the end goal. After characterization, agents should remove, inline,
or replace thin wrappers with direct `pandas`, `numpy`, `geopandas`, `shapely`,
`xarray`, `pathlib`, `pyarrow`, and standard-library calls whenever the wrapper does not
encode a real Flood-RM domain invariant.

Treat every new private helper as a cost. Keep it only when it protects a documented
domain contract or removes repeated nontrivial code from several call sites.

This is a `src/`-wide rule, not a power-specific rule. The deleted
`power.artifacts.py` module was only the first obvious example. The same deletion test applies to repeated helpers in
`design_events`, `sfincs_runs`, `wflow_runs`, `fiat_runs`, `study_location.py`, scripts,
and future packages.

## Src-Wide Plumbing Targets

Review these patterns before preserving, moving, or adding them:

- Path helpers such as `_location_path`, `_repo_path`, `_repo_root`,
  `resolve_*_path`, and module-level `default_*_root` constants.
- Runtime/config pass-throughs such as local `load_config`, `resolve_path`, and
  `load_runtime` wrappers. Notebook-facing `load_runtime` functions can remain public,
  but their internals should use shared, direct path conventions.
- Source/artifact helpers such as `source_artifact_path`, `read_source_artifact`,
  `write_source_artifact`, generic `write_manifest`, and optional hash wrappers.
- Scalar coercion helpers such as `parse_float`, `parse_int`, `finite_float`,
  `_present`, and slug/token helpers when direct pandas, numpy, or local code is clearer.
- Compatibility branches that silently accept several historical path layouts without a
  current notebook or artifact contract requiring them.

The preferred shape is:

- one tiny `src/paths.py` for repo/location path basics;
- domain modules own their artifact keys, schemas, and side effects;
- table-heavy modules use pandas/numpy/geopandas directly;
- ID helpers live beside the artifact writer that owns the ID contract;
- compatibility facades are deleted once imports are migrated.

Do not replace scattered wrappers with a larger framework. The target is fewer concepts,
not a more central concept.

## Power Artifact Helpers

Former transitional files removed by the follow-up artifact cleanup:

- `src/power/_artifact_ids.py`
- `src/power/_artifact_io.py`
- `src/power/_artifact_paths.py`
- `src/power/_artifact_values.py`

These files preserved behavior during the first power pass, then failed the deletion
test. The follow-up pass also deleted `src/power/artifacts.py` after migrating callers
to `src/paths.py`, `study_location.py`, and local artifact-owned helpers.

### Stable Grid IDs

`Stable Grid ID` is a real Grid Dataset contract: IDs must stay deterministic,
location-namespaced, and independent of row order. That does not mean Flood-RM needs a
separate `artifact_ids` module.

Preferred next state:

- Keep tests that assert the actual ID strings and namespace rules.
- Generate IDs directly where artifact rows or DataFrames are built when that is clearer.
- Use vectorized string operations where IDs are derived from a DataFrame column.
- Keep at most one tiny slug/token helper if repeated deterministic normalization remains
  nontrivial.

`stable_asset_id()` and `stable_control_unit_id()` are not scientific algorithms. They
now live as private helpers beside the SMART-DS-compatible rows they name; future
cleanup can inline or vectorize them if tests keep the published ID strings stable.

### Artifact IO

Generic IO wrappers are suspect by default.

- `write_parquet()` should be replaced by direct `pyarrow` or `pandas` calls at the
  artifact writer unless the helper is demonstrably preserving a published schema
  convention in several places.
- `maybe_sha256()` is likely over-protective. Prefer explicit required-artifact and
  optional-artifact manifest sections, or direct `path.exists()` checks where optionality
  is meaningful.
- `sha256()` and `short_hash()` may stay only if they remain widely used and clearer than
  local standard-library code.
- `validation_error(report, message)` should usually be replaced by direct list append or
  a clearer validation-table construction.
- `git_info()` is provenance plumbing; keep only if generated manifests actually require
  it and tests cover the manifest fields.

### Artifact Paths

The power-specific path modules have been replaced by `src/paths.py` plus local runtime
resolution in modules that still need Grid Dataset roots. The long-term target is not
one path mini-system per domain.

Preferred next state:

- Use one shared Study Location/runtime resolver for `repo_root`, `location_root`, and
  configured location-relative paths.
- Avoid import-time filesystem selection such as module-level `power_grid = ...` when a
  notebook/runtime object can pass explicit paths.
- Preserve notebook artifact paths and side effects until a migration note exists.
- Replace `sandbox_id` in code-facing names with `location_id`; keep legacy artifact
  fields only where existing published schemas require them.

### Artifact Values

Most value coercion helpers are candidates for removal.

- Prefer `pd.to_numeric(..., errors="coerce")`, `Series.notna()`, `np.isfinite`,
  `Series.between`, `fillna`, `astype`, `str.strip`, `str.lower`, and direct pandas
  grouping over hand-written row loops.
- Fail plainly where bad values indicate a broken artifact instead of silently returning
  defaults.
- Keep coordinate validity as a documented artifact validation rule, but implement it
  where validation happens, ideally vectorized.
- Keep slug/token normalization only if it remains part of the Stable Grid ID contract.

## Study Location Split Helpers

Current shared path and Study Location files:

- `src/paths.py`
- `src/aoi.py`

The old `_study_location_paths.py` helper has been replaced by `src/paths.py` so
`study_location.py`, `design_events`, `sfincs_runs`, and power artifact defaults point
at the same path helper module. The module is intentionally bare-minimum code: relative
explicit paths are repo-relative, Location Workspace paths resolve under
`locations/<name>/`, and GeoJSON helpers only read/write simple geometry features. The
remaining Study Location split is more defensible than the artifact split because it
separates distinct responsibilities: Study Area AOI geometry and HydroMT recipe merging.
The old `_study_location_templates.py` file failed the deletion test: runtime identity
resolution belongs in `study_location.py`, while starter-template/listing helpers had no
source or notebook callers.

## SFINCS Helper Facades

The same deletion test applies inside `src/sfincs_runs/`. The former
`sfincs_runs.notebook`, `sfincs_runs.scenarios.io`, `sfincs_runs.build_base.region_setup`,
and `sfincs_runs.single_use_case` modules were removed after their callers were migrated.
Keep SFINCS-specific file-format semantics explicit at the owning call site, but avoid
generic JSON/path/notebook pass-through modules.

Preferred next state:

- Keep `study_location.py` as the public facade until notebook/backend imports are
  intentionally migrated.
- Consolidate repeated path resolution across `study_location.py`,
  `sfincs_runs.config`, `design_events.runtime`, `wflow_runs`, `fiat_runs`, and
  grid-notebook setup into a very small shared runtime convention only after call order,
  environment behavior, and side effects are characterized. The former `power.notebook`
  facade and the former `design_events.utils` catchall have already been deleted after
  caller migration.
- Do not duplicate per-domain defaults behind generic path helpers. Domains should own
  their artifact keys; shared code should only resolve paths.
- Keep AOI and HydroMT recipe logic separate from generic plumbing because they encode
  Study Location and Model Recipe Configuration behavior.
- Move remaining model-recipe defaults only with recipe characterization tests for
  coastal and inland locations.

## Refactor Test

Before keeping a helper, answer:

1. Does it encode a named Flood-RM concept from `docs/CONTEXT.md`?
2. Is it used by several call sites with the same semantics?
3. Does it remove more complexity than it adds?
4. Would replacing it with a direct library call make the notebook-to-backend trace
   easier to follow?

If the answer to 1-3 is "no" and 4 is "yes", remove or inline it after tests prove the
artifact contract is preserved.
