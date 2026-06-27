# Power Code Simplification Plan

This plan guides the cleanup of `src/power` so the Grid Dataset code stays auditable, concise, and aligned with the project language in `docs/CONTEXT.md`.

## Goal

Make `src/power` easier to read and maintain without changing the Grid Dataset contracts or the scientific meaning of the notebooks.

The cleanup should reduce custom wrappers, defensive fallback branches, and over-specific names while preserving published artifacts, validation gates, and stakeholder-facing workflow categories.

This is primarily a holistic code-line reduction effort. The default move is to remove or merge code, not add code. Prefer deleting local definitions, collapsing duplicate branches, renaming for clarity, and using package-native calls over adding replacement helpers or new abstraction layers. A new helper is justified only when it removes substantially more code than it adds and clarifies a real Grid Dataset concept.

## Resolved Rules

1. Use `location_id` as the canonical namespace term.
   - Replace code-facing `sandbox_id` language with `location_id`.
   - Keep legacy compatibility only where an existing artifact or environment variable would otherwise break a rerun.

2. Delete generic IO wrappers.
   - Use `pd.read_csv`, `DataFrame.to_csv`, normal imports, and package-native IO directly.
   - Keep only artifact-specific helpers that encode Grid Dataset semantics.
   - Remove `require_pyarrow`; dependency errors should be ordinary import errors.

3. Separate contract validation from diagnostic review.
   - Production builders enforce artifact correctness and fail plainly.
   - Diagnostic explanation, broad exception summaries, optional plots, and review tables belong in `power/audit/` or notebook-facing review helpers.

4. Keep explicit schemas only for published artifact contracts.
   - Durable Grid Dataset outputs keep explicit schemas.
   - Transient debug CSVs and notebook summaries do not need explicit schemas.
   - Schema function names should be short and artifact-owned, such as `der_schema()` or `load_match_schema()`.

5. Preserve current package categories first.
   - Keep the existing top-level categories: `baseline_network/`, `resilience/`, `exports/`, `impact/`, `audit/`, `artifacts.py`, and `plotting.py`.
   - Split oversized files only where the domain boundary is obvious after shortening internals.

6. Use staged passes with verification.
   - Avoid one huge rewrite.
   - Finish each file family with import checks, focused tests where available, and notebook JSON checks when notebooks change.

7. Optimize for net deletion.
   - Measure progress by reduced local definitions, reduced line count, and clearer call sites.
   - Do not replace one generic wrapper with another generic wrapper.
   - Avoid adding compatibility shims unless an existing artifact or notebook would otherwise break immediately.
   - Treat new helper functions as a cost. Merge or delete first; add only when the surrounding code shrinks meaningfully.
   - Prefer fewer files, fewer branches, and fewer private functions unless a split preserves a clear domain category and reduces total complexity.

8. Use the `<500` line target as a pressure test, not a hard law.
   - Public API compatibility, tests, artifact contracts, and readable code beat line count.
   - When a file naturally contains multiple responsibilities, split it into named modules rather than compressing it into dense code.
   - When a file is cohesive, reduce redundancy without forcing an artificial split.
   - Avoid code golf, cryptic names, giant one-liners, or clever abstractions that hide the Grid Dataset domain.
   - Before renaming or deleting a public function, class, constant, signature, output key, or artifact path, search call sites and preserve compatibility unless the simplification plan explicitly calls for that terminology migration.

## Non-Negotiable Contracts

Do not weaken or remove:

- Stable Grid ID determinism and uniqueness.
- Required fields for published Grid Dataset artifacts.
- Valid-coordinate requirements for Flood-Relevant Assets.
- Block Invariant Contract checks.
- Load Match completeness checks.
- ONM/RPOP export structural validity.
- SMART-DS-compatible interface fields that downstream consumers rely on.

## Staged Work Plan

## Progress Notes

- Stage 1 started with shared artifact IO simplification: generic CSV/parquet/debug wrappers were removed in favor of direct package calls while keeping Stable Grid ID and parquet artifact conventions.
- Stage 2 started with Load Match / profile language cleanup: DER/profile APIs now prefer `location_id` and `load_match` language, verbose schema names were shortened, duplicated import blocks were removed, and published `sandbox_id` artifact fields / existing filenames were preserved for compatibility.
- Current measured Stage 2 slice: `src/power/resilience/der.py`, `src/power/resilience/profiles.py`, and two Marshfield augmentation notebooks show net deletion (`338` insertions / `598` deletions) with compile, import, and notebook JSON checks passing.
- Switch/load-block slice: `resilience/switches.py` had duplicate import/docstring sections removed and narrow notebook-facing calls migrated to `location_id`; current resilience slice shows net deletion (`375` insertions / `683` deletions) with compile, import, notebook JSON, and `git diff --check` passing.
- Plotting slice: `plotting.py` had duplicate import/docstring sections removed, `_line_gdf` / `_line_gdf_dft` merged, one-use tick helpers inlined, repeated switch-segment loops merged, legend boilerplate shortened, repeated GeoJSON FeatureCollection writing consolidated, and shared domain/AABB merge output factored. Public plotting signatures were preserved; compile/import/API checks passed.
- Current tracked power/notebook cleanup slice shows net deletion (`916` insertions / `1374` deletions) with `src/power` compile, focused power imports, notebook JSON checks, and `git diff --check` passing.
- Current line-count pressure targets: `resilience/switches.py` (~2,000 lines), `plotting.py` (~1,536 lines), `exports/restoration.py` (~1,600 lines), and `exports/smart_ds_grid.py` (~1,600 lines). These should be shortened by deleting duplicated local machinery first; split only where the domain boundary is obvious and reduces total complexity.
- Next deletion target: move to export modules (`exports/restoration.py` and `exports/smart_ds_grid.py`) unless a clear plotting split emerges; plotting has had the obvious repeated helper machinery removed.

### Stage 1 - Shared Artifacts and Naming

Target: former `src/power/artifacts.py` pattern and direct call sites.

- Replace `sandbox_id` with `location_id`.
- Simplify Study Location config resolution; fail plainly when config is invalid.
- Remove generic `read_csv`, `write_debug_csv`, and `require_pyarrow`.
- Keep Stable Grid ID helpers, hashing, provenance, and any small parquet helper that preserves a real artifact convention.
- Update imports and call sites.
- Prefer net line deletion in every touched file; if a local helper is added, remove more code nearby.

Verification:

- Confirm no source or notebook-facing test imports `power.artifacts`.
- Run focused tests or smoke calls for Stable Grid ID generation.
- Search for remaining non-legacy `sandbox` language under `src/power`.

### Stage 2 - Resilience Artifact Naming

Target: `src/power/resilience/` and affected notebooks.

- Align names with project language, especially `Load Match`.
- Shorten schema/version/function names.
- Keep compatibility aliases only when needed for a staged notebook transition.
- Preserve DER inventory, load profiles, controllable switches, load blocks, and ONM-facing outputs.

Verification:

- Import `power.resilience`.
- Compile all `src/power/resilience/*.py`.
- Validate notebook JSON for touched `01_grid/02_augment_network` notebooks.

### Stage 3 - Export Modules

Target: `src/power/exports/`.

- Replace generic CSV/parquet wrappers with direct pandas/pyarrow use.
- Keep published artifact schemas.
- Split `smart_ds_grid.py` into Stage A1 Grid Dataset export and Stage A2 event/telemetry export only if the shortened file remains hard to read.
- Split `restoration.py` only along obvious artifact boundaries such as ONM settings, event bundle, load uncertainty, and run bundle.
- Merge duplicated import blocks and local mini-frameworks before considering any new file split.

Verification:

- Import `power.exports`.
- Compile export modules.
- Run any lightweight artifact writer smoke test that does not require full network regeneration.

### Stage 4 - Switches and Load Blocks

Target: `src/power/resilience/switches.py`.

- Keep SSAP and Block Invariant Contract logic explicit.
- Move diagnostic-only reporting to audit/review helpers.
- Split into domain files only if boundaries are clear, such as switch siting, load blocks, and topology helpers.

Verification:

- Import split modules.
- Run focused load-block validation tests or smoke checks where possible.
- Confirm public notebook imports still read like Infrastructure Steps.

### Stage 5 - Audit and Plotting

Target: `src/power/audit/` and `src/power/plotting.py`.

- Keep broad diagnostic context in audit modules.
- Keep plotting functions as review helpers, not hidden production builders.
- Remove duplicate CSV/geospatial parsing where geopandas or pandas can be used directly.

Verification:

- Import audit and plotting modules.
- Run notebook JSON checks for touched audit notebooks.

### Stage 6 - Notebook Imports and Readability

Target: `locations/*/01_grid/**/*.ipynb`.

- Replace verbose imports with shorter artifact names.
- Keep each notebook step readable as a scientific workflow.
- Avoid notebook-local helper blocks when a function belongs in `src/power`.

Verification:

- Notebook JSON parses.
- No long helper cells are introduced.
- Dry-run or smoke only where execution would be expensive.

### Stage 7 - Regression Sweep

- Compile all `src/power/**/*.py`.
- Run available focused tests.
- Search for banned or legacy terminology.
- Review line-count deltas for touched files and prefer follow-up deletion when the net change grows.
- Report net line-count reduction for the power cleanup and call out any additions that remain because they protect a published artifact contract.
- Check that generated artifact filenames and schemas remain compatible unless intentionally changed.

## File Split Candidates

Split only after a shortening pass proves the file remains too broad:

- `resilience/switches.py`
  - possible split: switch siting, load-block validation, topology helpers.
- `exports/smart_ds_grid.py`
  - possible split: Stage A1 Grid Dataset export, Stage A2 event/telemetry export.
- `exports/restoration.py`
  - possible split: ONM settings, load uncertainty, event windows, run bundle.
- `plotting.py`
  - possible split: grid extent/source plots, infrastructure-step review maps, audit figures.

## ADR Decision

No ADR is needed yet. This is a reversible cleanup plan that applies existing project language and package boundaries rather than introducing a hard new architectural commitment.
