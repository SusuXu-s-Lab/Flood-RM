# Reduction Candidates

Detailed evidence lives in `docs/audit/reduction_candidates.json`. These are candidates
for review, not automatic edit instructions.

## Severity Key

- `BLOCKER`: do not edit or delete until the missing evidence is produced.
- `HIGH`: split or test first; behavior changes are risky.
- `MEDIUM`: likely reducible after call-site and artifact checks.
- `LOW`: safe to tidy only after preserving public imports and figure/data outputs.

## Global Cautions

- `REDUNDANT_OR_REPLACEABLE` does not mean "dead." It means "review for possible
  simplification."
- No candidate should be deleted from notebook call-site evidence alone. Check notebooks,
  scripts, tests, package facades, dynamic/string imports, and artifact-only workflows.
- Artifact schema modules are not removable plumbing unless the schema, manifest, ID, and
  side-effect contract is preserved.
- Helper-module splits are not automatically resolved architecture. After tests
  characterize behavior, thin wrappers should be inlined, deleted, or replaced with
  direct scientific-library calls when they do not encode named Flood-RM concepts.

## Strong Candidates

1. `src/wflow_runs/build_plan.py` (`HIGH`)
   - Evidence: 3513 lines, 80+ top-level functions, notebook-facing, 19 dataframe-loop
     hits, 52 path/config/json hits, 29 defensive compatibility hits.
   - Classification: `CORE_SCIENCE`, `NOTEBOOK_API`, `REDUNDANT_OR_REPLACEABLE`.
   - Reduce by splitting Wflow domain planning, source strategy, staticmap QA,
     reservoirs, HydroMT build steps, and repair helpers.

2. `src/sfincs_runs/build_base/region_notebook.py` (`HIGH`)
   - Evidence: 2263 lines, public call sites from Austin, Greensboro, and Marshfield
     region setup notebooks, 40 path/config/json hits, 56 plotting hits.
   - Classification: `CORE_SCIENCE`, `NOTEBOOK_API`, `REDUNDANT_OR_REPLACEABLE`.
   - Reduce by keeping a stable notebook facade and moving coastal/inland collection,
     coverage preflight, static input QA, and plotting into separate modules.

3. `src/power/resilience/switches.py` (`BLOCKER` for simplification, `HIGH` for
   facade-preserving split)
   - Evidence: 1998 lines, core SSAP and Switch-Bounded Load Block logic mixed with
     artifact row creation, 21 dataframe-loop hits.
   - Classification: `CORE_SCIENCE`, `REDUNDANT_OR_REPLACEABLE`.
   - Reduce only after characterization tests around SSAP, candidate selection,
     Fuse Proxy handling, and Block Invariant Contract checks.

4. `src/design_events/collect_sources/workflow.py` (`HIGH`)
   - Evidence: 2096 lines, called by three source-collection notebooks, 43 plotting
     hits, source-readiness and streamgage review mixed with notebook plots.
   - Classification: `NOTEBOOK_API`, `PLUMBING_IO`, `REDUNDANT_OR_REPLACEABLE`.
   - Reduce by separating runtime, collection plan, readiness tables, streamgage review,
     and plotting.

5. `src/study_location.py`, `src/aoi.py`, and `src/paths.py` (`MEDIUM`, shared path cleanup started;
   still a reduction target)
   - Evidence: the root interface was split from 1584 lines to a 214-line facade while
     preserving the live `study_location` imports.
   - Resolved: path resolution and GeoJSON IO now live in a bare-minimum `src/paths.py`;
     the old `src/_study_location_paths.py` file was deleted. The
     `_study_location_templates.py` file was also deleted after call-site review:
     `resolve_study_location` is real runtime identity logic and now lives in the
     facade, while starter-template/listing/coastal-wave convenience helpers had no
     source or notebook callers. Study Area geometry remains in `src/aoi.py`.
     HydroMT recipe inheritance was replaced by explicit location YAML, and the
     `_study_location_recipes.py` helper was deleted.
   - Grill-review update: the split is not the final architecture. Repeated path/runtime
     behavior across `study_location.py`, `sfincs_runs.config`, `design_events.runtime`,
     Wflow, FIAT, and grid-notebook runtime setup should be consolidated only where semantics
     match, with fewer path conventions as the target. The former `design_events.utils`
     catchall was deleted; do not recreate it under a new name.
   - Still candidate: the remaining facade owns YAML include loading and recursive
     config merging. Keep it only while `study_location.py` remains the notebook-facing
     Location Configuration seam.
   - Classification: `PLUMBING_IO`, `NOTEBOOK_API`, `REDUNDANT_OR_REPLACEABLE` for the
     remaining facade plumbing.

## Plotting And Reporting Candidates

- `src/design_events/plotting.py` (`MEDIUM`): 2240 lines, 264 Matplotlib-related hits. Split by
  return-period, pairing, catalog, streamflow, and copula diagnostics.
- `src/sfincs_runs/diagnostics.py` (`MEDIUM`): 2481 lines, 126 plotting hits plus event outcome
  tables and probability diagnostics. Split data preparation from plots/animations.
- `src/power/plotting.py` (`LOW` to `MEDIUM`): 1536 lines, notebook-facing grid figures. Split by Baseline
  Network, Controllable Switches, load blocks, and export views.
- `src/wflow_runs/visualize.py` (`MEDIUM`): 1095 lines, 69 plotting hits and geospatial plot
  helpers. Split event plots, basemap plots, animation, and geometry helpers.

These are lower scientific risk than event selection or topology solvers if plot data
frames and figure outputs are characterized first.

## Source Collection And External IO Candidates

- `src/design_events/collect_sources/national_hydrography.py` (`HIGH`): 1806 lines, 59
  path/config/json hits, network access, NHD/WBD/NLDI/hydrography/SSURGO concerns.
- `src/design_events/collect_sources/aorc_sst.py` (`HIGH`): 1327 lines, notebook-facing AORC SST
  collection with path and manifest handling.
- `src/design_events/collect_sources/usgs_streamgages.py` (`HIGH`): 1082 lines, streamgage
  discovery and record collection.
- `scripts/get_structures.py` (`MEDIUM`): public CLI plus provider registry, live ArcGIS access,
  GeoJSON clipping, derived SFINCS structures, and manifest writing.
- `scripts/onm_export.py` (`MEDIUM`): public CLI plus staging, bundle assembly, artifact copying,
  event windows, and Julia smoke assets.

Reduction path: make no live network calls in tests; first isolate pure path/manifest
formatting and dataframe/geospatial transformations.

## Export And Artifact Candidates

- Former `src/power/artifacts.py` (`RESOLVED`): deleted after migrating source callers
  to `src/paths.py`, `study_location.py`, or module-local helpers. Treat it as the first
  completed example of the repo-wide plumbing deletion test, not as a power-specific
  exception.
- Former `src/power/notebook.py` (`RESOLVED`): deleted after Marshfield grid notebooks
  were migrated to direct `sfincs_runs.config.load_runtime` and `build_grid_paths`
  imports. Keep watching other `*.notebook.py` files with the same deletion test before
  preserving them as public interfaces.
- Former `src/fiat_runs/notebook.py` (`RESOLVED`): deleted after the Marshfield FIAT
  notebook migrated to `fiat_runs.load_notebook_runtime` and kept workflow verbs in
  `fiat_runs.risk_workflow`. This preserves the FIAT risk notebook contract without a
  broad notebook re-export facade.
- `src/power/exports/restoration.py` (`HIGH`): 1594 lines, ONM settings, DSS text manipulation,
  event windows, asset-state events, run bundle materialization, and Julia smoke commands.
- `src/power/exports/smart_ds_grid.py` (`HIGH`): 1584 lines, SMART-DS assets/control units,
  synthetic event samples, SFINCS-depth samples, telemetry, schemas, and manifests.
- `src/fiat_runs/risk_workflow.py`, `src/fiat_runs/build_model.py`,
  `src/fiat_runs/diagnostics.py`: repeated path/config and artifact workflow logic.
- `src/generated_artifact.py`: resolved. The shallow generated-YAML wrapper was deleted;
  writer modules now call `yaml.safe_dump` directly and keep generated-file notices local
  where they are part of the artifact contract.

## Duplicate Function Names

The inventory found repeated names. These are not automatically duplicative, but they
are good review starting points:

- `load_runtime`: appears as separate notebook runtime helpers by domain. Keep separate
  public imports unless common behavior can move behind them without changing notebooks.
- `load_config`: appears in multiple domains. Consolidate only where schemas match.
- `write_manifest`: appears across source collection, Wflow, SFINCS, and scripts. Prefer
  small shared JSON/path helpers only if manifest semantics stay domain-owned.
- `parse_args` and `main`: normal CLI duplication; usually not worth abstracting.

Resolved SFINCS helper candidates:

- Former `src/sfincs_runs/notebook.py`: deleted after Marshfield flood notebooks moved
  to `sfincs_runs.config.load_sfincs_runtime`.
- Former `src/sfincs_runs/scenarios/io.py`: deleted; direct JSON writes replaced the
  generic wrapper, `parse_sfincs_inp` moved to `sfincs_runs.config`, and event-only
  SFINCS file writers are private to `scenarios.event_forcing`.
- Former `src/sfincs_runs/build_base/region_setup.py`: deleted; `RegionSetup` and
  `build_region_setup` moved to `build_base.static_intake`.
- Former `src/sfincs_runs/single_use_case/`: deleted; example-event selection is private
  to `scenarios.event_forcing.build_event`.

Do not mark SFINCS forcing, handoff placement, HydroMT recipe execution, or SFINCS
input-file semantics as removable plumbing without focused characterization tests.

## Src-Wide Plumbing Candidates

The artifact/path cleanup clarified that the former `power.artifacts.py` was one
example of a repo-wide pattern. Future passes should apply the same deletion test to:

- repeated `_location_path`, `_repo_path`, `_repo_root`, `_resolve_*_path`, and
  module-level `default_*_root` helpers;
- local `load_config`, `resolve_path`, and non-notebook-facing runtime pass-throughs;
- generic source-artifact, manifest, optional hash, and file IO wrappers;
- scalar parser, presence, slug, and token helpers that do not encode a named artifact
  contract.

Do not centralize these into a large framework. Prefer `src/paths.py` for bare
repo/location path basics, direct scientific-library calls, and domain-owned artifact
schemas/side effects.

## Stale Or Dangerous Candidates To Confirm

- `BLOCKER`: Notebook-local plotting/inspection notebooks with no `src` imports may duplicate
  backend plotting: `locations/*/01_grid/sds_plot.ipynb`,
  `locations/marshfield/01_grid/psds_plot.ipynb`, and some `03_ops/overlay.ipynb`
  notebooks. Do not delete or archive until their artifact reads/writes are traced.
- `HIGH`: `locations/austin/02_flood/05c_ship_calibrated.ipynb` and
  `locations/greensboro/02_flood/05c_ship_calibrated.ipynb` have no backend imports but
  appear to promote calibration artifacts. Treat as artifact-facing workflow contracts.
- `MEDIUM`: `scripts/onm_export.py` overlaps conceptually with `power.exports.restoration`; decide
  whether the script is only a thin CLI adapter or owns bundle policy.
- `MEDIUM`: Direct notebook imports from deep modules, such as
  `wflow_runs.build_plan.validate_staticmaps`, should either become documented public
  interfaces or move behind a facade with a migration note.
- `MEDIUM`: Broad `except Exception` and compatibility checks found in large modules should be
  reviewed for whether they protect workflow ergonomics or hide broken science.
- `MEDIUM`: competing static-data paths in `region_notebook.py` and `static_intake.py`
  need caller and artifact evidence before either is simplified.
- `MEDIUM`: Wflow domain-planning variants in `build_plan.py` may represent configured
  location modes, not stale alternatives.

## Do Not Reduce Yet

- Event-selection semantics in `src/design_events/build_events/selection.py`.
- Coastal and inland probability/fitting semantics in `src/design_events/build_events/*`
  and `src/design_events/fit_history/*`.
- SFINCS/Wflow handoff placement and flood-model setup semantics.
- SSAP, grid topology, switch allocation, and Block Invariant Contract logic.

For these, reduction starts with tests and documentation, not code movement.
