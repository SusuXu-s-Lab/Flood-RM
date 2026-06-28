# Design Events Report

Task: 4b design-events/source-collection reduction pass.

## Scope

Owned files touched:

- `src/design_events/collect_sources/workflow.py`
- `src/design_events/collect_sources/__init__.py`
- `src/design_events/build_events/catalog.py`
- `tests/design_events/test_collect_sources_prerequisites.py`
- `tests/design_events/build_events/workflow_test.py`

No notebooks, shared docs, `src/power/**`, `src/sfincs_runs/**`, `src/wflow_runs/**`, or
`src/fiat_runs/**` were edited.

Guarded concepts touched:

- `Source Artifact`: read/reuse checks remain in the source-collection workflow; manifest
  schema and source adapter outputs were not changed.
- `Event Catalog`: only a local variable shadowing defect in `build_event_catalog` was
  fixed; catalog columns, pairing, required forcing, and audit semantics were not changed.

## Evidence And Classifications

- `src/design_events/collect_sources/workflow.py`
  - Evidence: `rg` found no source/test/notebook call sites for private
    `_default_collect_all_funcs`; it duplicated `_default_run_collect_funcs`.
  - Classification: `REDUNDANT_OR_REPLACEABLE`, `PLUMBING_IO`.
  - Change: deleted `_default_collect_all_funcs` and made `collect_all_sources` reuse
    `_default_run_collect_funcs`.

- `src/design_events/collect_sources/workflow.py`
  - Evidence: `rg "\bplan = plan\("` found `collect_all_sources` and
    `refresh_wflow_hydrography_basemap` assigning to the imported `plan` function name.
  - Classification: `STALE_OR_DANGEROUS`, `NOTEBOOK_API`.
  - Change: renamed locals to `collection_plan`; preserved public function names and call
    signatures.

- `src/design_events/build_events/catalog.py`
  - Evidence: `rg "\bplan = plan\("` found the same shadowing defect in
    `build_event_catalog`.
  - Classification: `STALE_OR_DANGEROUS`, `CORE_SCIENCE`, `NOTEBOOK_API`.
  - Change: renamed the local to `catalog_plan`; did not alter Event Catalog schema,
    forcing attachment, wave analog policy, or audit behavior.

- `src/design_events/collect_sources/__init__.py`
  - Evidence: `collect_warmup` appeared twice in `__all__`; notebooks import the symbol
    by name, not by duplicate export position.
  - Classification: `REDUNDANT_OR_REPLACEABLE`, `NOTEBOOK_API`.
  - Change: removed the duplicate `__all__` entry only.

- `tests/design_events/**`
  - Classification: `VALIDATION_QA`.
  - Change: added focused no-network tests for `collect_all_sources`,
    `refresh_wflow_hydrography_basemap`, and `build_event_catalog`.

## Preserved Interfaces

- `design_events.collect_sources.workflow.plan`
- `SourceCollectionPlan`, `SourceCollectionStep`
- `prepare`, `run_collect`, `collect_all_sources`
- `refresh_wflow_hydrography_basemap`
- `design_events.collect_sources.collect_warmup`
- `design_events.build_events.catalog.build_event_catalog`

## Line Counts

Measured before editing:

- `src/design_events/collect_sources/workflow.py`: 2355
- `src/design_events/collect_sources/__init__.py`: 23
- `src/design_events/build_events/catalog.py`: 542
- `tests/design_events/test_collect_sources_prerequisites.py`: 81
- `tests/design_events/build_events/workflow_test.py`: not present in `HEAD` / untracked
  test tree

Measured after editing:

- `src/design_events/collect_sources/workflow.py`: 2323, net -32
- `src/design_events/collect_sources/__init__.py`: 22, net -1
- `src/design_events/build_events/catalog.py`: 542, net 0
- `tests/design_events/test_collect_sources_prerequisites.py`: 165, net +84
- `tests/design_events/build_events/workflow_test.py`: 65

Source net: -33 lines. Tests added: +149 lines in the current untracked test tree.

## Needs Evidence

- Repeated source-artifact manifest helpers in `aorc_sst.py`, `nwm.py`,
  `usgs_streamgages.py`, `lcra_hydromet.py`, `stream_geo_nldi.py`, `cora.py`, and
  `era5_waves/__init__.py` remain candidates. They are source-adapter-local artifact
  contracts and should not be centralized without matching schema/side-effect evidence.
- `collect_sources/workflow.py` still mixes Source Collection Plan, readiness tables,
  streamgage review, and plotting. Reducing it further needs notebook call-order and
  artifact-path characterization for `02_collect_sources.ipynb`.
- `build_events/selection.py`, `build_events/coastal.py`, probability modules, and
  `plotting.py` remain high-value candidates but were not changed because they touch
  tail sampling, event-member construction, probability weighting, timing, pairing, or
  diagnostic plotting semantics.

## Validation

- `uv run python -m pytest tests/design_events/test_collect_sources_prerequisites.py tests/design_events/build_events/workflow_test.py`
  - Passed: 6 passed in 3.02s.
- `uv run python -m compileall src`
  - Initial sandboxed run failed because uv could not access `~/.cache/uv`.
  - Rerun with escalation passed.
- `uv run python -m pytest tests/design_events`
  - Passed: 120 passed, 1 warning in 131.66s.
- `uv run python -m pytest`
  - Failed: 30 failed, 779 passed, 8 warnings in 192.53s.
  - Failure themes were outside this design-events pass: missing/changed cluster SLURM
    helpers, cross-domain coverage/domain convention expectations, stale notebook
    execution counts/content expectations, grid source-input expectations, one Wflow
    dynamic-handoff worklist expectation, and one Wflow NetCDF overwrite permission
    failure.

## Cross-Domain TODOs

- Integration lane: reconcile full-suite baseline failures before treating `uv run python
  -m pytest` as a green gate.
- Task 4c/4d: coordinate any future source-collection changes that affect Wflow handoff
  catalogs, streamflow member readiness, or SFINCS forcing expectations.
- Integration lane: decide whether untracked `tests/design_events/**` should be added to
  the repository baseline before future workers rely on `git show HEAD` line counts.
