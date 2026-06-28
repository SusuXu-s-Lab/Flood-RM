# Flood-RM Agent Instructions

Flood-RM is a scientific Python repository for flood-resilient microgrid workflows.
The clean notebooks under `locations/<name>/` are the intended user-facing workflows.
Most backend behavior lives under `src/`, including `power`, `design_events`,
`sfincs_runs`, `wflow_runs`, `fiat_runs`, `study_location.py`, and
`generated_artifact.py`.

These instructions apply when auditing, simplifying, refactoring, or documenting this
repository.

## Preserve Scientific Behavior

- Preserve scientific behavior before improving style.
- Do not alter formulas, event-selection semantics, flood-model setup semantics, grid
  topology logic, switch-allocation logic, or file-format semantics unless the change is
  explicitly justified and validated.
- Never silently replace domain logic with a library call unless equivalence is
  demonstrated or the old logic is clearly plumbing or reimplementation.
- Prefer smaller, flatter, traceable scientific Python over added architecture.

## Treat Notebooks As Public APIs

- The notebooks under `locations/<name>/` are the front-end contract.
- Backend refactors must preserve notebook imports, call order, outputs, artifact paths,
  and expected side effects unless a migration note is written.
- Every backend function used by notebooks should be easy to trace from the notebook to
  the implementation.

## Prefer Standard Scientific Python

- Prefer `pandas`, `numpy`, `scipy`, `geopandas`, `shapely`, `matplotlib`, `xarray`,
  `pathlib`, `dataclasses`, `typing`, `logging`, and standard library utilities over
  hand-rolled replacements.
- Remove or simplify redundant wrappers, excessive defensive checks, duplicated
  conversions, repeated path handling, custom plotting abstractions, and manual
  dataframe/list manipulation when standard functions are clearer.
- Helper-module extraction is a temporary safety move, not the desired end state. After
  characterization, remove, inline, or replace thin wrappers that do not encode a named
  Flood-RM concept.
- Do not add heavy new dependencies unless there is a strong reason.

## Refactor Incrementally

- Do not perform broad rewrites.
- First create maps, inventories, and tests.
- Then split oversized files by coherent responsibility.
- Then remove dead or redundant code.
- Then update imports and docs.
- Keep changes reviewable.

## Classify Code Before Editing

Use these categories in notes, plans, reports, and review comments:

- `CORE_SCIENCE`: hydrology, flood-driver generation, extreme-value treatment,
  Wflow/SFINCS/SnapWave coupling, grid construction, resilience metrics, switch
  allocation, OpenDSS/SMART-DS semantics.
- `NOTEBOOK_API`: functions/classes directly called by notebooks.
- `PLUMBING_IO`: path handling, config loading, serialization, manifests, file
  readers/writers, external command setup.
- `VALIDATION_QA`: checks, diagnostics, audits, smoke tests.
- `PLOTTING_REPORTING`: visualization and reporting helpers.
- `REDUNDANT_OR_REPLACEABLE`: duplicated utilities, hand-rolled
  pandas/numpy/scipy/matplotlib behavior, excessive compatibility shims, unnecessary
  wrappers.
- `STALE_OR_DANGEROUS`: unused code, contradictory logic, unreachable branches, stale
  aliases, conflicting implementations, misleading names.

## Validation Expectations

Always run or explain why you cannot run:

```bash
uv run python -m compileall src
uv run python -m pytest
```

- If tests are absent or too heavy, create lightweight tests for pure functions and
  smoke-test importability.
- Do not download large external datasets or require credentials by default.
- Avoid running long hydrodynamic solvers or network-heavy workflows unless explicitly
  asked.

## Documentation Expectations

- Maintain `docs/codebase_map.md` as the human-readable map.
- Maintain `docs/notebook_backend_trace.md` as the notebook-to-backend trace.
- Maintain `docs/refactor_roadmap.md` as the prioritized cleanup plan.
- Maintain `docs/reduction_candidates.md` for dead, duplicate, or replaceable code
  candidates.
- Follow `docs/plumbing_reduction_directive.md` before expanding or preserving any
  `src/` path, config, manifest, parser, hash, artifact, or compatibility-wrapper
  plumbing. This applies across `power`, `design_events`, `sfincs_runs`, `wflow_runs`,
  `fiat_runs`, `study_location.py`, and scripts.
- Use neutral language. Do not claim code is "AI-generated." Prefer "over-engineered,"
  "duplicative," "hand-rolled," "replaceable by standard library/scientific stack," or
  "unclear ownership."

## Agent Coordination

- Multiple agents may work in parallel only if file ownership is explicit.
- No two agents should edit the same source files.
- Domain agents should write findings to `docs/agent_reports/<domain>.md`.
- Integration should happen in a separate final pass.

## Source Code Rule

Do not edit source code during an audit or planning pass unless explicitly asked. When
implementation begins, keep changes scoped, validated, and traceable back to notebook
behavior and documented findings.
