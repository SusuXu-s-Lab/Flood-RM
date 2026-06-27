# Variable-Length Physical Scenarios

## Status

Accepted.

## Context

The first wave-coupled Marshfield single-use run exposed a modeling shortcut:
the selected coastal hydrograph covered only the compact peak window while
rainfall members were fixed to a 72-hour SST duration. That made the SFINCS
animation short, but the deeper issue was that fixed windows erase important
physical diversity: slow-build coastal events, fast flashy rainfall, long
rainfall tails, driver lead/lag, and recovery-relevant flood duration.

Dynamic microgrid and resilience studies need event duration and timing to
remain meaningful scenario attributes. Fixed-size ML tensors or convenient
notebook animations must not define the physical duration of the hydrodynamic
truth run.

## Decision

Use variable-length physical scenarios in the Event Catalog and SFINCS staging
workflow.

Each Event Catalog row should record an **Event Reference Time**, driver-native
**Event Timing Descriptors**, and any **Scenario Timing Edge Case** labels. For
Marshfield and other coastal wave-coupled scenarios, the default Event
Reference Time is the coastal/wave analog peak. For inland or fluvial
scenarios, it is the streamflow peak. Rainfall, soil moisture, streamflow, and
other drivers keep their own offsets relative to that reference instead of
being forced to peak at the same hour.

SFINCS staging should compute the concrete **Forcing Support Window** from the
union of paired driver windows plus configured spin-up and drain-down padding,
bounded by minimum and maximum run lengths. The Event Catalog stores physical
timing and provenance; staging writes exact `tstart`, `tstop`, and forcing-file
times to the run manifest.

For rainfall, prefer one-pass multi-duration SST characterization of physical
storm episodes over separate fixed-duration catalogs as the canonical event
model. A rainfall member may carry descriptors such as max 6-hour, 24-hour,
72-hour, total-event depth, native duration, and front-loaded/back-loaded shape.

Variable-length truth data remains canonical. Fixed-size downstream products
should be derived as **Standardized Training Views** using padding masks,
peak-centered slices, recovery-tail slices, or resampled summaries.

## Considered Options

- Keep compact coastal windows such as the current 12-hour peak-centered
  hydrograph. Rejected because it removes slow-build and long-tail behavior and
  can truncate physically relevant rainfall, wave, and flood-response timing.
- Force all SFINCS scenarios to 72 hours. Rejected because the 72-hour value is
  a rainfall-member convention, not a universal compound-event duration.
- Build separate canonical rainfall catalogs for each duration. Rejected as the
  primary model because it duplicates physical storms and wastes AORC I/O;
  multi-duration characterization from one source pass is clearer and more
  efficient.
- Force all drivers to share one peak time. Rejected because lead/lag structure
  is itself a meaningful compound-flood scenario attribute.

## Consequences

- Event Catalog schema and diagnostics need explicit timing descriptors and
  duration/lead-lag summaries.
- SFINCS staging needs a window planner instead of deriving `tstop` from the
  length of one coastal boundary series.
- The first implementation slice should introduce the SFINCS staging/window
  planner and manifest timing outputs before refactoring AORC SST collection;
  multi-duration rainfall collection can then populate richer descriptors into
  the same staging path.
- The planner should be a small pure timing module under
  `src/sfincs_runs/scenarios/`, separate from HydroMT/SFINCS file staging, so
  driver-window logic can be tested without opening large forcing datasets.
- Timing validation should be strict for production truth-set staging, but
  notebooks and legacy tests may use explicit fallback inference from existing
  catalog fields and hydrograph lengths. Any fallback must be labeled in the
  run manifest so compact legacy windows cannot silently become production
  truth runs.
- Resilience Stress/Training Set selection should first cover driver diversity
  and timing edge cases, then become consequence-aware after evaluated SFINCS
  outputs exist.
- ML and dynamic-microgrid pipelines must handle variable-length truth through
  derived standardized views rather than demanding fixed-duration physical
  scenarios.
