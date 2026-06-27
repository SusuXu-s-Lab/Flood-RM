# Inland Event Reference Time on the Rainfall Peak; Inland Timing Descriptors as Response-Side Diagnostics

## Status

Accepted. **Operationalizes the timing side of ADR-0016** (inland: rainfall is the
single driver, discharge is the Wflow response). Builds on ADR-0009 (variable-length
physical scenarios anchored on an Event Reference Time) and ADR-0012 (coverage-box /
stream-boundary handoff). Does not change the copula `driver_vector` (still `"rainfall"`
inland) and does not touch the Marshfield coastal compound-lag path
(`build_events/compound_timing.py`).

## Context

Two gaps were found while wiring inland (Austin, Greensboro) precipitation timing to the
Marshfield reference behaviour.

**1. The rainfall member tables for Austin and Greensboro carried no peak timing.** The
Direct AORC SST collector (`collect_sources/aorc_sst.py`) already computes and writes
`rainfall_peak_time`, `rainfall_peak_mm_per_hour`, and `rainfall_peak_time_source` per
storm window (the true hourly peak inside the accumulation window, moving-footprint
aware). Marshfield's `rainfall_members.csv` has these columns; Austin's and Greensboro's
were stale tables from an older collector and lacked them. Marshfield's 02 notebook also
carries an "AORC Re-Collect" convenience cell (`refresh_aorc_sst_only`) that regenerates
only the AORC SST source tables; the inland 02 notebooks did not.

**2. The inland builder ignored peak timing even when present.**
`build_events/probability/inland_dependence.py::build_inland_catalog` realizes rainfall
via `attach_field_preserving_realization(..., time_column="storm_start")` with no observed
lags, then set `event_reference_time = rainfall_member_time` — i.e. the **storm-window
onset**. Because `wflow_runs.replay.resolve_event_window` centres the Wflow forcing window
on the Event Reference Time (`[ref − pre, ref + post]`, default 48 h / 72 h), anchoring on
onset placed the true storm peak up to the full accumulation window (≈72 h) *after* the
reference — pushing the peak against the window tail with little recession captured. The
collected peak timing never reached the catalog or the diagnostics.

Inland has no second *observed* driver to lag against (discharge is the Wflow response,
not an independent driver — ADR-0016), so the Marshfield Compound Driver Lag (peak-RF vs
peak-NTR, copula-conditioned) does not transfer. The useful inland timing information is
**response-side**: where the rainfall peaks within the storm, how long the catchment takes
to translate that into discharge, and how antecedent wetness and season modulate it.

## Decision

**1. Inland Event Reference Time is the true rainfall peak.** `build_inland_catalog` reads
`rainfall_peak_time` from the realized rainfall member and sets
`event_reference_time = rainfall_peak_time`. The rainfall member keeps
`rainfall_member_time = storm_start` (it is the AORC event-window `.nc` lookup key in
`replay._event_rainfall_source_nc`). The catalog records:

| field | value | role |
| --- | --- | --- |
| `rainfall_member_time` | `storm_start` (unchanged) | AORC event-window `.nc` lookup key |
| `event_reference_time` | `rainfall_peak_time` | centres the Wflow forcing window + SFINCS alignment |
| `rainfall_peak_offset_hours` | `peak − storm_start` | Event Timing Descriptor |
| `rainfall_start_offset_hours` | `−rainfall_peak_offset_hours` | so `replay._catalog_rainfall_start` reconstructs `storm_start = ERT + offset` |
| `rainfall_peak_time_source` | from member table, else `inferred_midpoint` (flagged) | provenance |

When a realized member lacks `rainfall_peak_time` (a stale table), the builder falls back
to the storm-window midpoint, flags `rainfall_peak_time_source`, and warns — mirroring the
legacy-midpoint guard in `compound_timing.py` — rather than silently anchoring on onset.

**2. Inland timing descriptors are response-side diagnostics, never copula axes.** A new
`build_events/inland_timing.py` derives, from collected data only:

- **Storm loading pattern** — normalized peak position `peak_offset / duration ∈ [0,1]`
  with a front / center / back-loaded label (Huff terciles; Huff 1967, ARR temporal
  patterns). A new diversity axis for the Resilience Stress/Training Set and a data-driven
  source of Scenario Timing Edge Case tags.
- **Observed catchment basin lag** — `observed USGS peak − rainfall_peak_time` at the
  Primary Reference Gage for the member's historical storm date, joined with antecedent
  SOILSAT and season. The *observed reference* for the Wflow Readiness peak-timing check
  and the observed side of the Soil-Moisture Modulation Diagnostic.
- **Timing seasonality** — peak month/hour distribution (convective vs frontal/tropical).

These are reported in `03_build_event_catalog` and feed downstream Wflow-stage diagnostics
(pluvial–fluvial peak superposition, simulated basin lag) once Wflow discharge exists; they
do not enter the design probability.

## Consequences

- The Wflow forcing window is centred on the physical storm peak instead of onset, fixing a
  latent up-to-72 h mis-centering. The change is transparent to the handoff: the replay
  plumbing already reads `event_reference_time` from the catalog row; only its value changes,
  plus additive descriptor columns. `rainfall_member_time` is unchanged, so AORC
  event-window lookup is unaffected.
- Re-running AORC SST collection regenerates the peak columns automatically — the collector's
  reuse guards (`_ranked_storms_are_sst_equivalent`, `_stats_checkpoint_is_current`) already
  require `rainfall_peak_time`, so a stale table forces recompute.
- Implemented across `design_events/build_events/probability/inland_dependence.py` (ERT +
  descriptors), new `design_events/build_events/inland_timing.py` (descriptors + observed
  diagnostics), `design_events/plotting.py` (timing figures), and the Austin/Greensboro 02
  (ported AORC Re-Collect cell) and 03 (descriptor columns + diagnostics) notebooks.
- The coastal compound-lag path is untouched; inland remains rainfall-single-driver
  (ADR-0011 / ADR-0016).
