# Copula-Joint Compound-Driver Dependence Model

## Status

Accepted. Realizes the vine-copula upgrade that ADR-0008 deferred ("Future upgrades
can add a vine copula ... without renaming the current Event Catalog"); ADR-0008's
staged framework, return-period language, and provenance rules still hold.

## Context

The Event Catalog combined compound drivers with heuristic Forcing Pairing Policies
(seasonal-window rainfall permutation, antecedent-lag soil moisture, same-historical
coastal analog). Those are auditable realism constraints but not a recognized
joint-probability method, and the notebooks themselves stated they were "not a fitted
joint AEP model." For peer-reviewed use this is the one indefensible step: the closest
published, SFINCS-coupled blueprint (Maduwantha et al. 2026, building on Jane et al.
2020 and Maduwantha et al. 2024 — the lineage already in `design_event_methodology.md`)
fits a copula over the peak event drivers. Antecedent soil moisture is not a symmetric
event-peak driver in SFINCS/Wflow staging; it is an infiltration/runoff initial
condition sampled as an event conditioning attribute.

Two further constraints shaped the design. (1) Spatio-temporal rainfall structure must
survive: collapsing rainfall to a scalar was the failure mode of an earlier attempt.
(2) The Resilience Stress/Training Set needs ~195 rare+extreme events out of 500, but
the AND joint tail is far sparser than any marginal tail — a proportional copula sample
yields only ~27, so naive sampling silently starves the stress set.

## Decision

Adopt a **Copula-Joint Dependence Model** as the production `copula_joint` Forcing
Pairing Policy, retaining the heuristic policies as named sensitivity baselines.

- **Driver Dependence Vector**, config-declared per flood setting
  (`event_catalog.dependence.driver_vector`). Coastal = {non-tidal residual water
  level, rainfall}. Inland Wflow-coupled treats rainfall as the event driver, with
  **discharge a Wflow-derived response, not a copula dimension** — so rainfall is not
  double-counted and the sampled RP stays consistent with the routed flow. Streamflow
  POT becomes a Wflow-readiness validation anchor inland. Antecedent moisture is sampled
  outside the copula as an **Antecedent Moisture State** conditioned on storm context.
- **Vine copula** (`pyvinecopulib`), fit semiparametrically (rank pseudo-observations;
  AIC family selection incl. tail-dependence families) on the **Two-Sided Conditional
  POT Co-occurrence Sample** assembled from real aligned driver records. For the
  coastal rainfall/NTR sample, the production pairing window follows the direct
  Maduwantha et al. (2026) boundary-condition framework: condition on one driver and
  select the partner maximum within a 3 d window. Shorter windows may be run as
  sensitivity diagnostics, but are not a production default without an explicit
  citation or location-specific validation record.
- **AND joint-exceedance** labeling (`T = 1/(rate·P(all exceed))`, estimated by Monte
  Carlo from `simulate()`), with marginal Driver Return Periods kept as anchors and the
  rigorous `probability_weight` taken from the fitted joint density; "most-likely event"
  selection along AND isolines for benchmark design events (Salvadori & De Michele 2013).
- **Joint-tail-enriched sampling** with importance weights (Beck & Zuev 2015): the
  Probability Catalog is enriched by count so the rare/extreme bands fill the stress
  budget, while `probability_weight` reconstructs the true (mild-dominated) mass. A
  build-time budget gate fails loudly if a band cannot meet the 500-event requirement.
- **Field-Preserving Realization**: each sampled scalar index resolves to an observed
  AORC SST field / coastal record analog scaled by `K = target/observed` and
  time-lagged (Kim et al. 2023; Maduwantha et al. 2026). For coastal water level, the
  probability index is the non-tidal residual while the tide is preserved unscaled in
  the realized boundary record; SFINCS staging multiplies the `netampr` rainfall field
  by its scale factor. RF/NTR peak lag is assigned by Conditional Empirical Lag
  Analogue Sampling: weighted kNN over storm type, peak NTR, peak RF, and season, then
  inheritance of the observed analogue lag. Marginals and SST fields are unchanged.

## Consequences

- The compound step is now citeable end-to-end (Jane 2020; Maduwantha 2024/2026;
  Aas 2009/Dißmann 2013 vines; Salvadori 2016; Moftakhari 2019; Kim 2023; Beck & Zuev
  2015), with `copula_joint` selectable per location and the heuristic path retained for
  sensitivity comparison.
- Spatio-temporal rainfall is preserved by construction: the copula touches only scalar
  indices; the physical field is the scaled observed analog.
- Coastal tide is preserved by construction: the copula scales NTR, not the astronomical
  tide component.
- AND AEP carries Monte-Carlo sampling uncertainty worst in the deep tail; the catalog
  reports per-band hit counts and distinct support, and low-support tail bands are
  flagged rather than treated as precise.
- Antecedent moisture is attached after the peak-driver sample. Coastal Study Locations
  use the NWM soil-moisture artifact for the first implementation, sampled by month
  until Storm Type artifacts exist; `storm_type` is reserved as an Event Catalog column.
- The inland reframe means inland catalog rows no longer carry an independent streamflow
  design member; discharge is produced by Wflow from the realized rainfall + antecedent
  state, and streamflow frequency is a validation anchor.
- Implemented in `design_events.build_events` (`dependence`, `joint_exceedance`,
  `realization`, `joint_catalog`) and `design_events.fit_history`
  (`paired_observations`, `driver_records`); the SFINCS rainfall staging hook applies the
  scale factor. Remaining work: wire `build_joint_design_catalog` into the production
  notebook build behind `event_catalog.dependence.method`, and a soil-moisture member
  library for its realization (rainfall already has one).
