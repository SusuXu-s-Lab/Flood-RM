# AORC SST implementation audit — does the code represent the SST mechanism?

Purpose: map the **canonical stochastic storm transposition (SST)** procedure
onto the Flood-RM AORC SST collector so the mechanism can be trusted and
diagnosed, and record where the code simplifies, diverges from, or omits a
documented step. This audits `src/design_events/collect_sources/aorc_sst.py`
(catalog + transposition) and the rainfall use in
`src/design_events/build_events/probability/` + `03_build_event_catalog.ipynb`
(frequency), against the SST literature.

## Canonical SST (RainyDay / Wright; USACE; Foufoula-Georgiou)

1. **Storm catalog** — scan a gridded precip record with a moving window the
   size/shape of the watershed; keep the largest storms by watershed-accumulated
   depth for the target duration.
2. **Transposition domain + homogeneity** — a region over which storms are
   meteorologically *exchangeable* (equal likelihood of occurrence in space).
   The homogeneity assumption must be checked (seasonality, depth distribution,
   centroid spread), not assumed.
3. **Occurrence rate** — storms/year above the catalog threshold, modelled as a
   Poisson rate `λ`.
4. **Stochastic resampling (Monte Carlo)** — generate many synthetic years; in
   each, draw `k ~ Poisson(λ)` storms, sample storms with replacement, **transpose
   each to a random location** in the domain (uniform or kernel-weighted), overlay
   on the watershed, and record the watershed-accumulated annual maximum.
5. **Optional corrections** — non-uniform transposition-probability kernel,
   rotation, intensity/elevation rescaling.
6. **Frequency** — derive intensity-duration-frequency / return periods
   empirically from the synthetic annual-maxima distribution.

Sources: Wright et al. RainyDay; "Six decades of rainfall and flood frequency
analysis using SST" (J. Hydrology 2020); USACE HEC-HMS SST guidance; NOAA Atlas
14. See also `marshfield_sst_transposition_region.md`.

## What the code actually does

| Step | Code | Status |
|---|---|---|
| 1 catalog | `_spatial_stats` (rolling `duration_hours` sum, moving footprint via `_moving_footprint_plan`) → `_decluster_top_events` (now threshold-driven POT) | **Represented** |
| 2 domain | `transposition_region.geometry_file` bounds candidate footprint centres (`region_geom.covers(shifted)`) | **Represented** (domain), but see G3 |
| 2 homogeneity check | `run_aorc_homogeneity_diagnostic` | Represented as a review diagnostic — see G3 |
| 3 occurrence rate | `event_rate_per_year = n_declustered / record_years` in `03` | Represented, but on maximized depths (G4) |
| 4 Monte Carlo resampling | — | **NOT IMPLEMENTED** — see G1 |
| 4 transposition of the field | `_compute_selected_event_windows` → `_apply_field_transposition` | **Represented** for selected members — see G2 |
| 5 kernel / rotation / intensity | — | Omitted (acceptable simplification, G5) |
| 6 frequency | POT marginal fit on member depths + rank RP (`build_inland_catalog`, `design_catalog.py`) | Represented, but not SST Monte Carlo IDF (G1/G4) |

## Trust / diagnose gaps

**G1 (high) — transposition is a deterministic per-storm *maximization*, not random Monte Carlo.**
`_spatial_stats` evaluates the study footprint at every valid centre in the
domain and keeps the **maximum** footprint-mean (`best = argmax(means)`). This is
a maximizing-transposition envelope, not the frequency-preserving random
resampling of canonical SST. The rainfall marginal is then fitted on these
maximized depths, so the resulting return periods are **not** RainyDay-style SST
frequencies — they are biased toward the most severe transposable placement of
each storm. This is defensible only as an intentional conservative envelope and
must be labelled as such, not presented as an SST frequency estimate.

**G2 (resolved) — selected member NetCDFs now realize the spatial transposition.**
`_spatial_stats` records both the historical moving-footprint center that won the
scan and the actual Study Area footprint center. `_ensure_transposition_targets`
computes the offset between those footprint centers; `_compute_selected_event_windows`
then writes event-window NetCDFs with coordinates shifted by that offset and
records `aorc_sst_field_transposition`, `transposition_offset_*`, historical
centroid, transposed centroid, and footprint-center metadata in the artifact.
Downstream SFINCS/Wflow staging still only aligns time and applies
`rainfall_scale_factor`, so the realized rainfall field and member metadata now
describe the same transposed storm placement.

**G3 (medium) — homogeneity of the transposition domain is a review diagnostic, not an automatic gate.**
`marshfield_sst_transposition_region.md` "Required Checks" (compute AORC 72-h
maxima over footprint vs subregions; compare seasonality, depth distribution,
centroids) and `marshfield_aorc_homogeneity_results.md` describe a homogeneity
review. `run_aorc_homogeneity_diagnostic` implements the check and writes
samples, summary JSON, and sample points, but collection does not yet hard-fail
when the diagnostic indicates review is required.

**G4 (medium) — rate/return-period semantics.**
RP is `record_years / rank` on maximized depths and `λ = n/record_years`. With
threshold-driven selection the rate is now clean (members ÷ record), but RP is
on maximized depths (G1). Document that these are *maximizing-transposition POT*
return periods, distinct from SST Monte Carlo IDF.

**G5 (low) — omitted refinements.** No transposition-probability kernel, no
rotation, no intensity/elevation correction. Acceptable simplifications; record
as known limitations.

## Recommendation

1. Decide whether the production rainfall frequency should remain the current
   maximizing-transposition POT envelope or move to RainyDay-style random/kernel
   Monte Carlo annual maxima (G1/G4).
2. Promote the homogeneity diagnostic to an optional or required collection gate
   once acceptable thresholds are agreed for each Study Location (G3).
3. Add a diagnostic that compares the maximized vs historical footprint-depth
   distributions per location so the magnitude inflation from G1 is visible.
