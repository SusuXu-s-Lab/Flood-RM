# Reservoir-Regulated Mainstem Inflow for Austin p5u (Option B, per-domain)

## Status

**Proposed** (follow-on to ADR-0016, not yet accepted). Drafts the reservoir-aware /
gage-as-boundary handling that ADR-0016's Consequences flagged for large regulated mainstems:
"If a future large-mainstem domain warrants it, it is added per-domain, not globally." Austin
p5u is that domain. ADR-0016 (rainfall-driven Wflow generation), ADR-0011 (copula compound
dependence), and ADR-0012 (coverage-box / stream-boundary handoff) still hold for the
tributary crossings.

## Context

ADR-0016 makes inland discharge a **Wflow response** to rainfall + antecedent moisture, valid
where the same storm both floods the box and produces the upstream flow (small Piedmont
catchments like Greensboro Reedy Fork). Austin's active `austin_p5u` domain violates that
premise at its dominant crossing:

- **`austin_p5u_inflow_01` drains ~10,565 km²** — the Colorado River mainstem — versus the box's
  own area and the small tributary crossings (`inflow_02..08` at ~10–254 km²).
- The mainstem is **regulated by the Highland Lakes (LCRA)**: Buchanan, Inks, LBJ, Marble Falls,
  Travis, Austin. Mainstem discharge at the box is governed by **reservoir release schedules and
  flood operations**, not by rainfall-runoff over the modeled domain.
- The contributing area is far larger than any single AORC SST storm footprint, so a rainfall
  field transposed for the box does not produce the mainstem hydrograph.
- **No reviewed gage sits on the mainstem crossing's reach** (nearest reviewed gauges are small
  urban tributaries, 58–324 km²; the Colorado-at-Austin gage `08158000` is 101,032 km², ~10×
  downstream/larger). So neither a scale-matched validation anchor nor a clean Wflow-generated
  hydrograph is available there.

Consequently, running rainfall through Wflow to *generate* the 10,565 km² mainstem inflow is
not physically defensible, and the single-K amplification has no scale-matched reference for it.
The tributary crossings, by contrast, are genuine rainfall-runoff catchments and remain valid
under ADR-0016.

Crucially, this is also the setting where the **double-counting objection reverses**: because
the regulated mainstem stage and the local box rainfall are driven by *different* mechanisms
(upstream basin + reservoir operations vs. the local storm), they are **two genuinely external
drivers** — exactly when a copula earns its place (cf. ADR-0016's Maduwantha rationale, and
Couasnon et al. for riverine–coastal/fluvial compound boundaries).

## Decision (proposed)

Introduce a **per-crossing realization mode** so a single domain can mix ADR-0016 rainfall-driven
tributaries with an externally-prescribed regulated-mainstem boundary.

### 1. Per-crossing realization mode (config)

Each Stream-Boundary Handoff Source declares how its inflow is realized:

```yaml
inland_coupling:
  crossing_realization:
    default: wflow_response            # ADR-0016 rainfall-driven (tributaries)
    overrides:
      austin_p5u_inflow_01: external_boundary   # regulated Colorado mainstem
  external_boundary:
    austin_p5u_inflow_01:
      method: scaled_gage_analog       # B1 (default) | reservoir_release (B2)
      boundary_gage: "08158000"        # Colorado Rv at Austin (mainstem frequency anchor)
      drainage_area_ratio: true        # area-scale 08158000 -> crossing uparea (10565/uparea_gage)
      # method: reservoir_release      # B2: LCRA regulated-release schedule (future data work)
      # reservoir_operations: data/sources/lcra/highland_lakes_releases.csv
```

- `wflow_response` crossings: unchanged ADR-0016 path (Wflow generates, single-K, baseflow gate).
- `external_boundary` crossings: realized from observed mainstem records (or reservoir releases),
  **not** Wflow rainfall-runoff. Wflow may still *route* it, but does not *generate* it.

### 2. Realizing the mainstem boundary

**B1 — Scaled gage analog (recommended first step, coastal-literal).** Select a historical
mainstem analog at `boundary_gage`, scale its observed hydrograph to the design target by
Maduwantha Eq. 2 (`K = Q_target / Q_obs`), area-ratio-transfer to the crossing uparea, and feed
it as the upstream boundary inflow. This is the inland analog of the coastal scaled-surge
boundary — appropriate precisely because the mainstem is an external driver here.

**B2 — Reservoir-release-aware (later, data-dependent).** Replace/augment the analog with LCRA
regulated-release schedules (and flood-operation rules) so the mainstem boundary reflects
operations rather than a naturalized record. Requires sourcing Highland Lakes operations data;
deferred until that source is reviewed.

### 3. Compound dependence for austin_p5u (copula re-enabled, per-domain)

Because the regulated mainstem and the local rainfall are external drivers here, austin_p5u
**may** fit a bivariate copula on `(mainstem boundary discharge at boundary_gage, local
rainfall)` with a sampled lag — the genuine compound fluvial-pluvial case (ADR-0011 machinery,
the shared coastal `design_catalog.py` vine, *not* the inland rainfall-only module). The
mainstem driver is realized as an external boundary (B1/B2); the rainfall driver realizes the
pluvial-on-box field and the tributary Wflow forcing. This is opt-in per domain via
`event_catalog.dependence.driver_vector: ["mainstem_discharge", "rainfall"]` and does **not**
change Greensboro or the other small Austin tributaries.

### 4. Amplification, baseflow, and validation under the hybrid

- **Single-K** applies only to the `wflow_response` (tributary) discharge, anchored on a
  tributary reference gage (Barton Creek `08155400`). The `external_boundary` series is already
  at-frequency (scaled to target) and is excluded from K.
- **Baseflow validation** runs per realization mode: warm-state low-flow check for tributaries;
  for the boundary, validate against the mainstem gage's observed low-flow directly.
- **Wflow Readiness (5b)**: score tributary crossings vs Barton Creek IV; score the mainstem
  boundary vs `08158000` IV.

## Consequences

- austin_p5u becomes a documented **hybrid domain**: rainfall-driven tributaries + an external
  regulated-mainstem boundary, each validated against a scale-appropriate gage. No 10× scale
  mismatch, no rainfall-generated mainstem.
- The copula is re-enabled **only** where drivers are genuinely external (mainstem × rainfall),
  honoring ADR-0016's no-double-count principle rather than contradicting it.
- New config surface (`crossing_realization`, `external_boundary`) and a boundary-inflow
  realization function in the coupling layer; the existing rainfall-driven path is unchanged for
  every other crossing and location.
- B2 (reservoir operations) is gated on sourcing LCRA Highland Lakes release data; B1 is
  implementable immediately from USGS mainstem records.

### Open questions before acceptance

- Is `08158000` (Colorado-at-Austin, downstream of the box) the right mainstem frequency anchor,
  or is an *upstream* Colorado gage closer to `inflow_01`'s reach available? (Reach-trace needed.)
- Does austin_p5u's design intent want the **regulated** (operations) or **naturalized** mainstem
  frequency? That decides B1 vs B2 and the copula marginal.
- Should the smaller Austin tributary crossings (`inflow_02 ~254 km²`) stay `wflow_response`, or
  is there a size threshold above which `external_boundary` is preferred?

## Alternatives considered

- **Force Option A everywhere (status quo).** Rejected for the mainstem: rainfall-generated
  10,565 km² discharge is not physical, and there is no scale-matched K/validation anchor.
- **Drop the mainstem crossing.** Rejected: it is the dominant flood driver for the box.
- **One global compound copula for all inland domains.** Rejected: it would re-introduce the
  double-counting ADR-0016 fixes for the small rainfall-dominated catchments.
