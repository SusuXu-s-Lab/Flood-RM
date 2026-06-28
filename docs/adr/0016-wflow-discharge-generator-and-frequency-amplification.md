# Wflow as Inland Discharge Generator; Gages Calibrate; Single-K Frequency Amplification

## Status

Accepted. **Operationalizes** the inland reframe ADR-0011 deferred to "remaining work"
("inland catalog rows no longer carry an independent streamflow design member; discharge
is produced by Wflow from the realized rainfall + antecedent state, and streamflow
frequency is a validation anchor"). Does not supersede ADR-0011; it realizes it and adds
the runtime coupling mechanics. ADR-0010 (Wflow as the hydrologic bridge that *produces*
routed discharge) and ADR-0012 (coverage-box / stream-boundary handoff) still hold.

## Context

The inland Wflow-SFINCS coupling drifted from ADR-0011 in two compounding ways.

**1. The catalog made streamflow a copula dimension.** `03_build_event_catalog`
declared `driver_vector = "streamflow, rainfall"` and tagged streamflow the "inland
frequency axis," fitting its POT on a cross-basin `groupby("time").max()` envelope over
~33 reviewed gages (`basis_site_no = "network_max"`, peaks ~90,900 cfs over thousands of
sqmi). The three SFINCS crossings for `greensboro_rural` (`inflow_01/02/03`) drain only
~164–282 km². A mainstem-scale frequency envelope was being assigned to headwater creeks,
and the envelope itself (a per-timestep max across hydrologically unrelated basins) is not
a physical hydrograph.

**2. The runtime injected a scaled gage hydrograph into Wflow.**
`prepare_wflow_streamflow_realization_for_event_model` wrote the scaled USGS gage record
into the event model as external river inflow (`river_water__external_inflow_volume_flow_rate`)
**while Wflow also ran full-basin AORC rainfall**, and routed the sum to SFINCS. Below the
gage, rainfall-runoff *and* the prescribed gage flow both reached the same crossing — the
gauged catchment's response was counted twice. A single point gage also cannot coherently
force three ungauged tributary boundaries.

The deeper question a reviewer raises: *is the modeled discharge a boundary condition or an
output of the rainfall-runoff model?* It cannot coherently be both. For a small inland
catchment where one storm both floods the local domain and produces the upstream discharge,
rainfall and discharge are **not two independent forcing mechanisms** — discharge is the
routed catchment response to precipitation, storage, infiltration, drainage, and antecedent
state, which is exactly what Wflow simulates. Sampling streamflow in a copula *and* driving
Wflow with the sampled rainfall represents the discharge process twice (statistically as
sampled streamflow frequency; physically as Wflow rainfall-runoff). That is the
double-counting objection. Maduwantha et al. (2026) and Couasnon et al. fit copulas over
genuinely external drivers (storm tide/surge, sea-level anomaly, tides, rainfall) — not over
a cause and its own response.

## Decision

For inland Wflow-coupled Study Locations, treat discharge as a **hydrologic response
variable, not an independent stochastic driver.**

| Element | Role |
| --- | --- |
| Rainfall (depth/intensity/duration/structure) | stochastic design driver |
| Antecedent Moisture State / initial wetness | conditioning state |
| Wflow discharge | modeled response |
| Streamflow POT at the **Primary Reference Gage** | calibration / validation / frequency anchor |
| **Same-Frequency Amplification (K)** | empirical single multiplicative bias/frequency correction |
| Copula | dependence among true external drivers/states, not cause-and-response |

Causal order is preserved end-to-end:
`P(event) = P(rainfall, AMC)` → `Q(t) = Wflow(rainfall, AMC, basin parameters)` →
`K = streamflow_target_RP(ref gage) / Q_wflow_peak(ref gage)`.

Concretely:

- **Copula `driver_vector` becomes `"rainfall"`** (with Antecedent Moisture State sampled
  as a conditioning attribute, per ADR-0011). Streamflow is removed as a copula dimension.
  Inland AND joint-RP labeling applies only where a second *external* driver exists (e.g.
  coastal × rainfall); pure-fluvial inland design probability is carried by rainfall + AMC.

- **Primary Reference Gage** per Wflow submodel = the reviewed USGS gage on the
  highest-stream-order / primary inflow river, selected from **Wflow-native geometry**
  (largest-`uparea` crossing → upstream reach; `rivwth`/`rivdph` from `setup_rivers`), with
  NHDPlus `StreamOrde` (Strahler) as optional review cross-check only (per *USA-First Wflow
  Source Strategy*). It must be in the submodel's Wflow gauge-output set so Wflow reports a
  simulated discharge there. For `greensboro_rural` this is **`02094500` (Reedy Fork)** —
  largest Wflow `uparea` (339 km²), long record, inside the box, and its Wflow `uparea`
  matches the USGS drainage area (131 sqmi · 2.59 ≈ 339 km²).

- **Wflow is the generator.** Event models run with **rainfall + antecedent moisture only**;
  no external river inflow is injected. Wflow produces spatially coherent hydrographs at all
  crossings simultaneously, distributing to ungauged tributaries by calibrated physics.

- **Single-K amplification on Wflow output.** After per-submodel discharge is merged into
  `events/<event>/sfincs_discharge.nc`, one event-level `K` is computed at the Primary
  Reference Gage and applied **uniformly** to every handoff series and timestep (preserving
  shape, timing, and inter-tributary structure). `K` is a Maduwantha-Eq.-2 frequency
  correction applied at the model-output stage, anchored on observed streamflow frequency —
  not a prescribed boundary forcing. A configured `k_band` (e.g. 0.5–2.0) flags events whose
  `K` is implausible: that signals a poorly matched analog rainfall to fix upstream, not a
  scaling to lean on. Provenance (`K`, reference gage, target RP, raw vs amplified peak) is
  recorded per event.

- **USGS hourly instantaneous (IV) records are repurposed** from per-design-event runtime
  forcing to **calibration / validation / POT-fitting** inputs. No design run depends on a
  cached IV file; the production hard-fail (`require_instantaneous_usgs`) is removed.

- **Baseflow comes from warm hydrologic states, validated against observed low-flow.** A
  rainfall-driven Wflow run from dry channels flatlines the hydrograph and distorts the rising
  limb. Per hydromt-wflow (`model.states`), baseflow is established by **warm states**
  (`setup_cold_states` + warmup spin-up promoted to the event instate), not by injected
  forcing — so no non-standard model syntax is introduced. A **baseflow Wflow Readiness gate**
  (`validate_baseflow_against_observed`) compares the zero-rain control's per-handoff baseflow
  (which already isolates baseflow) to the observed low-flow at the Primary Reference Gage
  (median by default; `annual_mean`/`q90` configurable), transferred to each crossing by the
  drainage-area (`uparea`) ratio. A near-dry channel fails the gate so spin-up is extended,
  rather than handing SFINCS a flatlined inflow.

## Consequences

- Discharge is unambiguously a Wflow-derived response (restores ADR-0010 intent); the
  probability model and the physics model share one causal order; no double-counting.
- Three coverage-box crossings are forced coherently from one storm — impossible with
  point-gage injection.
- The frequency claim is defensible: design probability is assigned to meteorological forcing
  + antecedent state; observed streamflow POT constrains model credibility and frequency
  consistency through calibration, validation, and a single transparent `K`. This avoids
  assigning two independent return periods to one rainfall-runoff process, while still using
  gaged peak-flow records as empirical frequency information (consistent with USGS
  flood-frequency practice and Bulletin 17C).
- `05b_calibrate_wshed` shifts from auditing random catalog samples to validating Wflow's
  simulated hourly hydrograph against observed USGS IV for named historical events
  (peak/timing/volume; Nash–Sutcliffe / PBIAS) — the Wflow Readiness gate — and reporting the
  per-event `K`.
- The rejected alternative — keeping streamflow as a copula dimension and forcing Wflow
  output to it — is coherent only if streamflow is realized as an **external upstream boundary
  hydrograph** (scaled analog, not Wflow-regenerated), which suits large-mainstem
  rain-on-an-already-high-river settings, not the small rainfall-dominated Piedmont catchments
  modeled here. If a future large-mainstem domain warrants it, it is added per-domain, not
  globally.
- Implemented across `design_events.build_events` (03 copula driver vector + in-domain
  streamflow POT anchor), `wflow_runs.replay` / `wflow_runs.streamflow_realization` /
  `wflow_runs.coupling_qa` (remove injection, add single-K at merge, precip-only zero-rain
  control, K-band QA), `src/wflow_runs/cache_usgs_event_streamflow_iv.py` (calibration/validation
  fetch), and `study_location` defaults (drop forcing flags).
