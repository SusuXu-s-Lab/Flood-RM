# DER Placement Methodology For Marshfield

Decision date: 2026-05-21.

This note locks in the deterministic, citeable method for placing and sizing
distributed energy resources (DERs) on the Marshfield Control Sandbox. The
goal is not to invent behind-the-meter DERs that do not exist. The goal is to
assemble the smallest defensible DER inventory needed to validate
SSAP-derived Switch-Bounded Load Blocks for grid-forming reachability and
supply adequacy under Moring et al. (2025).

## Decision

Place and size DERs in two ordered layers, with a deferred third layer that
is only invoked if the first two fail the downstream supply-sufficiency
feasibility check:

1. **Layer 1 — Evidence-anchored DERs from public records.** One row per
   public backup-power asset documented in Marshfield's 2023 Multi-Hazard
   Mitigation Plan and aligned MassGIS/MassCEC/DOER public records. These
   are real infrastructure; the synthetic step is only the electrical
   interconnection to the SHIFT-generated network.
2. **Layer 2 — REopt resilience sizing on the Layer 1 facilities.** For each
   Layer 1 site, call NREL's REopt platform with the facility's expected
   load profile, a tier-derived `critical_load_fraction`, and a
   FEMA-standard 72-hour outage duration. REopt returns optimal PV,
   battery, and generator capacities under the resilience objective. These
   capacities define the `gfm_capable` portion of `der_inventory.parquet`.
3. **Layer 3 — Supplemental synthetic GFM-DERs via Arefifar et al. (2013).**
   Only invoked if the combined Layer 1 + Layer 2 inventory fails the
   supply-adequacy condition of the Arefifar / Barani partitioning
   formulation on the candidate Switch-Bounded Load Block set. Layer 3 is
   deferred until that feasibility check is run; this note records the
   method and citation so the deferred step is auditable.

Background grid-following PV at non-critical buildings is out of scope for
the first DER inventory pass and is not required for SSAP switch siting.

## Why Three Layers

Moring's DNMG operation requires every energized connected component to contain
a valid voltage source, and the Barani/Arefifar supply-sufficient microgrid
lineage provides a useful later check that candidate blocks have enough local
generation. The Layer 1 evidence list satisfies the resilience-narrative anchor
(real backup-power assets at named critical facilities) but does not by itself
guarantee capacity sufficiency. Layer 2 sizes the Layer 1 sites against a
published resilience objective so the capacities are defensible. Layer 3 exists
only to close residual supply-adequacy gaps if Layer 1+2 cannot satisfy the
downstream block feasibility check; if Layer 1+2 are sufficient, Layer 3 stays
unused.

This layering preserves the Marshfield evidence-first convention recorded in
[stage_b_critical_facilities_and_switches.md](stage_b_critical_facilities_and_switches.md):
public-source anchors first, model-based supplements second, and synthetic
gap-fillers only when justified by an explicit downstream feasibility check.

## Layer 1 — Evidence-Anchored DERs From The MHMP

### Seed list

The first `der_inventory.parquet` Layer 1 rows correspond one-to-one with
the generator-backed emergency support facilities named in
Marshfield's 2023 Multi-Hazard Mitigation Plan and the seeded
`critical_facilities.parquet` list in
[stage_b_critical_facilities_and_switches.md](stage_b_critical_facilities_and_switches.md):

- Town Hall
- Police/EOC
- Central Fire Station
- Council on Aging Building
- DPW Building
- Governor Winslow School
- Furnace Brook Middle School
- South River School
- Daniel Webster School
- Marshfield High School
- Martinson Elementary School
- Eames Way School
- School Administration Building
- Grace Ryder (planned/authorized)
- Tea Rock Gardens (planned/authorized)
- Marshfield Wastewater Treatment Plant (planned/authorized; tier_0 water-systems lifeline)

### Evidence and provenance

Each Layer 1 row must reference a `facility_id` in `critical_facilities.parquet`
and record `evidence_rank = town_plan_primary` (or
`evidence_rank = planned_authorized` for Grace Ryder and Tea Rock Gardens).
The Layer 1 resource is by default modeled as a `genset` (diesel or natural
gas) consistent with municipal-emergency-generator practice; an existing
public battery or PV record at the same facility upgrades the row's
`resilience_asset_type`.

Layer 1 capacities are not assumed. Capacities are assigned by Layer 2
sizing; until Layer 2 runs, Layer 1 capacities are null.

## Layer 2 — REopt Resilience Sizing

### Tool and citation

DER capacities at every Layer 1 facility are sized using NREL's REopt
platform.

Citation: D. Cutler, D. Olis, E. Elgqvist, X. Li, J. Eichman, K. Anderson,
N. DiOrio, B. Becker, R. Bromm, P. Konkel, and E. Hotchkiss.
"REopt: A Platform for Energy System Integration and Optimization." National
Renewable Energy Laboratory, NREL/TP-7A40-70022, 2017.
Public API and web tool: https://reopt.nrel.gov.

REopt is the de facto NREL/DOE methodology for resilience-driven sizing of
PV + battery + generator capacity at a specific site, given the site's load
profile, a critical-load fraction during outage, and an outage duration.

### Inputs per facility

For each Layer 1 facility, the REopt call is parameterized as follows.

| REopt input | Marshfield source |
|---|---|
| `latitude`, `longitude` | `critical_facilities.parquet` row geometry |
| Annual hourly `load_profile` | `load_profile_assignments.parquet` archetype assignment derived from NREL ResStock or ComStock for the facility's building type, per [simulated_data_protocol.md](simulated_data_protocol.md) Stage B load-profile rules. When `MARSHFIELD_USE_OEDI_LOAD_PROFILES=1`, the notebook pulls public OEDI End-Use Load Profiles for Plymouth County (`G2500230`), converts the 15-minute total-electricity energy profile to 8760 hourly kW, scales to the facility peak assumption, and records the selected building ID/profile URL in provenance. Live REopt payloads set `ElectricLoad.year = 2024` so `loads_kw` has an explicit calendar basis; this remains a calendar-alignment assumption until the full adopted weather/load year policy is locked. |
| `critical_load_fraction` | Tier-derived value from the mapping table below |
| `outage_start_hour` | Default `outage_start_hour = peak_load_hour - 12`; ensemble runs over seasonal start hours allowed in provenance |
| `outage_duration_hours` | **72** by default (FEMA Community Lifelines) |
| `pv.tilt`, `pv.azimuth`, `pv.min_kw`, `pv.max_kw` | Defaults; max bounded by roof footprint when available |
| `battery.min_kw`, `battery.min_kwh`, `battery.max_kw`, `battery.max_kwh` | Defaults |
| `generator.min_kw`, `generator.max_kw`, `generator.fuel_*` | Defaults consistent with diesel/natural gas backup-generator practice |
| Utility tariff | Current live-sizing path uses OpenEI/URDB `urdb_label` inputs for NSTAR Electric Company / Eversource South Shore `Delivery with Standard Offer` rates effective 2026-02-01. Residential/public-housing rows use South Shore R-1 (`698931e6d9dd764bf1013fef`); nonresidential rows use South Shore G-1/G-2/G-3 labels by peak-kW band, with rows below published 100 kW applicability marked `needs_account_rate_confirmation`. Facility account-specific tariffs are still required before making cost claims. |

Live data/API execution is gated. Set `MARSHFIELD_USE_OEDI_LOAD_PROFILES=1`
to refresh public OEDI ResStock/ComStock profiles before sizing. The notebook
validates that every row to be sized has an assigned electrical interconnection
and an 8760 hourly profile before constructing the REopt client. Set
`MARSHFIELD_RUN_LIVE_REOPT=1` only for an intentional API-backed refresh. Set
`MARSHFIELD_REOPT_LIMIT=1` for the first smoke run; unset it after inspecting
the cached result to refresh all Layer 1 rows.

When no live REopt API key or cached REopt response is available, integration
smoke assets may be refreshed with the explicit offline surrogate
`OfflineReoptSurrogateClient`. This path preserves the Layer 2 payload,
critical-load-fraction mapping, outage duration, and provenance fields, but
sizes only a genset from the outage-window peak critical load plus a reserve
margin. Rows produced this way carry `reopt_status =
optimal_offline_surrogate` and `reopt_version =
offline_reopt_surrogate.v0.1`; they are suitable for PMONM/DynaGrid plumbing
tests, not REopt cost or techno-economic claims.

### Outputs persisted to `der_inventory.parquet`

For each facility, REopt returns:

- `pv_kw` — recommended photovoltaic capacity (grid-following only).
- `bess_kw`, `bess_kwh` — battery power and energy capacity (grid-forming
  capable when inverter is GFM-rated; default `gfm_capable = true`).
- `genset_kw` — generator capacity (grid-forming capable;
  `gfm_capable = true`).
- `feasible` — boolean flag indicating REopt found a solution meeting the
  critical-load fraction through the outage duration.
- Row-level `source_provenance` records the REopt status, run UUID, API
  version, payload digest, load year, tariff placeholder, outage inputs, and
  Layer 1 provenance. The full REopt response is stored separately in the
  redacted `reopt_cache/` file keyed by payload digest.

### Tier → critical-load fraction mapping

This is the locked mapping from Marshfield's criticality tiers to REopt's
`critical_load_fraction` input. Values follow the NREL REopt resilience
convention published in Anderson et al. (2017) and Stout et al. (2018).

| Marshfield tier | `critical_load_fraction` | Basis |
|---|---:|---|
| `tier_0_life_safety` | **1.00** | Life-safety facilities maintain essentially all normal load during outage (EOC/dispatch, acute medical, major water/wastewater treatment). REopt life-safety convention. |
| `tier_1_response` | **0.50** | Response facilities maintain about half of normal load — dispatch, communications, server cooling, partial HVAC for crew comfort. REopt essential-commercial convention. |
| `tier_2_lifeline_support` | **0.25** | Lifeline-support facilities (schools as shelters, town hall, public works, lifeline water assets not promoted to tier_0) maintain shelter-grade essentials only. REopt typical-commercial convention. |
| `tier_3_standard` | not sized | Standard load is not critical infrastructure and is not REopt-sized. Tier_3 rows are intentionally excluded from the Layer 2 sizing loop. |

Reasoning for tiering: a school used as a heating shelter does not need its
full normal HVAC, kitchen, and instructional plug load; a police dispatch
center cannot drop comms or server-room cooling; a hospital cannot drop
anything. A single uniform fraction either oversizes lifeline-support sites
or undersizes life-safety sites. Tiering aligns REopt's sizing input with
the actual essential-load engineering of each facility class.

There is no single canonical FEMA or IEEE table that maps criticality tier
directly to critical-load fraction. The 1.00 / 0.50 / 0.25 values used here
are the conventional REopt three-tier mapping documented in published
NREL/DOE resilience analyses (Anderson et al. 2017; Stout et al. 2018) and
in REopt user documentation, qualitatively backed by IEEE Std 446-1995
(Orange Book) essential-load classifications.

### Outage duration default

**72 hours** is the locked default `outage_duration_hours` for all Layer 2
REopt calls.

Citation: Federal Emergency Management Agency. "Community Lifelines
Implementation Toolkit, Version 2.0." FEMA, 2023.
https://www.fema.gov/sites/default/files/documents/fema_community-lifelines-toolkit-2.0.pdf

The FEMA Community Lifelines framework defines stabilization as restoration
of essential lifeline services within 72 hours of an incident. Sizing
backup capacity for a 72-hour outage matches the stabilization horizon
emergency managers plan against.

Sensitivity runs at 24 hours (utility-restoration benchmark) and
168 hours (7-day extreme-event/hurricane benchmark) are allowed and recorded
in `source_provenance` per facility, but they do not change the default.

## Layer 3 — Deferred Supplemental DERs Via Arefifar 2013

Layer 3 is invoked only when the combined Layer 1 + Layer 2 inventory fails
the supply-adequacy condition required by the downstream supply-sufficient
microgrid partitioning MILP. Until that downstream feasibility check is
implemented and reported as infeasible, Layer 3 is not run and no synthetic
supplemental DERs are added.

Citation: S. A. Arefifar, Y. A.-R. I. Mohamed, and T. H. M. El-Fouly.
"Optimum Microgrid Design for Enhancing Reliability and Supply-Security."
*IEEE Transactions on Smart Grid*, vol. 4, no. 3, pp. 1567–1575, 2013.
DOI: 10.1109/TSG.2013.2259854. Cited by Moring et al. (2025) as reference
[18] in the supply-sufficient microgrid construction lineage.

Layer 3 supplemental DER rows must record `evidence_rank =
synthetic_supply_sufficiency` and `placement_rule =
arefifar_2013_supply_adequacy`. Candidate buses are restricted to load-bearing,
preferably three-phase buses within a documented geographic radius of any
prospective Switch-Bounded Load Block that lacks sufficient generation.

## Downstream Consumers — SSAP Blocks, Moring RPOP, And Supply-Adequacy Checks

The `der_inventory.parquet` produced by Layers 1 and 2 (and, conditionally,
Layer 3) is consumed after SSAP switch placement. It validates that the
SSAP-derived `switch_bounded_load_blocks` have grid-forming reachability and
enough local resource capacity for the Moring et al. (2025) DNMG/RPOP
operation layer.

Barani et al. and Arefifar et al. are retained as supply-sufficient
microgrid partitioning/design references for later validation and possible
Layer 3 DER gap filling. They do not produce the adopted
`controllable_switches.parquet`; switch placement is owned by
[rpop_ready_switch_siting.md](rpop_ready_switch_siting.md)'s SSAP workflow.

Citation: M. Barani, J. Aghaei, M. A. Akbari, T. Niknam, H. Farahmand, and
M. Korpås. "Optimal Partitioning of Smart Distribution Systems into
Supply-Sufficient Microgrids." *IEEE Transactions on Smart Grid*, vol. 10,
no. 3, pp. 2523–2533, 2019. DOI: 10.1109/TSG.2018.2803215. Cited by Moring
et al. (2025) as reference [17].

See [rpop_ready_switch_siting.md](rpop_ready_switch_siting.md) for the adopted
Stage B switch-placement method.

## `der_inventory.parquet` Schema Sketch

One row per placed DER unit. Detailed schema lock lands in
[simulated_data_protocol.md](simulated_data_protocol.md); this section names
the fields that must round-trip from Layer 1 + Layer 2 into the partitioning
MILP input.

| Field | Design |
|---|---|
| `sandbox_id` | `marshfield` |
| `der_id` | Stable ID: `marshfield:asset:der:<facility_token>:<resource_token>` |
| `facility_id` | Must reference `critical_facilities.parquet` for Layer 1/2 rows; nullable only for Layer 3 supplemental synthetic rows |
| `load_asset_id` | Preferred service-point binding via `critical_load_assignments.parquet`; nullable when the DER is bound to a feeder proxy or supplemental synthetic bus |
| `bus` | Existing OpenDSS bus the DER is interconnected at |
| `block_id` | Switch-Bounded Load Block containing the DER or served load once block assignment is resolved |
| `assignment_status` | `assigned` when the row has an electrical bus or block binding; `unassigned` when the row is evidence-backed but not yet operationally located |
| `unassigned_reason` | Machine-readable reason for an unassigned row, such as `pending_critical_load_assignment`; null only when `assignment_status = assigned` |
| `phases` | OpenDSS phase string |
| `nominal_voltage_kv` | Interconnection nominal voltage |
| `resilience_asset_type` | `pv`, `bess`, `genset`, `microgrid`, `resilience_hub`, or `composite` |
| `pv_kw` | Recommended PV capacity from REopt (Layer 2), nullable for non-PV rows |
| `bess_kw`, `bess_kwh` | Battery power and energy capacity (Layer 2), nullable for non-BESS rows |
| `genset_kw` | Generator capacity (Layer 2), nullable for non-genset rows |
| `gfm_capable` | Boolean; `true` for BESS and genset (grid-forming inverter or synchronous), `false` for PV-only rows |
| `placement_rule` | `evidence_anchored_mhmp`, `reopt_resilience_sizing`, or `arefifar_2013_supply_adequacy` |
| `evidence_rank` | `town_plan_primary`, `planned_authorized`, `massgis_primary`, or `synthetic_supply_sufficiency` |
| `confidence` | `high`, `medium`, or `low` |
| `outage_duration_hours` | REopt input recorded for provenance (default 72) |
| `critical_load_fraction` | REopt input recorded for provenance (tier-derived) |
| `reopt_feasible` | REopt feasibility flag for Layer 2 rows |
| `source_provenance` | JSON with REopt status, run UUID, payload digest, ResStock/ComStock archetype, tariff, weather/load year, seed, and outage inputs; the full redacted REopt response is stored in `reopt_cache/` keyed by payload digest |
| `schema_version` | `stage_b_der_inventory.v0.1` |

## Validation Metrics

Minimum validation for the Layer 1 + Layer 2 DER inventory:

- Count of DER rows by `placement_rule`, `resilience_asset_type`, and
  `gfm_capable`.
- Coverage: every Layer 1 facility in the MHMP seed list produces at least
  one Layer 2-sized DER row, or an explicit `reopt_feasible = false`
  exception is recorded in provenance.
- Assignment completeness: every DER row must either reference an existing
  OpenDSS `bus` or Switch-Bounded Load Block with `assignment_status =
  assigned`, or carry `assignment_status = unassigned` plus an
  `unassigned_reason`. Unassigned rows remain evidence inventory only and are
  not treated as PowerModelsONM voltage sources.
- DER interconnection assignment: Layer 1/2 evidence-backed DER rows inherit
  `load_asset_id`, `bus`, `phases`, and `nominal_voltage_kv` only from
  `critical_load_assignments.parquet` rows with `assignment_status =
  assigned`. A critical facility with backup-power evidence but no valid
  electrical service assignment stays `unassigned`.
- Total `genset_kw` and `bess_kw` by criticality tier, cross-checked against
  the static peak `kW` of the associated `load_asset_id`.
- Phase compatibility: every `gfm_capable = true` DER row is three-phase or
  the connected block has only single-phase or two-phase load (Moring
  eqn. 5 GFM-eligibility precondition).
- Spatial distribution plot of DERs on the SHIFT network with criticality
  tier overlay.
- Confirmation that Layer 3 supplemental DERs were not synthesized unless
  the Layer 1 + Layer 2 inventory fails the downstream supply-adequacy
  feasibility check, with the failing block report preserved.

## Non-Goals

- Do not seed behind-the-meter residential or commercial PV at every
  non-critical building in the first DER inventory pass. Background BTM PV
  is deferred until after the SSAP block and RPOP validation pipeline is working
  end-to-end.
- Do not assume utility-validated DER inventory or interconnection records.
  Layers 1 and 2 produce public-evidence-anchored, REopt-sized capacities,
  not utility data.
- Do not invoke Layer 3 supplemental DERs before the Layer 1 + Layer 2
  inventory has been tested against the downstream supply-adequacy feasibility
  check.
- Do not use REopt's economic objective (NPV, LCOE) as the placement
  driver. The driver is the resilience objective with a critical-load
  fraction and outage duration; economic outputs are recorded but do not
  alter placement.

## References

- Cutler, D.; Olis, D.; Elgqvist, E.; Li, X.; Eichman, J.; Anderson, K.;
  DiOrio, N.; Becker, B.; Bromm, R.; Konkel, P.; Hotchkiss, E. "REopt: A
  Platform for Energy System Integration and Optimization." NREL/TP-7A40-70022,
  National Renewable Energy Laboratory, 2017.
  https://www.nrel.gov/docs/fy17osti/70022.pdf
- Anderson, K.; Olis, D.; Becker, B.; Cook, J.; Cutler, D.; Inskeep, B.
  "Sustainability of the Electricity Grid: A Resilience Framework."
  NREL/CP-7A40-67687, National Renewable Energy Laboratory, 2017.
- Stout, S.; Cliff, K.; Anderson, K.; Ericson, S.; Jorgenson, J.
  "Distributed Energy Planning for Climate Resilience." NREL/TP-7A40-71310,
  National Renewable Energy Laboratory, 2018.
- Federal Emergency Management Agency. "Community Lifelines Implementation
  Toolkit, Version 2.0." FEMA, 2023.
  https://www.fema.gov/sites/default/files/documents/fema_community-lifelines-toolkit-2.0.pdf
- Arefifar, S. A.; Mohamed, Y. A.-R. I.; El-Fouly, T. H. M. "Optimum
  Microgrid Design for Enhancing Reliability and Supply-Security."
  *IEEE Trans. Smart Grid*, vol. 4, no. 3, pp. 1567–1575, 2013.
  DOI: 10.1109/TSG.2013.2259854.
- Barani, M.; Aghaei, J.; Akbari, M. A.; Niknam, T.; Farahmand, H.; Korpås, M.
  "Optimal Partitioning of Smart Distribution Systems into Supply-Sufficient
  Microgrids." *IEEE Trans. Smart Grid*, vol. 10, no. 3, pp. 2523–2533, 2019.
  DOI: 10.1109/TSG.2018.2803215.
- Moring, H.; Poolla, B. K.; Nagarajan, H.; Mathieu, J. L.; Bernstein, A.;
  Fobes, D. M. "Reconfiguration and Real-Time Operation of Networked
  Microgrids Under Load Uncertainty." arXiv:2504.15084, 2025.
- IEEE Std 446-1995 (R2000). "IEEE Recommended Practice for Emergency and
  Standby Power Systems for Industrial and Commercial Applications."
