# RPOP-Ready Switch Siting For Marshfield

Decision date: 2026-05-21. Updated 2026-05-27.

This note defines the Stage B method for replacing the initial geometric
`controllable_switch_synthesis` heuristic with a citeable optimization method
run directly on the NLR SHIFT-generated feeder graph. The goal is not to infer
a real Eversource switch inventory. The goal is to synthesize logical
Controllable Switches that create useful Switch-Bounded Load Blocks for
DNMG/RPOP studies over the Marshfield Grid Dataset.

## Adopted Method

Stage B switch siting is a DNMG substrate-building problem on a synthetic
NLR SHIFT radial network, not a utility switch-investment study. The active
method is Usberti et al.'s polynomial-time exact Sectionalizing Switch
Allocation Problem (SSAP) formulation: for each transformer-bridged feeder
island, choose a fixed budget of sectionalizing switch locations from a
documented ordinary-line candidate set that minimizes an additive
expected-energy-not-supplied proxy. Flood-RM adapts the SSAP topology, load,
exposure, and admissible-candidate inputs to the public-data synthetic-grid
setting, then validates that the resulting Switch-Bounded Load Blocks can be
used by DNMG/RPOP.

The Marshfield objective is a reproducible dynamic-microgrid sectionalizing
proxy rather than a utility-grade reliability claim: minimize connected load at
risk across switch-induced zones. The nominal DNMG run uses homogeneous edge
exposure so dense coastal topology is not penalized for short synthetic line
segments; line-length-weighted exposure is retained only as a sensitivity
view. This requires only the generated feeder topology, transformer winding
connectivity, source/root buses, line lengths, phases, and static load. It does
not require utility failure rates, customer interruption cost surveys, repair
times, switch costs, switching-station costs, or private restoration
procedures.

The SSAP output produces the normally closed sectionalizing switches and the
first `switch_bounded_load_blocks`. Cross-feeder normally open tie switches are
an optional second-stage augmentation, generated from phase/voltage-compatible
nearest-neighbor feeder pairs and validated against the same Moring/RPOP
radiality assumptions. A broader utility reliability-cost MILP is deliberately
out of scope until the repo has defensible public inputs for those costs and
restoration schemes.

## Plug-and-Play SSAP

The plug-and-play method means the citeable SSAP optimizer stays fixed while
Marshfield supplies auditable inputs around it. The optimizer receives a
rooted radial topology, a switch budget, static load weights, edge exposure
weights, and an admissible ordinary-line candidate set. Those are
caller-supplied inputs; they are not changes to the SSAP algorithm.

This keeps the DNMG substrate agnostic to flood scenarios. Flood depth, FEMA
zone membership, SFINCS exposure, shoreline proximity, and ERAD fragility are
not objective terms, admissibility gates, or tie-breakers for switch
placement. They enter later as asset-state, availability, and stress-test
layers once the switch network already exists.

The only current candidate-policy plug-ins are realism/exportability gates for
the synthetic feeder graph: optional minimum line exposure, minimum downstream
block bus count, minimum downstream block load, and a documented per-component
candidate screen for transformer-bridged feeder islands. The nominal
Marshfield DNMG run sets the line-exposure gate to zero and uses homogeneous
exposure so dense short-line areas are not disadvantaged by line length. These
gates prevent SSAP from spending switches on source stubs, tiny generated
spurs, non-useful DNMG block boundaries, or intractably large candidate
frontiers. They do not impose geographic de-clustering, so clusters are
allowed when the topology, switch budget, and static load/exposure objective
make adjacent block boundaries defensible.

## SSAP Frontier Placement

Marshfield now uses exact SSAP frontier placement over a documented admissible
candidate set, global marginal-benefit budget selection, and DNMG block
validation.

Usberti et al. show that SSAP is tractable on radial distribution networks
when the objective is additive, which matches the NLR SHIFT feeder islands and
the Flood-RM topology/load exposure proxy. The instance supplies a fixed study
budget `m`; each component's fixed-budget point is solved exactly over its
admissible candidate set, and the global selector repeatedly accepts the next
switch with the largest marginal reduction in the SSAP objective until the
nominal study budget is exhausted. Transformer bridges are topology edges, not
switch candidates: they keep MV/LV service components connected for load,
phase, and block accounting, while only ordinary line edges can receive
sectionalizing switches.

When a per-feeder fairness floor is configured but the global budget cannot
cover every eligible island, the scarce floor budget is allocated by next
marginal SSAP benefit rather than by feeder or component identifier order.
This keeps the fairness floor from deterministically starving later-generated
coastal feeder regions while still avoiding flood, shoreline, FEMA, SFINCS,
or ERAD signals in placement.

The nominal objective uses homogeneous per-edge exposure (`1.0` per eligible
line) because DNMG/RPOP needs useful switch-bounded blocks in dense topology,
not only long-line fault exposure reduction. A length-weighted objective is
retained as sensitivity provenance. If the two frontiers disagree, the
artifact records both objective summaries; no flood, shoreline, FEMA, SFINCS,
or ERAD signal participates in either frontier.

Barani et al. (2019) and Arefifar et al. (2012/2013) are retained as later
DER-aware supply-adequacy checks for the resulting blocks. They are no longer
the adopted switch-placement method, because they partition or design
microgrid areas rather than directly placing controllable switches on the
SHIFT feeder graph.

## Decision

Use an artifact-first SSAP workflow:

1. Root each SHIFT feeder at its source or substation-adjacent bus.
2. Convert the radial primary feeder into the arborescence required by SSAP,
   including non-switchable transformer bridge edges so service components are
   connected for DNMG block accounting.
3. Assign each candidate edge two exposure views: homogeneous `1.0` for the
   nominal DNMG-block placement run, and line length in meters for sensitivity
   provenance.
4. Assign each downstream load point its static `kW` from the Baseline Network,
   with optional Marshfield Service-Priority Weight multipliers after critical
   load assignments validate.
5. Apply documented candidate-admissibility gates, including a per-component
   candidate screen when needed to keep the exact frontier solve tractable on
   transformer-bridged feeder islands.
6. Solve exact SSAP frontier points for loaded feeder components; evaluate
   points lazily so only globally competitive marginal points are solved.
7. Select exactly the nominal sectionalizing budget globally by marginal SSAP
   benefit, not by fixed switch density.
8. Split each selected line and insert a normally closed OpenDSS
   `Line.Switch=Yes` sectionalizing switch.
9. Optionally add a small number of normally open cross-feeder ties from
   eligible phase/voltage-compatible nearest-neighbor bus pairs.
10. Open all synthesized Controllable Switches and compute the resulting static
   Switch-Bounded Load Blocks.
11. Export stable OpenDSS, Parquet, ONM-settings, block, and provenance
    artifacts.

This is an optimization-generated planning proxy. It is academically grounded,
but it is not utility-validated asset data.

## Why Not Utility Reliability-Cost Optimization Yet

Full switch/tie planning in the utility reliability literature often requires
customer counts, customer interruption costs, failure rates, repair times,
switching times, switch costs, tie-line costs, candidate restoration schemes,
and sometimes stochastic weather scenario generation. Marshfield does not yet
have enough local public-data support to defend all of those inputs at
utility-planning fidelity.

The exact radial SSAP adaptation is therefore the right first method:

- It uses the data already available in the Baseline Network and Stage B
  artifacts.
- It still performs an optimization over the documented admissible ordinary-
  line candidate set instead of hand-assigning switch locations.
- It preserves the block-and-switch abstraction required by Moring et al.
  (2025).
- It directly supports DNMG/RPOP substrate generation while avoiding
  unsupported utility-cost claims.
- It remains inspectable in a notebook, deterministic, and easy to rerun.

## Data Inputs

Required now:

- Baseline Network buses, lines, feeder IDs, source/root buses, coordinates,
  phases, line lengths, and static load `kW/kvar`.
- Existing Asset Registry rows for loads, buses, transformers, sources, and
  lines.
- Switch budget policy, recorded as a target switch density, target block count,
  per-feeder maximum, or fixed study budget.

Optional before first switch siting:

- Public critical-facility evidence and Marshfield Service-Priority Weights
  once those assignments validate. These may weight the load term in the SSAP
  objective, but they are not required for the first topology/load run.

Not required for switch siting:

- Load time series.
- Flood depth, FEMA flood zones, SFINCS exposure, or ERAD failure probability.
- Private utility switch records.
- Utility failure rates, repair times, customer interruption cost curves, or
  customer counts.
- DER locations and capacities. DER inventory is required later for
  supply-adequacy and GFM reachability validation, not for SSAP switch siting.

Required later for RPOP/MPC:

- Load-profile assignments from public building-stock sources such as NREL
  End-Use Load Profiles, ResStock, ComStock, and dsgrid.
- Block-level load uncertainty envelopes.
- DER and grid-forming eligibility artifacts.
- Event-conditioned asset-state and availability artifacts.

## Candidate Generation

### Sectionalizing Candidates

Candidate sectionalizing switches are ordinary enabled feeder line sections in
the transformer-bridged rooted SHIFT feeder arborescence. A line is eligible
when:

- It is on an ordinary feeder line, not a source stub, disabled line, regulator
  bypass, transformer edge, or already-synthesized switch/tie line.
- It splits a feeder into two connected components when opened.
- It has phase and voltage metadata sufficient for OpenDSS and ONM export.
- Its downstream component contains at least one load bus or enough topology to
  be a meaningful block boundary after minimum-block filters are applied.

Minimum-block filters are candidate-admissibility filters applied before SSAP
optimization. If opening a candidate edge would create a tiny spur block below
the configured `min_block_bus_count` or `min_block_kw` threshold, that edge is
excluded from the candidate set while the rest of the feeder remains in the
objective. These filters are topology/load realism gates, not flood-risk
scoring or geographic spacing rules.

Large synthetic feeder islands may also use a recorded per-component
candidate-count screen. This is an admissible-set definition, not a relaxation
inside the optimizer: SSAP remains exact for the candidate set it is given, and
the cap is recorded in switch provenance and diagnostics.

The OpenDSS export must split the selected line into two line segments with a
`Switch=Yes` line between them. A parallel switch line is not sufficient,
because Switch-Bounded Load Blocks are defined by opening Controllable Switches.

### Tie Candidates

Candidate tie switches are normally open cross-feeder connections. Prefer
locations that satisfy:

- Endpoints are on different feeders.
- Endpoints are at the same nominal voltage level.
- Phase compatibility is defensible; use the common phase count or restrict to
  compatible phase sets when phase labels are available.
- Geographic distance is short enough to be a plausible reserve/tie connection.
- Closing the tie can create a restoration path between Switch-Bounded Load
  Blocks without violating radial DNMG assumptions.
- Per-feeder-pair and total-count caps prevent unrealistic tie density.

The first SSAP implementation may omit ties. When ties are synthesized, the tie
candidate distance is a feasibility and cost proxy, not the objective by itself.

## Candidate Scoring

The primary SSAP objective is expected downstream unserved load:

```text
minimize sum_over_fault_edges(exposure_weight(edge) * interrupted_load_after_switching(edge))
```

For the nominal Marshfield DNMG run:

- `exposure_weight(edge) = 1.0` for every objective edge.
- candidate switch edges are ordinary physical line edges passing
  exportability, block-quality, and per-component admissible-set gates.
- `interrupted_load_after_switching(edge)` is the static downstream `kW` that
  remains isolated by the nearest upstream sectionalizing switch under the
  candidate allocation.

For sensitivity provenance only, the exposure term may be replaced by valid
line length in meters. Flood exposure remains excluded from placement scoring.

After critical-load assignments validate, the load term may be upgraded to a
service-priority-weighted proxy:

```text
effective_load_kw = static_kw * service_priority_weight
```

or a blended objective that preserves total `kW` while adding a critical-load
bonus. Flood exposure remains excluded from placement scoring.

Do not introduce a binary uninterruptible/interruptible load label. Fayyazi et
al.'s UL/IL distinction maps to Marshfield Service-Priority Weights.

## Selection Rule

Use exact deterministic SSAP frontier selection:

1. Build a transformer-bridged rooted feeder arborescence.
2. Apply ordinary-line, exportability, block-quality, and per-component
   admissibility gates to form the candidate set.
3. Compute load and homogeneous/length-weighted exposure arrays required by
   the Galias/Usberti tree-dynamic-programming formulation.
4. Solve exact SSAP for `m = 0..M` candidate budgets on each loaded component.
5. Select the nominal study budget globally by marginal benefit while
   preserving each component's exact fixed-budget SSAP placement.
6. Validate that opening the selected switches creates static
   Switch-Bounded Load Blocks for DNMG/RPOP.
7. Emit provenance with the objective value, frontier budget, marginal
   benefit, exposure policy, sensitivity summary, rejected candidates, and
   source artifacts.

Optional tie augmentation is deterministic but separate from SSAP:

1. Generate cross-feeder candidate ties from nearest phase/voltage-compatible
   bus pairs.
2. Accept candidates under feeder-pair caps, total tie caps, length caps, and
   radiality validation.
3. Record tie provenance separately from the SSAP sectionalizing provenance.

## Artifact Contract

The implementation must produce or update:

- `controllable_switches.parquet`
- `switches.dss`
- ONM settings sidecar identifying dispatchable switches and normal states
- `switch_bounded_load_blocks.parquet` as Stage B Candidate Control Units
- validation summary with scoring distributions and block-quality metrics
- plots of switches, blocks, critical-facility overlay, and tie connections

Every Controllable Switch row must include:

- stable `switch_id`
- `switch_role`: `sectionalizing` or `tie`
- OpenDSS element/source ID
- from/to buses, phases, normal state, initial state
- `placement_rule = ssap_radial_sectionalizing_switch_allocation` for SSAP
  sectionalizing switches, or `phase_compatible_nearest_neighbor_tie` for
  optional ties
- sub-rule: `galias_usberti_ssap` or `cross_feeder_restoration_tie`
- full JSON provenance with objective terms, budgets, caps, and thresholds
- whether it opens an existing line

## Block Invariant Contract

Decision date: 2026-05-22.

Per Moring et al. (2025) §II-A and §II-B every row of
`switch_bounded_load_blocks.parquet` must pass the Block Invariant Contract
before the Augmented Network artifact set ships:

- **B**: block has `load_kw > 0` **or** a substation PCC. A load-less trunk
  block that contains a `sources.csv` PCC is a valid trunk; a block with
  neither load nor PCC is a dead spur and is a hard fail.
- **C**: line edges inside a block must not cross voltage classes. Multi-
  voltage blocks are allowed only when bridged by a transformer winding
  (Moring §II-A treats transformers as ordinary network elements that do
  not partition blocks). A voltage transition across a `line_class=line`
  edge with no transformer between the endpoints is a data error and is a
  hard fail.
- **D/E**: every block records `phi_max ∈ {1,2,3}` (Moring §II-B-2a) and the
  union of per-bus phase sets.
- **F**: every block is acyclic (Moring §II-B "spanning tree, devoid of
  loops"). Edge count includes transformer winding-bus bridges.
- **G**: every block records its **Block Voltage Source Reachability**:
  `substation_pcc`, `gfm_eligible_der`, `both`, `pending_der_inventory`, or
  `none`. `pending_der_inventory` is reported while the DER Assignment
  Completeness Gate has only evidence-backed unassigned DER rows. A
  GFM-capable DER becomes a local voltage source only after it carries
  `assignment_status = assigned` and a valid `bus` or `block_id`. G becomes a
  `none` means the static block has no local voltage source; it is allowed
  only when the separate switch-reachability precheck proves the block can
  join a source-hosting connected component.
- Source-less blocks are not invalid merely because they lack a local PCC or
  GFM-DER. A separate switch-reachability precheck records whether they can
  join a source-hosting block through Controllable Switches; the DNMG/RPOP
  optimizer then decides which switch closures keep the energized connected
  components radial and source-backed.

Implementation:
`power.blocks.derive_validated_blocks(...)` raises
`BlockInvariantViolation` on B/C/D/E/F. The companion
`build_block_artifact_rows(...)` emits the canonical Parquet rows; the
companion `inject_block_ids_into_onm_settings(...)` propagates per-load
`block_id` and a per-block `microgrid` section into the ONM settings
sidecar so Moring's `z^bl_l` and Φ-max constraints can attach to the same
partition at run time.

Tests: `tests/test_marshfield_blocks.py` covers each invariant in a small
fixture plus a region_000 acceptance run.

## Validation Metrics

Minimum validation:

- Count by switch role and placement sub-rule.
- Baseline connected components versus Switch-Bounded Load Blocks.
- Distribution of block bus count, load `kW/kvar`, phase count, and service
  priority weight.
- Fraction of blocks with at least one boundary switch.
- Tie count by feeder pair.
- Number of candidate switches rejected by source-stub, exportability, phase,
  and block-quality filters.
- SSAP objective value by feeder and selected switch budget by feeder.
- Confirmation that flood exposure was not used in placement scoring.
- Confirmation that critical-load-boundary switches are not synthesized until
  `critical_facilities.parquet` and `critical_load_assignments.parquet`
  validate.

Later validation:

- Grid-forming DER reachability by block.
- PowerModelsONM parsing and switch recognition.
- RPOP feasibility on a small set of event scenarios.
- NRELDynaGrid-style real-time OPF checks inside selected DNMG connected
  components.
- SFINCS/ERAD stress tests on asset availability after switch placement.

## Citeable Components

Network construction:

- SHIFT/NREL-shift provides a citeable path for synthetic distribution feeder
  model generation from open geospatial data and export through GDM/DiTTo.
  Repository citation: `https://github.com/NLR-Distribution-Suite/shift`.

Switch placement and planning:

- Usberti, Vizcaino González, Silva de Assis, and Cavellucci (2025) provide
  the active algorithmic method: a polynomial-time exact dynamic-programming
  algorithm for SSAP on radial networks with an additive reliability objective.
  Flood-RM uses this as the switch-placement basis for the NLR SHIFT radial
  feeder graph.
  DOI: `https://doi.org/10.1016/j.epsr.2025.112016`.
  Local reference: `docs/power/reference/A polynomial-time exact algorithm for the sectionalizing switch allocation.pdf`.
- Levitin, Mazal-Tov, and Elmakis (1995) introduced the classic genetic
  algorithm formulation for optimal sectionalizing in radial distribution
  systems with alternative supply.
  DOI: `https://doi.org/10.1016/0378-7796(95)01002-5`.
- Billinton and Jonnavithula (1996) is a classic IEEE reference for optimal
  switching-device placement in radial distribution systems using outage,
  maintenance, and investment costs.
  DOI: `https://doi.org/10.1109/61.517529`.
- Abiri-Jahromi, Fotuhi-Firuzabad, Parvania, and Mosleh (2012) formulate
  sectionalizing switch placement as a distribution-automation planning
  problem balancing customer outage cost with capital, installation, and O&M
  switch costs.
  DOI: `https://doi.org/10.1109/TPWRD.2011.2171060`.
- Galias (2019) provides the accepted deterministic tree-structure algorithmic
  basis for optimal switch placement in radial distribution networks.
  DOI: `https://doi.org/10.1109/TPWRS.2019.2909836`.
- IEEE 1806-2021 provides industry-facing guidance for reliability-based
  placement of distribution switching and overcurrent protection equipment up
  to 38 kV: `https://standards.ieee.org/standard/1806-2021.html`.
- Fayyazi, Azad-Farsani, and Haghighi (2024) provide resilience-oriented
  sectionalizing and tie-switch siting using stochastic fault scenarios, EENS,
  SAIDI, and load-priority distinctions.
  DOI: `https://doi.org/10.1016/j.ress.2023.109919`.
- Barani et al. (2019) and Arefifar et al. (2012/2013) are supply-sufficient
  microgrid partitioning/design references for later DER-aware block
  validation. They are not the adopted switch-placement method.

DNMG/RPOP operation:

- Moring et al. (2025) define the block-and-switch abstraction used here:
  blocks are static when every switch is open, while connected components/DNMGs
  change as switches close.
  arXiv: `https://arxiv.org/abs/2504.15084`.
- PowerModelsONM.jl is the intended artifact consumer for DNMG restoration,
  switch decisions, grid-forming DER selection, and load shedding:
  `https://github.com/lanl-ansi/PowerModelsONM.jl`.
- PowerModelsDistribution/OpenDSS compatibility requires synthesized switches
  to be parser-visible OpenDSS `Line` objects with `Switch=Yes`, with sidecar
  settings for dispatchability where needed.

Hazard and availability:

- ERAD provides the citeable fragility/resilience-analysis lineage for
  distribution-system hazards, including flooding. Marshfield uses ERAD-derived
  flood-depth fragility curves after switch placement, not as a switch-siting
  rule.
  NREL software page:
  `https://www.nlr.gov/research/software/erad--energy-resiliency-analysis-tool-for-distribution-system`.

Real-time operation:

- NRELDynaGrid-style model-free real-time OPF is the intended later layer for
  adjusting DER injections inside selected DNMG connected components between
  slower RPOP reconfiguration periods:
  `https://github.com/NatLabRockies/NRELDynaGrid/tree/main`.

## Non-Goals

- Do not claim the synthesized switches are real Eversource switches.
- Do not optimize switch placement for FEMA flood zones, SFINCS depth, or ERAD
  failure probability.
- Do not require load time series before switch siting.
- Do not copy SMART-DS load profiles into Marshfield as source data.
- Do not treat Fuse Proxy rows as Controllable Switches.
- Do not use individual buildings as the first RPOP control units.
