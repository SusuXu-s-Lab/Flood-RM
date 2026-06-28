# Flood-RM Flood-Resilience Scenario Ensemble

This context names the flood-scenario concepts used to build repeatable SFINCS truth data and flood-resilience evaluation layers across Marshfield and SMART-DS study locations.

## Working agreements

- Conversation decisions made after older docs take precedence when they conflict with historical notes or plans.
- Follow existing file style as closely as possible; prefer small surgical edits over broad rewrites.
- Keep code and docs concise enough to navigate; avoid long-winded wrappers when a direct module is clear.
- Treat the `GNS/` folder as legacy project material unless a task explicitly reopens it.
- This repo owns dataset development for both flood and grid artifacts. Downstream optimization, MPC, and control-policy training are out of scope except where their dataset inputs define required schemas or quality checks.
- "Sandbox" is dft/optimization language and does not appear in Flood-RM. The equivalent concept here is Grid Dataset and Grid Notebook Workflow.

## Module layout (src/)

- `src/design_events/` — flood event catalog and source collection (migrated from `dft.flood.design_events`)
- `src/sfincs_runs/` — SFINCS baseline build and scenario runs (migrated from `dft.flood.sfincs_runs`)
- `src/power/` — grid dataset build with stakeholder-facing modules: `baseline_network/` (Baseline Network), `resilience/` (critical facilities, load profiles, DER inventory, REopt, Controllable Switches, SSAP, Switch-Bounded Load Blocks), `exports/` (SMART-DS-compatible Grid Dataset artifacts, ONM/RPOP export, event windows), `impact` (fragility and flood impacts), `plotting.py` (review figures), and `audit/` (Synthetic Validation Audit). Absorbs `dft.power` and `dft.sandboxes.marshfield` (no sandboxes sub-namespace). Import notebook-facing modules, e.g. `from power.exports import export_stage_a1`.
- `src/fragility/` — flood-depth fragility curves linking flood layers to asset-state estimates (migrated from `dft.fragility`)

## Notebook Workflow layout

```
01_grid/
  01_base_network.ipynb          # SHIFT → GDM → DiTTo → Asset Registry (Baseline Network)
  02_augment_network/            # Infrastructure Steps that build the Augmented Network
    01_der_inventory.ipynb       # critical facilities, critical-load assignments, Layer 1 DER inventory
    02_load_profiles.ipynb       # critical-facility load profile assignment
    03_switch_synthesis.ipynb    # SSAP controllable-switch placement — partitions network
    04_load_blocks.ipynb         # block validation — needs switches + DER rows (Block Invariant Contract)
    05_onm_export.ipynb          # PowerModelsONM export and event window bundle
  03_audit_network.ipynb         # SMART-DS validation report card (Krishnan et al.)

02_flood/
  01_region_setup.ipynb          # Asset Registry → Study Area AOI, Grid Footprint, static intake
  02_collect_sources.ipynb       # CORA, rainfall, waves, NWM source artifacts
  03_build_event_catalog.ipynb   # Event Catalog and stress/training catalog
  04/                            # coupled Wflow-SFINCS base builds and smoke examples
  05_create_scenarios.ipynb      # scenario folders for truth-set runs
  06_evaluate.ipynb              # truth-set stats and evaluation products
```

- The two previous stubs (`00_build_grid_dataset.ipynb`, `00_validate_grid_dataset.ipynb`) are deleted; this sequence replaces them.
- `plot_network.ipynb` (dft) is removed as a standalone; its plots are absorbed into `01` and `02_augment_network/` notebooks.
- Each notebook follows `code/syntax.md`: teachable, modular, short comments, no hardcoded paths. Long notebook-embedded logic is extracted into `src/power/` for reuse.

## Grid section in `grid.yaml`

The `grid:` block declares power-grid artifact roots for one Study Location. Paths are location-relative unless absolute.

```yaml
grid:
  power_grid_root: data/power_grid                         # Grid Dataset root
  power_extent: data/power_grid/power_extent.geojson        # Marshfield Power Extent polygon
  shift_cache: data/power_grid/shift_cache                  # tiled SHIFT graphs, parcel pkl — kept to avoid full rebuild
  opendss_root: data/power_grid/derived_opendss             # DiTTo-written per-feeder OpenDSS files
  asset_registry: data/power_grid/asset_registry            # canonical buses/lines/loads/transformers/sources CSVs
  augmented_artifacts: data/power_grid/augmented            # load blocks, DER inventory, profiles, switch artifacts
  onm_export: data/power_grid/onm_export                    # PowerModelsONM export
  figures: data/power_grid/figures                          # review plots
```

- `aoi.source` reads from `grid.asset_registry` after `01_base_network.ipynb` runs; `aoi.source_project` (dft hardcoded path) is removed.
- Notebooks access these as resolved grid path objects via `load_runtime()` and `build_grid_paths()`.
- Notebook paths are intentionally not stored in `config.yaml`; the root README owns the standard dataset production order.
- `01_base_network.ipynb` must run before `01_region_setup.ipynb` (flood side). No artifact copying between repos. The Asset Registry and SMART-DS-compatible Parquet artifacts (`assets.parquet`, `control_units.parquet`) are the full flood-grid handoff contract.

## Docs layout (docs/power/)

```
docs/power/
  reference/
    moring_2025_networked_microgrids_load_uncertainty.pdf   # Block Invariant Contract, DNMG
    A polynomial-time exact algorithm for the sectionalizing switch allocation.pdf  # exact SSAP algorithm
    krishnan_smart_ds_validation.pdf                        # audit notebook quality gates
  methodology/
    rpop_ready_switch_siting.md     # SSAP placement rationale (from dft)
    der_placement_methodology.md    # DER sizing methodology (from dft)
    simulated_data_protocol.md      # simulated data schema and stages (from dft)
```

## Language

**Flood-Resilience Scenario Ensemble**:
A location-indexed dataset of event forcing, SFINCS truth runs, subgrid flood layers, and resilience-ready summary metrics.
_Avoid_: raw SFINCS output dataset, GNS training set

**Study Location**:
A named place where the ensemble can be generated with its own grid, forcing sources, and output folders.
_Avoid_: project, single Marshfield run

**Location Workspace**:
The stakeholder-facing folder for one Study Location, containing its config YAML, notebooks, local data artifacts, and run outputs.
_Avoid_: home notebook folder, project root, pipeline package

**Grid Notebook Workflow**:
The digestible notebook sequence under `01_grid/` that builds and validates Grid Dataset artifacts before flood notebooks consume them.
_Avoid_: control notebook, optimization notebook, mixed flood-grid notebook

**Flood Notebook Workflow**:
The digestible notebook sequence under `02_flood/` that defines the region, collects flood sources, builds the Event Catalog, runs SFINCS, and evaluates flood products.
_Avoid_: shared notebooks folder, grid dataset builder

**Wflow-Coupled Notebook Sections**:
Inland-specific sections inside the standard Flood Notebook Workflow that handle streamgage network review, Wflow build/readiness, and Wflow-SFINCS handoff without creating a separate notebook sequence.
_Avoid_: separate inland workflow, hidden Wflow preflight, new city-specific notebook order

**Location Configuration**:
The small `config.yaml` launch file for one Study Location. It keeps identity, CRS, flood setting, driver choices, coastal-wave selection, and `includes:` pointers to detailed YAML files.
_Avoid_: project.yaml, scattered config files

**Study Location Place Identity**:
The canonical human-readable place identity for a Study Location, stored once in `config.yaml` as fields such as `project.place_name` and `project.country`. Detail configurations may reference it when they need a public-source query string, but they own how that identity is interpreted for their domain.
_Avoid_: notebook-local place string, duplicated city name, workflow-specific identity.

**Location Detail Configuration**:
An included YAML file referenced by `config.yaml` that owns one operational slice of a Study Location definition, such as data sources, Grid Dataset settings, or SFINCS settings.
_Avoid_: notebook config, hidden defaults, generated plan file

**Model Recipe Configuration**:
A HydroMT-facing YAML recipe included by `config.yaml`, written with top-level HydroMT steps such as `setup_*`, `write_*`, or component writes such as `forcing.write` so SFINCS and Wflow model setup is readable without following inherited defaults.
_Avoid_: sparse model overlay, hidden shared model method, generated HydroMT plan

**Location Data Workspace**:
The stage-oriented data folder inside a Location Workspace, organized as static inputs, collected sources, Event Catalog artifacts, SFINCS artifacts, and evaluation outputs.
_Avoid_: component data folder, pipeline outputs folder

**Grid Dataset**:
The location-indexed power-grid dataset artifacts needed for flood exposure, asset-state sampling, telemetry simulation, and downstream resilience analysis.
_Avoid_: optimization model, control policy, private utility model

**Grid Dataset Development Boundary**:
The repo boundary that includes grid source ingestion, synthetic-grid provenance, SMART-DS-compatible artifact exports, and quality checks, but excludes optimization, MPC execution, and control-policy training.
_Avoid_: end-to-end grid optimization repo, downstream controller

**Baseline Network**:
The SHIFT → GDM → DiTTo Layer 1 physical network produced by `01_base_network.ipynb`. "SHIFT" is source provenance and tool lineage; "Baseline Network" is the workflow name for the unaugmented physical network.
_Avoid_: SHIFT control network, real utility baseline, augmented network.

**Augmented Network**:
The Baseline Network plus Stage B Synthesized Infrastructure added by the `02_augment_network/` notebooks: Controllable Switches, Switch-Bounded Load Blocks, DER inventory, load profiles, and ONM export. The Augmented Network is the dataset delivered to downstream DNMG/RPOP consumers.
_Avoid_: mutating Layer 1 truth, undocumented additions, utility asset inventory.

**Infrastructure Step**:
One auditable addition stage in `02_augment_network/` that creates a named synthesized-infrastructure artifact and plots the resulting change before the next step begins. Each step records placement rules, provenance, stable IDs, and validation checks.
_Avoid_: hidden notebook side effect, bulk unreviewed augmentation, plot-only change.

**Synthetic Physical Network**:
The Layer 1 OpenDSS circuit generated from public geographic inputs and synthetic distribution design assumptions by the SHIFT → GDM → DiTTo pipeline.
_Avoid_: SMART-DS substrate, utility feeder truth.

**Grid Source Anchor**:
A geospatial point used as the voltage-source seed for SHIFT Baseline Network generation. Source anchors are generated from configured public sources such as OSM power substations inside the Study Location Source Area, then reviewed in the Location Workspace. A location may provide a local reviewed override under its data workspace when public tags are incomplete or over-inclusive.
_Avoid_: docs-side production JSON, hidden substation list, utility-validated source inventory.

**Reviewed Source Anchor Artifact**:
The location-local production artifact, typically `data/static/grid/source_anchors.geojson`, that freezes the Grid Source Anchors used by Baseline Network generation. Public-source fetches may write candidate artifacts such as `source_anchor_candidates.geojson`, but reviewed anchors are reused by default so regenerated Grid Datasets do not drift when upstream public tags change.
_Avoid_: live-only substation fetch, unreviewed production anchors, docs-side anchor JSON.

**Source Anchor Review Gate**:
The Baseline Network build gate that writes source-anchor candidates, then stops until a reviewed Source Anchor Artifact exists unless `accept_unreviewed_source_anchors` is explicitly enabled for prototyping.
_Avoid_: silently promoted OSM anchors, hidden first-run mutation, production runs from unreviewed roots.

**Asset Registry**:
The deterministic CSV representation of buses, lines, transformers, sources, loads, load buses, and feeders extracted from the generated OpenDSS circuit. Canonical tables: `buses.csv`, `lines.csv`, `loads.csv`, `transformers.csv`, `sources.csv`, `load_buses.csv`, `feeders.csv`, `summary.json`.
_Avoid_: inferred asset inventory, synthetic DER table.

**Controllable Switch**:
A Stage B synthesized operational device with an open/closed state used for restoration, islanding, load-block boundaries, and RPOP actions. Must record placement rules and provenance; not inherited from Layer 1.
_Avoid_: fuse proxy, passive protective device, Layer 1 physical-network mutation.

**Fuse Proxy**:
A Stage A asset derived from SHIFT fuse-like line equipment in the Asset Registry. Flood-relevant for exposure, failure-state, and telemetry, but not a Controllable Switch and must not define load blocks by itself.
_Avoid_: controllable switch, synthesized switch, RPOP switch.

**Switch-Bounded Load Block**:
A static electrically connected block formed when all synthesized Controllable Switches are opened. The first controllable restoration unit for DNMG/RPOP studies. Blocks are static under a switch set; Dynamic Networked Microgrids are the connected components created when selected switches close.
_Avoid_: arbitrary load cluster, individual-building control unit, Stage A feeder unit.

**Dynamic Networked Microgrid**:
A connected component formed by closing one or more Controllable Switches between Switch-Bounded Load Blocks. A DNMG is an optimization-time topology, not a persistent asset inventory row; it must contain a valid voltage source when energized. The block partition plus switch inventory that enables DNMG operation is the dataset this repo builds.
_Avoid_: fixed microgrid boundary, utility-validated microgrid, load block.

**Block Invariant Contract**:
The hard-fail validation gate every row of `switch_bounded_load_blocks.parquet` must pass, derived from Moring et al. (2025) §II-A and §II-B. A valid block is: (B) non-empty in load, (C) confined to a single Bus Voltage Class, (D/E) records `phi_max` and the union of per-bus phase sets, (F) acyclic in the static line graph with all Controllable Switches open, (G) records Block Voltage Source Reachability.
_Avoid_: soft warnings, downstream filtering of invalid blocks, deferring topology invariants to runtime.

**RPOP-Ready Switch Siting**:
The Stage B switch-placement objective that applies exact fixed-budget SSAP to transformer-bridged NLR SHIFT feeder islands over a documented admissible ordinary-line candidate set: place Controllable Switches on the synthetic feeder graph to create useful Switch-Bounded Load Blocks for DNMG/RPOP operation. Flood exposure is excluded from placement rules and belongs to later asset-state and stress-test layers.
_Avoid_: flood-exposure placement rule, critical-load-only placement, arbitrary geometric placement.

**SSAP (Sectionalizing Switch Allocation Problem)**:
The exact fixed-budget radial optimization used as Flood-RM's deterministic switch-placement solver, following the Usberti et al. 2025 polynomial-time exact SSAP formulation. Flood-RM solves it exactly on each supplied transformer-bridged feeder island and admissible ordinary-line candidate set, then hands the resulting Switch-Bounded Load Blocks to ONM/RPOP.
_Avoid_: ad hoc switch placement, flood-weighted siting, greedy heuristic without provenance.

**Transformer-Bridged Feeder Island**:
A rooted feeder component used for SSAP and block accounting where transformer windings are included as topology bridges so service areas remain electrically connected, while only ordinary line edges are eligible for synthesized sectionalizing switches.
_Avoid_: treating transformer windings as switch candidates, physical-lines-only service fragments.

**Flood-Relevant Asset**:
An asset whose state, failure probability, or telemetry depends on spatial flood exposure or a geospatial join. Must have valid coordinates before Stage A validation passes.
_Avoid_: metadata-only row, non-spatial artifact.

**Load Match**:
The auditable service-proxy relationship that matches a public Critical Facility to a synthetic load bus or control unit in the Augmented Network.
_Avoid_: critical load assignment, utility service record, true customer account

**Topology-Only Asset**:
An asset included in the Grid Dataset for electrical topology and visualization but excluded from current flood-depth exposure and spatial-join requirements (e.g. overhead lines).
_Avoid_: omitted asset, flood-relevant asset.

**Synthetic Validation Audit**:
The report-card artifact produced by `03_audit_network.ipynb` that compares the Baseline and Augmented Network artifacts against synthetic-distribution validation criteria in Krishnan et al. Records passed, partial, and missing validation capabilities without claiming SMART-DS equivalence or utility validation.
_Avoid_: certification, SMART-DS regional validation, utility acceptance test.

**SMART-DS-Compatible Interface**:
The export layer that maps Grid Dataset assets, control units, and artifacts onto the schemas used by SMART-DS regional workflows. Compatibility is achieved through generated artifacts, not by claiming the network is a SMART-DS region.
_Avoid_: fake SMART-DS, copied SMART-DS region, direct SMART-DS equivalence.

**Location ID**:
The short Study Location identifier used as the namespace prefix for Grid Dataset artifact IDs, such as `marshfield`.
_Avoid_: sandbox id, project id, generated UUID namespace.

**Stable Grid ID**:
A deterministic namespaced identifier derived from source semantics, not table row position, used to keep regenerated Grid Dataset artifacts joinable across runs. Uses `<location>:*` namespace (e.g. `marshfield:*`).
_Avoid_: row number IDs, generated UUIDs, order-dependent IDs.

**SHIFT Construction Delta**:
The documented gap between the RNM-US paper reference build (used for SMART-DS datasets) and the Baseline Network. Gaps filled after the SHIFT build are recorded as Stage B Synthesized Infrastructure. Unfilled gaps remain explicit audit blockers.
_Avoid_: claiming RNM-US equivalence, hiding post-SHIFT augmentation, treating Stage B synthesis as utility evidence.

**Study Area AOI**:
The configured spatial footprint for a Study Location, built from repeatable source coordinates such as power-grid assets or SMART-DS bus coordinates and written as `data/static/aoi/study_area.geojson`.
_Avoid_: hand mesh, manually drawn bbox, implicit notebook footprint

**Source Area Patch**:
A named, explicit geometry addition or correction applied while resolving a public place boundary for grid source ingestion. A Source Area Patch may be a bbox or geometry file, must live in the Location Configuration or Location Data Workspace, and exists to make boundary choices reproducible.
_Avoid_: hidden notebook bbox, silent town-boundary edit, docs-side production geometry.

**Grid Footprint**:
The SFINCS/static-intake footprint resolved from the Study Area AOI and declared in `config.yaml`.
_Avoid_: old mesh region, hidden GIS sidecar

**HydroMT Data Catalog**:
The HydroMT-required external data catalog schema used by SFINCS build notebooks.
_Avoid_: main location config, project config

**Workflow Tool**:
A reusable command-line script or shell helper that supports Location Workspaces but is not owned by any one Study Location.
_Avoid_: location script, one-off helper

**Reference Study Location**:
The Study Location used to prove the global workflow before other locations are added.
_Avoid_: one-off pilot, special case

**First Wflow-Coupled Study Location**:
The first inland/fluvial Study Location used to prove the Wflow-coupled workflow before the pattern is copied to other inland locations.
_Avoid_: inland pilot, special case, throwaway location

**Reference Location Copy**:
The supported way to start a new dataset by copying the Reference Study Location workspace, renaming it, and editing the split YAML files and local source inputs to match the new place.
_Avoid_: generated workspace, Marshfield runtime special case, workflow override file

**Location-Neutral Workflow Text**:
Notebook headings, reusable module names, and root documentation language that describe the dataset production stage without baking in the Reference Study Location name.
_Avoid_: Marshfield region setup heading, Marshfield-only source module, location-specific name for current Study Location

**Location Evidence**:
Human-review material that documents why a Study Location uses particular local assumptions or static inputs, such as reports, surveys, mitigation plans, or source notes. It may keep place-specific filenames because it is evidence, not a reusable production interface.
_Avoid_: required production input, generated artifact, reusable module config

**Event Catalog**:
The event-indexed table and supporting files that describe historical, design, stochastic, and scenario-shifted flood drivers before SFINCS runs.
_Avoid_: production catalog, sampled peaks

**Event Timing Descriptor**:
An Event Catalog field that records a driver member's start, peak, end, native duration, or offset relative to the Event Reference Time. These descriptors define the physical scenario; concrete SFINCS `tstart` and `tstop` values are computed later during run staging.
_Avoid_: hard-coded model runtime, animation frame count

**Probability Catalog**:
The full weighted Event Catalog used to estimate annualized flood and power-system outcomes after response metrics are evaluated.
_Avoid_: SFINCS budget, stress-only set, all-tail sample

**Resilience Stress/Training Set**:
A consequence-enriched subset of the Event Catalog selected for high-fidelity SFINCS depth profiles, grid-impact analysis, and Dynamic Networked Microgrid contingency-operation studies. Selection is two-pass: first driver-diverse using Event Catalog descriptors, then consequence-aware after SFINCS/evaluation feedback exists. It deliberately includes benchmark driver slices, rainfall-heavy SST members, wet antecedent states, wave-sensitive or streamflow-sensitive analogs, timing edge cases, and near-threshold response cases while retaining links to Probability Catalog weights. The default high-fidelity run budget is stratified, not all-tail: target approximately 5% mild, 28% common, 28% significant, 12% rare, and 27% extreme events unless a location documents an explicit override.
_Avoid_: unbiased sample, max-depth-only set, mild-event dump

**Scenario Timing Edge Case**:
A deliberately named lead/lag or duration pattern used to diversify the Resilience Stress/Training Set without rewriting the physical timing in the Probability Catalog. Examples include rainfall-before-coastal, rainfall-coincident, rainfall-after-coastal, long-rain-before-surge, wave-heavy-at-peak, wet-soil-low-rain, fast-flood-short-warning, and slow-flood-long-duration.
_Avoid_: arbitrary augmentation, forced synchronized peaks

**Staged Stochastic Scenario Framework**:
An auditable ordering for scenario generation: source records, driver marginals, dependence-aware member construction, scenario shifts, hydrodynamic simulation, and weighted evaluation.
_Avoid_: piece-meal generator, heuristic scenario drawing

**Compound Driver Lag**:
The within-event timing offset between driver peaks (e.g. peak rainfall relative to peak NTR/surge), drawn from observed co-occurrence offsets and bounded by the co-occurrence pairing window. For the Marshfield production catalogue, the literature-supported default is a 3-day (72 h) RF/NTR pairing window: Maduwantha et al. (2026, HESS 30:401-420, Sect. 4.1 and Sect. 6) select the partner maximum within a 3 d window and report manually checking that 3 d generally captured both peaks. Shorter windows such as 24-36 h are sensitivity diagnostics only unless a location-specific analysis or cited method documents that change. Compound Driver Lag is a physical timing property of a single compound event. It is NOT the difference between the absolute calendar timestamps of two independently sampled observed analogs (a 1978 surge analog vs a 2011 rainfall analog) — that quantity spans decades and is a provenance artifact, never the compound lag.
_Avoid_: analog calendar-time delta, member_time minus event_time, unbounded lag, uncited fixed lag window

**Conditional Empirical Lag Analogue Sampling**:
The production method for assigning Compound Driver Lag after the copula has sampled the Driver Probability Indices: within the sampled Storm Type stratum when enough observations exist, draw an observed RF/NTR co-occurrence analogue by weighted k-nearest-neighbor similarity on peak NTR, peak rainfall, and season, then inherit that analogue's observed peak-RF-to-peak-NTR lag. This keeps lag empirical and conditionally tied to storm context while avoiding deterministic nearest-neighbor reuse. A fitted conditional lag distribution is a fallback only when analogue diagnostics show insufficient observed support.
_Avoid_: deterministic nearest-lag lookup, independent marginal lag draw, synthetic lag distribution as default

**Compound Dependence Diagnostic**:
A driver/flood-response association figure that must be read consistently with how the joint model is fit: storm-type-stratified (TC / nor_easter / other_non_tropical, never pooled into one cloud), probability-weighted (the tail-enriched catalogue's `probability_weight` is the headline; an unweighted scatter Kendall τ is labeled diagnostic-only), and on the same physical coastal axis the copula uses (NTR/surge, not total still water level). Diagnostic covariation only — not causal attribution.
_Avoid_: pooled cross-storm-type τ, unweighted τ on the enriched catalogue presented as climatology, mixing NTR and total water level across panels

**Benchmark Design Slice**:
A named annual-chance or return-period checkpoint used for comparison and communication, not a complete probability-weighted flood-risk estimate.
_Avoid_: full risk curve, canonical flood map

**Coastal Driver Return Period**:
The annual-exceedance return period assigned to an Event Catalog row from the fitted coastal **non-tidal residual (NTR/surge)** marginal — not the total still water level — before hydrodynamic response is simulated. NTR is computed from the total-water-level record via UTide (`coastal_components`); the astronomical tide and MSL are preserved unscaled in the realized boundary (ADR-0011).
_Avoid_: flood return period, map return period, total water-level marginal, tide-inclusive coastal index

**Flood Annual Exceedance Probability**:
The annual probability that a simulated flood-response metric exceeds a threshold at a location, asset, or evaluation layer after hydrodynamic response is evaluated.
_Avoid_: catalog return period, driver severity

**Average Annual Outcome**:
The expected annual damage, disruption, exposure, or resilience metric integrated over flood-response probabilities.
_Avoid_: single-event loss, design-storm impact

**Joint Probability Flood-Frequency Model**:
A probability model that estimates joint annual-exceedance behavior from driver marginals, dependence, event rates, and hydrodynamic response.
_Avoid_: Event Catalog, stochastic scenario ensemble

**Driver Probability Index**:
The scalar per-driver summary (coastal **non-tidal residual / surge peak** — stored as `coastal_water_level`/`coastal_peak_m` but tide-removed, distinct from the realized total `coastal_absolute_peak_m`; rainfall event depth; streamflow peak) used as the random variable in marginal and joint (copula) probability models. It indexes — but does not replace — a Field-Preserving Realization, so reducing a driver to its index for statistics never collapses the physical forcing.
_Avoid_: spatially-uniform rainfall value, scalar applied directly as forcing, lumped block-mean driver

**Antecedent Moisture State**:
The catchment wetness state at storm onset used as an infiltration/runoff initial condition, sampled as a conditional event attribute rather than as a symmetric copula peak driver; coastal Study Locations use the NWM soil-moisture artifact for the first implementation.
_Avoid_: soil-moisture peak, independent soil-moisture driver, unconditional AMC bin

**Antecedent Moisture Conditioning Month**:
The calendar month used to sample Antecedent Moisture State, taken from rainfall onset when available, then Event Reference Time, then coastal analog peak time.
_Avoid_: always using coastal peak month, row-order month, unrecorded seasonal bin

**Storm Type**:
An optional coastal-event conditioning attribute with canonical labels `TC`, `nor_easter`, `other_non_tropical`, and `unresolved`.
_Avoid_: untracked hurricane label, assuming every coastal storm is tropical, collapsing nor'easters into an undifferentiated non-TC tail, ad hoc storm-class strings

**Field-Preserving Realization**:
The full spatio-temporal driver forcing attached to an Event Catalog row by conditioning on its Driver Probability Index — e.g. a transposed AORC SST rainfall field staged as SFINCS `netampr`, an analog hydrograph template, or a coastal water-level/wave analog window. The two-layer separation (scalar index for probability, field for physics) is how the workflow keeps best-practice marginal/joint statistics without discarding rainfall structure.
_Avoid_: scalar depth applied uniformly, lumped block-mean forcing, regenerating fields from summaries only

**Event Catalog Plan**:
The resolved input/output paths, configured forcing-member tables, pairing
policies, required forcings, and wave-analog rule used to build one Event
Catalog for a Study Location and scenario.
_Avoid_: catalog notebook setup, implicit recipe inputs

**Tail-Enriched Design Ensemble**:
A design ensemble that intentionally samples more rare flood-driver combinations than their natural probability mass would produce, while recording event weights for probability-weighted analysis.
_Avoid_: unbiased random sample, worst-case set

**Sampling Weight**:
The event-level multiplier that records how body/tail sampling was enriched before full probability weighting is evaluated.
_Avoid_: flood probability weight, return period, severity score

**Probability Weight**:
The event weight used to estimate Flood Annual Exceedance Probability or Average Annual Outcome from a Tail-Enriched Design Ensemble.
_Avoid_: sampling budget, severity score

**Forcing Pairing Policy**:
The declared rule used to combine coastal, rainfall, streamflow, and soil-moisture members into one Event Catalog row. The production policy is `copula_joint` (the Copula-Joint Dependence Model, ADR-0011); the seasonal-window / antecedent-lag heuristics are retained as named sensitivity baselines.
_Avoid_: random join, hidden dependence assumption

**Copula-Joint Dependence Model**:
The production compound-driver dependence model (ADR-0011): a vine copula (pyvinecopulib) fit semiparametrically on the Two-Sided Conditional POT Co-occurrence Sample over a config-declared Driver Dependence Vector, sampled with joint-tail enrichment, AND-labeled, and realized through Field-Preserving Realizations. Replaces heuristic pairing as the `copula_joint` Forcing Pairing Policy while keeping marginals, hydrograph templates, and SST fields unchanged.
_Avoid_: heuristic seasonal pairing as production, scalar-only rainfall, copula fit on independent member libraries

**Driver Dependence Vector**:
The config-declared set of Driver Probability Indices the copula is fit over, keyed to flood setting: coastal = {coastal water level, rainfall}; inland Wflow-coupled = {rainfall}, with Antecedent Moisture State as a conditioning attribute and discharge as a Wflow-derived response rather than a copula dimension (ADR-0011, operationalized by ADR-0016). The observed streamflow POT at the Primary Reference Gage is retained as a validation/calibration and Same-Frequency Amplification (K) anchor inland, not a copula dimension — so one rainfall-runoff process is never assigned two independent return periods.
_Avoid_: putting antecedent moisture or Wflow-derived discharge in the copula, `driver_vector = "streamflow, rainfall"` inland, hard-coded per-location vector

**Two-Sided Conditional POT Co-occurrence Sample**:
The paired-observation table that fits the copula: condition on each driver in turn, take its declustered peaks-over-threshold, and pair the concurrent extreme of the other drivers within a documented window (Wahl 2015; Jane 2020; Maduwantha 2026). For the coastal RF/NTR catalogue, use the Maduwantha et al. (2026) 3 d pairing window as the production default unless an explicit, cited, location-specific method supersedes it. Assembled from real aligned driver records, not the independent member libraries.
_Avoid_: independent member libraries, single-driver POT, simultaneous-peak assumption, undocumented window tuning

**AND Joint Return Period**:
The pre-SFINCS severity label from the AND joint-exceedance scenario (all drivers exceed): `T = 1 / (rate * P(all exceed))`, with the rigorous `probability_weight` taken from the fitted joint density (Moftakhari 2019; Salvadori 2016). Marginal Driver Return Periods are kept as communication anchors.
_Avoid_: univariate driver RP as the joint label, OR-scenario RP without saying so

**Operationally Severe Plausible Dependence**:
A stress-set-only forcing-pairing policy that preserves seasonal plausibility while intentionally pairing high coastal drivers with severe rainfall timing and wet antecedent states for contingency-operation testing.
_Avoid_: joint probability model, unbiased dependence model, random worst case

**Historical Compound Pair**:
A stress-set row whose coastal-water-level analog and rainfall member are paired by their original observed timing rather than by seasonal permutation.
_Avoid_: synthetic co-occurrence, arbitrary rainfall pairing

**Historical Coastal Analog**:
An observed storm-window reference that supplies temporally coherent coastal water-level and wave forcing for a synthetic or shifted Event Catalog row.
_Avoid_: random wave match, independent wave sample

**Source Artifact**:
A small manifest and the files produced by a source collection step before those files are assembled into an Event Catalog.
_Avoid_: API dump, raw pull

**Source Collection Plan**:
The ordered list of configured source collection steps, each with its validated
time window and source-specific settings for one Study Location.
_Avoid_: notebook checklist, ad hoc pull list

**Study Location Reproduction Plan**:
The root README instruction book for reproducing a Flood-Resilience Scenario Ensemble in one Study Location, from location definition through Grid Dataset, Static Intake, Source Artifacts, Event Catalog, Truth Set staging/runs, and Evaluation Layers. It is maintained as project documentation, not generated per Location Workspace.
_Avoid_: Marshfield checklist, notebook order notes, source collection plan, generated production plan

**Location Definition Interface**:
The public `define_location` interface that reads and validates a Study Location's YAML files without generating project documentation or running dataset stages.
_Avoid_: create location, generated location plan, hidden setup wizard

**Location Definition**:
The object returned by `define_location`, containing the Study Location name, root, merged configuration, resolved paths, grid settings, data-source settings, SFINCS settings, validation results, and dataset production stage order.
_Avoid_: raw config dict, notebook runtime, generated plan file

**Static Intake**:
The AOI/grid-footprint-driven preparation of static inputs for one Study Location, including terrain, land cover, soils, coastline, and HydroMT catalog inputs.
_Avoid_: base data notebook, SFINCS build

**Direct AORC SST Collection**:
Repo-owned stochastic-storm-transposition rainfall collection that scans AORC
precipitation over a documented transposition region and writes rainfall member
tables and selected storm-window artifacts. Dewberry StormHub is a reference
implementation for SST concepts, not a runtime package for this workflow.
_Avoid_: StormHub runtime dependency, custom rainfall blob, manifest-only adapter

**Hydrodynamic Truth Set**:
The SFINCS inputs, manifests, outputs, and subgrid flood layers generated from an Event Catalog.
_Avoid_: model cache, raw runs

**Standardized Training View**:
A fixed-shape or fixed-horizon derivative of variable-length truth data for machine-learning or dynamic-microgrid consumers. It may use padding masks, peak-centered slices, recovery-tail slices, or resampled summaries, but it does not redefine the physical Event Catalog scenario.
_Avoid_: canonical event duration, truth run

**Baseline Build Plan**:
The selected hydrodynamic build path for one Study Location, including regular-grid, wave-coupled, or Wflow-coupled mode, notebook template, model root, grid footprint, and required Source Artifacts.
_Avoid_: notebook choice, model setup checklist

**SFINCS Structure Layer**:
A static line-feature layer applied during baseline SFINCS model construction to represent coastal flood-control geometry that the grid or DEM may under-resolve, such as overtoppable seawall/road-crest weirs and thin-dam jetty or closure controls. Drainage structures are excluded unless invert elevation, hydraulic dimensions or rating, and upstream-to-downstream orientation are confirmed.
_Avoid_: drainage guess, local stormwater network, surveyed design set

**Single-Use-Case Test Plan**:
The isolated one-event SFINCS check run before cluster submission, preferring a
recent actual historical extreme event when the Event Catalog contains one.
_Avoid_: quick run, random smoke event

**Subgrid Flood Layer**:
A raster flood-depth or water-surface product derived from SFINCS subgrid output mechanisms for asset-level evaluation.
_Avoid_: raw `zs - zb` map, bathtub downscaling

**Evaluation Layer**:
A table or raster that summarizes flood exposure metrics for downstream resilience analysis.
_Avoid_: power-system dataset

**Boundary CORA Node**:
The CORA wet grid node selected near the SFINCS boundary centroid and used as the local water-level forcing source.
_Avoid_: Boston gauge, buoy average

**Boundary Water-Level Source**:
The source used to build coastal water-level forcing for a Study Location.
_Avoid_: boundary node, gauge, CORA when the source may vary

**Boundary CORA Water-Level Trend**:
The secular water-level trend estimated from annual means of the Boundary CORA Node time series.
_Avoid_: gauge trend, peak trend

**MSL-Shift Scenario**:
A scenario that translates stationary surge hydrographs by a fixed mean-sea-level offset.
_Avoid_: climate-change surge projection

**NOAA Offset Provenance**:
The source metadata that ties an MSL-shift offset to a published scenario family, projection year, baseline year, location basis, and access date.
_Avoid_: hard-coded offset

**Residual Trend**:
A statistically detectable monotonic trend remaining in the selected peak series after detrending.
_Avoid_: failed output

**Threshold/Model Sensitivity**:
A review table showing how return-period magnitudes change when POT threshold, peak spacing, and tail distribution choice change.
_Avoid_: production catalog

**Stochastic Storm Transposition Event**:
A precipitation event generated by moving an observed storm over a plausible transposition domain for a Study Location. For rainfall diversity, one physical storm episode may carry multi-duration descriptors such as max 6-hour, 24-hour, 72-hour, and total-event depth rather than being duplicated into separate fixed-duration catalog rows.
_Avoid_: synthetic rainfall blob, fixed-duration rainfall row

**Forcing Support Window**:
The modeled time span over which all dynamic drivers for one Event Catalog row are staged together for SFINCS. It may be longer than any single driver member because it can include spin-up, peak alignment, rainfall duration, wave/water-level analog duration, and drain-down.
_Avoid_: fixed 72-hour event, animation length, coastal peak duration

**Event Reference Time**:
The canonical time origin used to align driver members inside one variable-length physical scenario. For coastal wave-coupled scenarios it defaults to the coastal/wave analog peak. For **inland Wflow-coupled** scenarios (rainfall is the single driver and discharge is the Wflow response, ADR-0016) it defaults to the **true rainfall peak** (`rainfall_peak_time`) — there is no independent streamflow member at catalog-build time, so the rainfall peak, not storm-window onset, centers the Wflow forcing window (`resolve_event_window`). Only **external-boundary fluvial** settings that realize streamflow as an upstream boundary hydrograph (large-mainstem, ADR-0017) use the streamflow peak. The rainfall member keeps `rainfall_member_time = storm_start` (the AORC event-window lookup key) and records its `rainfall_peak_offset_hours` relative to this reference instead of being forced to peak at the reference hour.
_Avoid_: rainfall start, storm-window onset as inland reference, SFINCS `tref`, shared forced peak time

**Inland Storm Timing Descriptor**:
A response-side timing attribute derived for inland Wflow-coupled Event Catalog rows from the collected `rainfall_peak_time`, never a copula dimension (discharge is the Wflow response, ADR-0016/ADR-0019). The family: the **storm loading pattern** (normalized peak position within the storm window, front/center/back-loaded; Huff terciles), the **observed catchment basin lag** (observed USGS peak minus rainfall peak at the Primary Reference Gage, the observed reference for the Wflow Readiness peak-timing check and the observed side of the Soil-Moisture Modulation Diagnostic), and **timing seasonality** (peak month/hour, convective vs frontal/tropical). It diversifies the Resilience Stress/Training Set and supplies data-driven Scenario Timing Edge Case tags without altering design probability.
_Avoid_: inland compound lag as a copula axis, RF-vs-discharge lag as an independent driver, forcing rainfall and Wflow discharge to share one sampled peak time

**Antecedent Soil-Moisture State**:
The pre-event soil wetness condition estimated from NWM retrospective variables and attached to an Event Catalog member.
_Avoid_: AMC bin

**Streamflow Driver**:
A river or channel inflow time series sourced from USGS gages, NWM streamflow, or another documented source for a Study Location.
_Avoid_: discharge hack

**USGS Streamgage Record**:
An observed discharge time series and station metadata from a selected USGS streamgage used to build streamflow Source Artifacts.
_Avoid_: gage API dump, arbitrary river point

**USGS Streamgage Network**:
A hydrologically selected set of USGS Streamgage Records with documented roles such as frequency analysis, Wflow calibration, validation, and SFINCS inflow handoff.
_Avoid_: nearest-gage list, single-outlet shortcut, API search result

**Active USGS Streamgage Candidate**:
A USGS streamgage returned by source discovery with a currently active discharge record at the time the candidate artifact is built.
_Avoid_: discontinued gage, historical-only station, inactive nearest gage

**Reviewed Streamgage Network Artifact**:
The location-local production artifact that freezes the selected USGS Streamgage Network, preserving active-record eligibility, excluding inactive gages, and recording hydrologic roles and reviewer rationale for reproducible Wflow-coupled runs.
_Avoid_: live API result, notebook-local gage list, unreviewed candidate set

**Reviewed Streamgage Schema**:
The minimum fields required in a Reviewed Streamgage Network Artifact: `site_no`, `site_name`, `status`, geometry, `drainage_area_sqmi`, `period_start`, `period_end`, `record_years`, `completeness_score`, `roles`, `frequency_basis`, `wflow_submodel_id`, `sfincs_domain_id`, `sfincs_handoff_id`, `review_status`, and `review_notes`.
_Avoid_: raw NWIS schema, notebook-only columns, undocumented handoff fields

**Streamgage Role List**:
The canonical `roles` list in the Reviewed Streamgage Schema used to mark one or more hydrologic/modeling roles for a gage.
_Avoid_: role-specific production booleans, single role column, hidden notebook filter

**Streamflow Frequency Basis Group**:
A named group identifier in the Reviewed Streamgage Schema that links one or more Frequency-Basis Streamgages to the POT fit/provenance used by coherent streamflow members.
_Avoid_: frequency boolean, unnamed POT fit, single-gage-only basis

**Streamgage Review Status**:
The controlled review state for a streamgage row: `candidate`, `accepted`, `accepted_with_warning`, or `rejected`.
_Avoid_: free-text approval, hidden reject flag, unreviewed production row

**USGS Streamgage Discovery Slice**:
The first Greensboro Wflow-coupled implementation slice, which discovers active USGS streamgage candidates and produces the reviewed streamgage-network artifact before Wflow or SFINCS domain planning proceeds.
_Avoid_: Wflow-first setup, SFINCS-first setup, manual gage list

**Streamflow Driver Return Period**:
The annual-exceedance return period assigned to an Event Catalog row from the fitted streamflow marginal before hydrodynamic response is simulated.
_Avoid_: Wflow return period, flood-map return period

**Frequency-Basis Streamgage**:
An active USGS Streamgage Record in the reviewed network whose POT fit defines Streamflow Driver Return Periods for one streamflow member or SFINCS inflow boundary.
_Avoid_: single primary gage, calibration-only gage, nearest active gage

**Coherent Streamgage Network Event**:
A streamflow Event Catalog member that preserves one physically related historical or design event across multiple active streamgages, with one Event Reference Time and gage-specific hydrographs and return-period descriptors.
_Avoid_: independently mixed gage peaks, arbitrary multi-inflow design row

**Streamgage Network Analog**:
A historical Coherent Streamgage Network Event selected as the timing and shape template for a design streamflow member.
_Avoid_: independent hydrograph template, synthetic shape, unrelated gage event

**Scaled Streamgage Network Design Event**:
A design streamflow member built by scaling a Streamgage Network Analog toward target gage-specific return-period magnitudes while preserving network timing, rise, recession, and inter-gage relationships within documented bounds. **Superseded for inland Wflow-coupled Study Locations by ADR-0016**: inland catalog rows carry no independent streamflow design member; discharge is Wflow-generated from rainfall + Antecedent Moisture State and frequency-corrected by the single-K Same-Frequency Amplification. The term is retained only for settings that realize streamflow as an external upstream boundary hydrograph (coastal / large-mainstem).
_Avoid_: fully synthetic hydrograph, independently sampled gage design peaks, inland scaled-gage injection into Wflow

**Primary Reference Gage**:
The single reviewed USGS streamgage per Wflow Submodel, on the highest-stream-order / primary inflow river, whose POT defines the streamflow frequency target and at whose cell the Same-Frequency Amplification (K) is computed against simulated Wflow discharge. Selected from Wflow-native geometry — the largest-`uparea` Stream-Boundary Handoff Source's upstream reach, with `rivwth`/`rivdph` from `setup_rivers` — using NHDPlus `StreamOrde` only as a review cross-check (USA-First Wflow Source Strategy). Must be in the submodel's Wflow gauge-output set. For `greensboro_rural` this is `02094500` (Reedy Fork).
_Avoid_: cross-basin max-envelope (`basis_site_no = "network_max"`), nearest gage, tributary gage of lower stream order, a gage Wflow does not output

**Same-Frequency Amplification**:
The single per-event multiplicative factor `K = streamflow_target_RP(Primary Reference Gage) / Q_wflow_peak(Primary Reference Gage)` applied uniformly to every Wflow-generated handoff hydrograph and timestep before SFINCS staging (Kim et al. 2023; Maduwantha et al. 2026, Eq. 2, applied at the model-output stage). It is an empirical bias/frequency correction anchoring Wflow output to observed streamflow frequency, never a prescribed boundary forcing; a configured `k_band` flags implausible K as a poorly matched analog rainfall to fix upstream, not a scaling to lean on.
_Avoid_: per-crossing independent scaling, scaling injected gage forcing, treating K as the primary discharge signal

**Wflow Readiness Validation**:
A lightweight event-replay gate that checks Wflow outlet placement, peak-flow magnitude, peak timing, and hydrograph volume against the active streamgage network before Wflow-coupled outputs are treated as production flood layers.
_Avoid_: full calibration, blind default model, production-by-smoke-test

**Wflow Readiness Report**:
The reviewed pass/warn/fail artifact that summarizes Wflow Readiness Validation diagnostics without enforcing universal numeric thresholds across Study Locations.
_Avoid_: hard-coded calibration score, informal notebook note, hidden reviewer judgment

**Wflow Hydrologic Bridge**:
The Wflow model run that transforms event meteorology and hydrologic states into routed inflow hydrographs for SFINCS while preserving the Event Catalog's streamflow frequency provenance.
_Avoid_: frequency model, design-event generator, standalone watershed study

**USA-First Wflow Source Strategy**:
The source-selection rule for inland US Study Locations that prefers local US hydrography and soil evidence over global defaults: a local DEM-derived HydroMT LDD basemap for Wflow `setup_basemaps` and `setup_rivers`, reviewed USGS streamgages for gauges and handoff, and SSURGO-derived local soil evidence. NHDPlus/3DHP vectors are optional QA/review evidence, not Wflow build geometry. Global sources such as MERIT Hydro and SoilGrids are fallbacks or prototype sources, not the Greensboro production default.
_Avoid_: blind MERIT Hydro default, raw NHDPlus-as-basemap, global-first US workflow

**SSURGO-Derived Wflow Soil Evidence**:
The local soil evidence available for US inland Study Locations from SSURGO mapunit polygons and attributes, including hydrologic soil group and saturated hydraulic conductivity rasters. These rasters already cover SFINCS infiltration inputs, but Wflow SBM still requires reviewed Wflow-ready soil parameter maps derived from or augmented by that local evidence.
_Avoid_: assuming SoilGrids is required by default, treating HSG alone as complete Wflow SBM parameterization

**Wflow-Coupled Truth Set**:
A Hydrodynamic Truth Set whose SFINCS discharge forcing is staged from Wflow outputs for an inland or fluvial Study Location.
_Avoid_: coastal wave-coupled run, Wflow-only result, direct-gage SFINCS shortcut

**Dual Fluvial-Pluvial Forcing**:
The inland Wflow-coupled forcing pattern where SFINCS receives Wflow-routed river inflows and direct rainfall over the SFINCS grid for the same Event Catalog row.
_Avoid_: inflow-only inland run, rainfall-only floodplain run, Wflow replacing SFINCS rainfall

**Coherent Inland Forcing Row**:
An inland Event Catalog row that mirrors the Marshfield compound-forcing pattern by carrying one explicit streamflow/network event, one rainfall member, optional antecedent soil-moisture member, driver timing descriptors, and provenance before Wflow and SFINCS staging.
_Avoid_: Wflow-local event recipe, separate rainfall for Wflow and SFINCS, coastal-only catalog logic

**Inland Rainfall Pairing Priority**:
The rule that pairs rainfall to streamflow by first using same-storm rainfall for historical streamgage-network analogs, then falling back to seasonal-window permutation when coherent rainfall is unavailable.
_Avoid_: independent rainfall shuffle, forced same-peak rainfall, arbitrary design storm

**Inland Antecedent Moisture Pairing**:
The rule that pairs antecedent soil moisture to the selected rainfall member when rainfall is coherent, and otherwise to the Dominant Streamgage Network Peak.
_Avoid_: fixed soil state, rainfall-independent antecedent shuffle, global AMC bin

**Dominant Streamgage Network Peak**:
The selected peak time of the coherent streamgage-network event used as the Event Reference Time for inland and fluvial Event Catalog rows.
_Avoid_: rainfall peak, arbitrary model start, forced synchronized peak

**Grid Footprint**:
The bounding-box-derived SFINCS domain used to align flood layers with downstream asset grids.
_Avoid_: mesh footprint

**SMART-DS Evaluation Footprint**:
The asset and evaluation coverage area derived from SMART-DS-compatible grid artifacts. It defines the minimum hydraulic coverage for SFINCS, usually through a coverage bounding box, but it does not define the upstream Wflow watershed.
_Avoid_: Wflow watershed, hydrologic basin, upstream routing domain

**Hydrologic Modeling Domain**:
The combined modeling extent required to represent upstream Wflow routing plus SFINCS hydraulic coverage around the SMART-DS Evaluation Footprint. It is not one polygon type: Wflow may be a larger outlet-delineated watershed/subbasin domain, while SFINCS may be a smaller coverage box.
_Avoid_: SMART-DS-only watershed, asset-only domain, clipped upstream basin

**Reviewed Wflow Subbasin Region**:
A HydroMT-Wflow `subbasin` region derived from a reviewed SFINCS handoff streamgage outlet, with reviewed drainage area used as `uparea` snapping evidence. The build lets HydroMT-Wflow derive the basin from the local LDD basemap rather than forcing NHDPlus/3DHP polygon bounds.
_Avoid_: using NHDPlus polygons as the HydroMT mask by default, bbox-only Wflow build, arbitrary hand-drawn basin

**Wflow-Native River Geometry**:
The `rivers` geometry written by HydroMT-Wflow `setup_rivers` to `staticgeoms/rivers.geojson`, derived from the Wflow DEM/LDD/upstream-area basemap and river mask. It is the preferred river-line source for Wflow-SFINCS boundary handoffs.
_Avoid_: separately digitized river line, NHD-as-authoritative handoff line, nearest-boundary shortcut

**Stream-Boundary Handoff Source**:
The SFINCS discharge source point placed where Wflow-Native River Geometry enters a SFINCS Coverage Box. Reviewed streamgages define the upstream Wflow outlet/submodel; they are not forcibly snapped to the SFINCS boundary. Missing, ambiguous, or downstream-only intersections are review-required and may imply a larger Wflow domain or revised coverage box.
_Avoid_: reviewed gage as SFINCS source, nearest point on bbox, hard-stuck boundary condition

**Wflow Submodel**:
One Wflow model partition inside a Study Location's Hydrologic Modeling Domain, used when active streamgage networks, watersheds, or SFINCS handoff reaches are hydrologically separate.
_Avoid_: one model per city assumption, arbitrary tile, SMART-DS feeder region

**SFINCS Coverage Box**:
The rectangular hydraulic modeling region around a SMART-DS Evaluation Footprint or evaluation component. Its boundary is where Wflow discharge enters through Stream-Boundary Handoff Sources; it is not expected to be a perfect watershed.
_Avoid_: watershed polygon, reviewed-gage-count primary region, hydrologic basin mask

**SFINCS Domain**:
One hydrodynamic model domain inside a Study Location, usually represented for Austin/Greensboro by a SFINCS Coverage Box around the SMART-DS Evaluation Footprint. It simulates local hydraulic response and receives Wflow discharge at stream-boundary source points.
_Avoid_: Wflow watershed, outlet-delineated basin, reviewed-gage-count region

**Wflow-SFINCS Domain Set**:
The configured collection of Wflow Submodels, SFINCS Coverage Boxes, and Stream-Boundary Handoff Sources used by one Study Location under one Event Catalog.
_Avoid_: separate Study Locations, unrelated model folders, hidden per-domain recipe

**Greensboro Wflow-SFINCS Domain Set**:
The first inland setup now modeling only the selected `greensboro_east` SMART-DS SFINCS Coverage Box, with a larger encompassing Wflow HUC basin retaining all accepted gages inside that basin.
_Avoid_: importing `greensboro_west` into the SFINCS domain set, clipping Wflow to the hydraulic coverage box, dropping accepted basin gages

**Austin Wflow-SFINCS Domain Set**:
The inland setup currently modeling only the selected `austin_p4u` SMART-DS SFINCS Coverage Box. Austin Wflow may be one large upstream hydrologic model or multiple reviewed submodels later, but it remains review-required until handoff outlet gages and stream-boundary crossings are selected.
_Avoid_: silently restoring all Austin subregions, one giant Austin SFINCS box, fake bbox watershed, unreviewed Wflow handoff

**Domain-Specific Run Manifest**:
The per-Wflow-submodel or per-SFINCS-domain manifest that records model-specific forcing files, timing, and outputs while referencing the shared Event Catalog row.
_Avoid_: per-domain event catalog, detached run recipe, duplicate event id

**Multi-Domain Evaluation Merge**:
The rule that combines multiple SFINCS Domain outputs into one asset-level Evaluation Layer by taking each SMART-DS asset's maximum modeled flood depth across domains while preserving source-domain provenance.
_Avoid_: separate domain exposure tables, averaging overlapping depths, losing domain provenance

**Infiltration Treatment**:
The SFINCS rainfall-loss configuration used for a run, such as no infiltration, curve-number infiltration, or curve-number recovery.
_Avoid_: soil model

**Soil-Moisture Infiltration Role**:
The location-type-specific way **Antecedent Moisture State** affects flood response. Coastal Marshfield uses NWM `SOILSAT_TOP` to restage SFINCS CN-with-recovery initial storage (`sfincs.seff`) for rain-on-grid runs, while the static SFINCS base carries SSURGO HSG/Ksat rasters (`sfincs.smax`, `sfincs.ks`). Inland Austin and Greensboro use antecedent moisture through the Wflow-coupled rainfall-runoff state and then hand routed discharge to SFINCS; their SFINCS infiltration is still enabled for local direct rainfall, but it is not the upstream hydrologic model.
_Avoid_: treating SFINCS infiltration as Wflow, treating soil moisture as an independent peak driver, assuming a coastal soil-moisture member changes runoff without `sfincs.seff` staging

**Soil-Moisture Modulation Diagnostic**:
The validation figure that shows **Antecedent Moisture State** is excluded from the copula as a *driver* (ADR-0011) yet its physical effect is reproduced on the *response* side. It separates two relationships the raw scatter conflates: driver-side dependence (does moisture co-vary with rainfall depth) versus response-side modulation (does moisture bend the rainfall→runoff transform). The figure targets the latter: event precipitation (x) against the location-typed rainfall→runoff response (y), colored by Antecedent Moisture State. The response engine is location-typed — Wflow peak discharge at the **Primary Reference Gage** inland, **`sfincs.seff`**-driven rain-on-grid infiltration-excess coastal. Inland (gaged) locations overlay realized design events on an observed backdrop as **Wflow Readiness Validation** evidence; coastal Marshfield plots catalog events on its modeled seff response surface and inherits the inland evidence rather than claiming an observed coastal discharge cloud it cannot produce.
_Avoid_: soil moisture as a copula axis, observed-cloud claim for an ungaged coastal location, rainfall-only response scatter without antecedent coloring, reading driver-side dependence into a response-modulation figure

**Coastal Wave Coupling**:
A property of a coastal Study Location declared in its `config.yaml` that selects the wave build path: `true` routes through the quadtree + SnapWave + IG-wave notebook, while `false` leaves the Study Location on a non-wave hydrodynamic path. True for Marshfield and San Francisco; not the controlling build decision for inland Wflow-coupled Austin and Greensboro.
_Avoid_: wave flag, ad hoc switch

**Wave-Coupled Truth Set**:
A Hydrodynamic Truth Set built on a quadtree SFINCS grid with the internal SnapWave solver enabled and infragravity-wave physics turned on (mt_Faber branch); produces wave-augmented water levels, runup-gauge time series, and subgrid flood layers in one run.
_Avoid_: wave run, SnapWave dataset

**SnapWave Boundary Forcing**:
The paired hourly time series of significant wave height, peak wave period, mean wave direction, and directional spreading at SnapWave boundary support points, sourced from ERA5 ocean wave variables on the Copernicus CDS. Infragravity components are derived inside SnapWave from these incident-wave inputs via the Herbers (1994) formulation, not from separate input files.
_Avoid_: WW3 hindcast, synthetic spectra

**Runup Gauge Transect**:
A computational output transect declared in the SFINCS `rugfile` along which SFINCS reports the per-timestep max wet point against a configurable depth threshold; transect outputs land in `sfincs_his.nc` alongside observation points. Not a physical instrument and not validation data.
_Avoid_: tide gauge, observation gauge

## Relationships

- A **Flood-Resilience Scenario Ensemble** contains one or more **Study Locations**.
- A **Study Location** has one **Location Workspace** under `locations/<study_location>/`.
- A **Location Workspace** has one **Location Configuration** named `config.yaml`.
- A **Location Configuration** may include **Location Detail Configuration**
  files so the launch file stays readable for new Study Locations.
- Every configured Study Location keeps `config.yaml` as a small launch file:
  identity, flood setting, event drivers, user-facing scenarios, and `includes:`
  pointers only.
- SMART-DS Study Locations use `smartds.yaml` for minimally required region data
  pointers, AOI source settings, and flood-evaluation footprint inputs.
- Marshfield keeps `grid.yaml`, shaped like `smartds.yaml`, because its Grid
  Dataset is a generated SHIFT/GDM/DiTTo grid rather than a SMART-DS region.
- Every Study Location's hydrodynamic detail file is named `sfincs.yaml` and is
  wired through `includes.sfincs`. It owns SFINCS settings, forcing, parameters,
  and native HydroMT-SFINCS build/update recipes under `hydromt:`.
- Inland Wflow-coupled Study Locations use `wflow.yaml` through
  `includes.wflow`. It owns Wflow settings, forcing, parameters, domain/handoff
  configuration, source collection details, and native HydroMT-Wflow `steps:`
  recipes under `hydromt:`.
- Coastal wave-coupled Study Locations use `snapwave.yaml` through
  `includes.snapwave`. It owns SnapWave settings, wave forcing, runup-gauge
  parameters, and the wave-coupled native HydroMT-SFINCS recipes.
- `config.resolved.yaml` is not checked into Location Workspaces. It may be
  generated on demand as a disposable merged view by
  `python tests/flood_rm/show_resolved_config.py`.
- Generated native HydroMT recipe files may be materialized under
  `data/<model>/config/` for CLI use; source-of-truth model recipes live in the
  model YAML files.
- Monolithic legacy `config.yaml` loading is allowed only as a short migration
  bridge while the Reference Study Location and notebook callers move to split
  YAML files. It should not remain a parallel configuration style.
- Notebook paths are not part of **Location Configuration**. The root README
  instruction book owns the standard dataset production order; notebooks follow
  conventional names under each **Location Workspace**.
- Notebook headings and reusable source modules use **Location-Neutral Workflow
  Text** unless they are explicitly documenting Reference Study Location
  evidence.
- A **Location Workspace** separates the **Grid Notebook Workflow** from the **Flood Notebook Workflow** so grid dataset development can be reviewed before flood products are generated.
- Austin and Greensboro keep the standard **Flood Notebook Workflow** shape and
  add **Wflow-Coupled Notebook Sections** inside the relevant stages rather
  than starting a separate inland notebook sequence.
- A **Location Workspace** has one **Location Data Workspace** under `locations/<study_location>/data/`.
- A **Location Data Workspace** may contain both flood artifacts and **Grid Dataset** artifacts when they are needed to develop flood-grid datasets.
- The **Grid Dataset Development Boundary** allows reusable grid-data ingestion and quality checks, but leaves optimization and control-policy execution to downstream consumers.
- A **Study Area AOI** is generated from configured source data and becomes the **Grid Footprint** used by static intake and SFINCS setup.
- A **HydroMT Data Catalog** may live inside the **Location Data Workspace**, but it does not replace the **Location Configuration**.
- A **Workflow Tool** lives in the owning module or cluster folder because it can support multiple **Study Locations**.
- Marshfield is the **Reference Study Location** for the global workflow.
- Greensboro is the **First Wflow-Coupled Study Location**; Austin follows the
  same inland/Wflow workflow with six SMART-DS subregion SFINCS Coverage Boxes
  and a reviewed-network gate before Wflow production.
- A new dataset may start from a **Reference Location Copy** of Marshfield,
  then replace location-specific YAML values and local source files.
- Reference Study Location evidence may keep Marshfield-specific filenames, but
  reusable interfaces must accept the current Study Location name from the
  **Location Definition Interface** rather than importing hardcoded Marshfield
  paths.
- **Location Evidence** stays under the Location Workspace documentation tree;
  production inputs stay under generic stage-oriented data paths and are
  referenced by **Location Detail Configuration** files.
- A **Study Location** has one **Grid Footprint** and one or more **Boundary Water-Level Sources**.
- A **Location Definition Interface** is the first code entrypoint for reading
  a new **Study Location** and validating its split YAML files.
- A **Location Definition** exposes dataset production stages, not notebook
  paths, because the root README instruction book owns the human workflow.
- A **Source Artifact** records the source, Study Location, time window, produced files, and provenance metadata for a collection step.
- **Static Intake** prepares non-event-specific inputs before dynamic source collection.
- A **Study Location Reproduction Plan** is the launch point for a new
  **Study Location** and orders the **Grid Notebook Workflow**,
  **Static Intake**, **Source Collection Plan**, **Event Catalog Plan**,
  truth-set staging/runs, and evaluation handoff.
- A **Source Collection Plan** determines which **Source Artifacts** should be
  collected for a configured **Study Location**.
- A **Staged Stochastic Scenario Framework** orders source records, driver
  marginals, dependence-aware member construction, scenario shifts,
  hydrodynamic simulation, and weighted evaluation before a
  **Flood-Resilience Scenario Ensemble** is presented.
- An **Event Catalog Plan** resolves the inputs and **Forcing Pairing Policy**
  used to build an **Event Catalog**.
- An **Event Catalog** defines the drivers used to build a **Hydrodynamic Truth Set**.
- An **Event Catalog** stores **Event Timing Descriptors** for physical driver
  timing; SFINCS run staging computes concrete model `tstart`/`tstop` from the
  **Forcing Support Window** and writes the exact values to the run manifest.
- A **Probability Catalog** keeps the weighted ensemble intact for annualized
  summaries such as expected outage hours, load unserved, islanding operations,
  and annual damage.
- A **Resilience Stress/Training Set** spends scarce high-fidelity simulations
  around flood-to-grid transition states and rare compound drivers; it is not a
  replacement for the weighted **Probability Catalog**. Selection starts with
  a Marshfield-like stratified severity budget (about 5% mild, 28% common,
  28% significant, 12% rare, 27% extreme), plus mandatory benchmark and
  compound-driver cases, before SFINCS and becomes consequence-aware as
  evaluated response data accumulates.
- **Scenario Timing Edge Cases** guide selection into the
  **Resilience Stress/Training Set** while the **Probability Catalog** preserves
  source-record lead/lag timing from its **Forcing Pairing Policy**.
- **Operationally Severe Plausible Dependence** belongs to the
  **Resilience Stress/Training Set**, not the **Probability Catalog**.
- A **Historical Compound Pair** keeps observed coastal-rainfall timing inside
  the **Resilience Stress/Training Set** so several high-magnitude real events
  remain available as anchoring cases.
- A **Benchmark Design Slice** may be selected from the **Event Catalog** for
  communication using a **Coastal Driver Return Period** before SFINCS and a
  **Flood Annual Exceedance Probability** after SFINCS.
- A **Joint Probability Flood-Frequency Model** is realized by the
  **Copula-Joint Dependence Model** (ADR-0011): a vine copula over the
  **Driver Dependence Vector**, fit on the **Two-Sided Conditional POT
  Co-occurrence Sample**, AND-labeled, and joint-tail-enriched.
- A **Copula-Joint Dependence Model** produces a **Tail-Enriched Design
  Ensemble** whose **Probability Weight** comes from the fitted joint density
  (so the **Probability Catalog** stays mild-dominated by mass while the joint
  tail is full enough for the **Resilience Stress/Training Set** budget), and
  attaches a **Field-Preserving Realization** to every sampled
  **Driver Probability Index**.
- An **Antecedent Moisture State** is sampled as an event conditioning
  attribute for infiltration/runoff response, not as a member of the
  **Driver Dependence Vector**.
- **Antecedent Moisture State** conditions on event month and **Storm Type**
  when storm-type tagging exists; until then, month-only conditioning is the
  documented fallback.
- **Antecedent Moisture Conditioning Month** uses rainfall onset first, then
  **Event Reference Time**, then coastal analog peak time.
- Coastal **Antecedent Moisture State** uses the NWM soil-moisture source
  artifact until a location documents a model-state alternative.
- **Storm Type** is initially reserved as an Event Catalog column only; plots and
  conditioning use it after storm-identification artifacts exist.
- A **Tail-Enriched Design Ensemble** is represented in the **Event Catalog**
  with a **Sampling Weight** on each event; **Probability Weight** is required
  before reporting **Average Annual Outcome** summaries.
- A **Forcing Pairing Policy** must be recorded when independent or dependent driver members are combined; `copula_joint` is the production policy and the seasonal/antecedent heuristics remain as named sensitivity baselines.
- A **Forcing Support Window** is event-specific and should be derived from the
  paired driver members plus explicit spin-up and drain-down padding, bounded by
  configured minimum and maximum run lengths, not hard-coded as the duration of
  one rainfall or coastal source.
- An **Event Reference Time** aligns variable-length driver members without
  erasing their native lead/lag structure.
- A **Historical Coastal Analog** supplies both **Boundary Water-Level Source**
  forcing and **SnapWave Boundary Forcing** for a non-historical wave-coupled
  Event Catalog row.
- A **Direct AORC SST Collection** produces **Stochastic Storm Transposition Events** for Marshfield.
- A **Hydrodynamic Truth Set** produces **Subgrid Flood Layers** and **Evaluation Layers**.
- A **Standardized Training View** may be derived from a variable-length
  **Hydrodynamic Truth Set**, but fixed-size ML tensors do not define the
  physical scenario duration.
- A **Baseline Build Plan** selects whether a Study Location produces a regular
  **Hydrodynamic Truth Set**, **Wave-Coupled Truth Set**, or
  **Wflow-Coupled Truth Set**.
- A **SFINCS Structure Layer** belongs to baseline model construction before
  event forcing is staged so all Event Catalog rows share the same static flood
  controls.
- A **Single-Use-Case Test Plan** validates one Event Catalog row through SFINCS
  scenario creation, local run staging, and stats before cluster batch launch.
- A **Boundary CORA Node** provides the hourly water-level record for the **Boundary CORA Water-Level Trend**.
- A **Boundary CORA Water-Level Trend** removes secular MSL drift before fitting historical peaks.
- An **MSL-Shift Scenario** adds a fixed offset after historical peak fitting and event-member construction.
- **NOAA Offset Provenance** documents the source of each **MSL-Shift Scenario** offset.
- A **Residual Trend** is disclosed in diagnostics; it does not block research or training artifact generation.
- **Threshold/Model Sensitivity** supports review of the production catalog but does not define production event members.
- A **Stochastic Storm Transposition Event** contributes precipitation forcing and can be paired with an **Antecedent Soil-Moisture State**.
- A **Streamflow Driver** is optional per **Study Location** and must document whether USGS gage data or NWM streamflow was used.
- For inland and fluvial **Study Locations**, source discovery filters to
  **Active USGS Streamgage Candidates** before human review selects a
  **Reviewed Streamgage Network Artifact**.
- A **Reviewed Streamgage Network Artifact** must satisfy the **Reviewed
  Streamgage Schema** so frequency analysis, Wflow submodel planning, SFINCS
  handoff, and reviewer decisions share one production interface.
- The **Streamgage Role List** is the canonical role representation; boolean
  columns such as `is_frequency_basis` may be derived later but are not the
  production contract.
- `frequency_basis` in the **Reviewed Streamgage Schema** stores a
  **Streamflow Frequency Basis Group** identifier, not a boolean.
- `review_status` in the **Reviewed Streamgage Schema** stores a
  **Streamgage Review Status** value.
- Greensboro's first Wflow-coupled implementation slice is the **USGS
  Streamgage Discovery Slice**; Wflow domain-set planning follows the reviewed
  active streamgage network.
- A **USGS Streamgage Network** supplies observed discharge records for
  frequency analysis, Wflow calibration/validation, and handoff checks;
  configured **Frequency-Basis Streamgages** supply **Streamflow Driver Return
  Periods** while a **Wflow Hydrologic Bridge** supplies routed discharge
  forcing for the **Wflow-Coupled Truth Set**.
- A **USGS Streamgage Network** may contain multiple **Frequency-Basis
  Streamgages**; each streamflow Event Catalog member must identify the gage
  whose POT fit produced its **Streamflow Driver Return Period**.
- When multiple **Frequency-Basis Streamgages** contribute to the same
  streamflow member, the member must be a **Coherent Streamgage Network Event**
  rather than an independent mix of unrelated gage-specific peaks.
- A **Wflow-Coupled Truth Set** for Austin and Greensboro uses **Dual
  Fluvial-Pluvial Forcing**: Wflow-routed inflows for channels/boundaries and
  direct SFINCS rainfall for local floodplain and urban flooding.
- Austin and Greensboro should follow the Marshfield Event Catalog and staging
  shape as **Coherent Inland Forcing Rows**: the Event Catalog owns forcing
  provenance, pairing, and timing; Wflow and SFINCS both consume the same
  rainfall member and timing from that row.
- The **SMART-DS Evaluation Footprint** defines minimum asset/evaluation
  coverage for Austin and Greensboro, while the **Hydrologic Modeling Domain**
  may extend beyond it to include upstream Wflow watersheds, routing reaches,
  and the streams that cross into the SFINCS coverage box.
- A Study Location may contain multiple **Wflow Submodels** when the
  **Hydrologic Modeling Domain** crosses separate active streamgage networks,
  watersheds, or SFINCS handoff systems.
- A Study Location may contain one or more **SFINCS Coverage Boxes**. Together
  with the configured **Wflow Submodels** and **Stream-Boundary Handoff
  Sources**, these form a **Wflow-SFINCS Domain Set** that still belongs to one
  Study Location, Event Catalog, and SMART-DS Evaluation Footprint.
- Greensboro currently selects only the `greensboro_east` SFINCS Coverage Box;
  Austin currently has six SMART-DS AUS subregion boxes and should not collapse
  them into one hydraulic domain unless a later reviewed domain decision says
  so.
- One **Event Catalog** row defines an event across the whole
  **Wflow-SFINCS Domain Set**. Individual submodels/domains write
  **Domain-Specific Run Manifests**, but they do not create independent event
  catalogs.
- Multi-domain flood outputs are reduced to one asset-level **Evaluation
  Layer** through **Multi-Domain Evaluation Merge**: use maximum depth per
  SMART-DS asset across domains and retain source-domain provenance and
  overlap diagnostics.
- For **Coherent Inland Forcing Rows**, the **Event Reference Time** is the
  **Dominant Streamgage Network Peak**; rainfall and other drivers preserve
  their paired offsets relative to that streamflow reference time.
- **Coherent Inland Forcing Rows** use the **Inland Rainfall Pairing Priority**:
  prefer rainfall from the same historical streamgage-network storm window, and
  use seasonal-window permutation only when coherent rainfall is unavailable.
- **Coherent Inland Forcing Rows** use **Inland Antecedent Moisture Pairing**:
  pair soil moisture relative to the selected rainfall member when rainfall is
  coherent, and relative to the **Dominant Streamgage Network Peak** when
  rainfall coherence is unavailable.
- Design streamflow members beyond the observed record should be **Scaled
  Streamgage Network Design Events** built from **Streamgage Network Analogs**,
  not fully synthetic hydrographs.
- Default HydroMT-Wflow may be used for prototypes and first-pass scenario
  construction, but production Austin/Greensboro flood layers require
  **Wflow Readiness Validation** summarized in a **Wflow Readiness Report**.
  Full calibration is a future upgrade unless readiness validation exposes
  material bias.
- US inland Wflow builds should follow the **USA-First Wflow Source Strategy**:
  reviewed USGS streamgages, a local DEM-derived HydroMT LDD basemap, and
  local SSURGO soil evidence take precedence over global HydroMT defaults.
  NHDPlus/3DHP vectors may support QA, but should not be hard-coded into the
  Wflow build path.
- SFINCS discharge source points for Austin and Greensboro should follow the
  **Stream-Boundary Handoff Source** convention: derive river lines from
  HydroMT-Wflow `setup_rivers` output first, then fall back to NHDPlus/3DHP
  review geometry only when Wflow-native river geometry is unavailable.
- **SSURGO-Derived Wflow Soil Evidence** is already available for Greensboro
  SFINCS infiltration as HSG and Ksat rasters; Wflow SBM soil maps should be
  derived or augmented from those local soils where possible, with SoilGrids
  treated as a documented fallback.
- An **Infiltration Treatment** belongs to a SFINCS run manifest, not to a **Study Location** globally.
- **Coastal Wave Coupling** is declared per **Study Location** and selects the build notebook; `true` produces a **Wave-Coupled Truth Set**, `false` produces a regular-grid **Hydrodynamic Truth Set**.
- A **Wave-Coupled Truth Set** is forced at the wave boundary by **SnapWave Boundary Forcing** and produces **Runup Gauge Transect** outputs in addition to **Subgrid Flood Layers**.
- A **Forcing Pairing Policy** for any non-historical event (Stochastic Storm Transposition Event, MSL-Shift Scenario, or other synthetic catalog member) must use the **same historical analog** to supply both the water-level forcing and the **SnapWave Boundary Forcing**. Mixing wave-and-no-wave events in the same catalog is disallowed because it biases peak water-level magnitudes.

## Example dialogue

> **Dev:** "Is the deliverable just the SFINCS run folders?"
> **Domain expert:** "No — the deliverable is the **Flood-Resilience Scenario Ensemble**. The run folders are provenance for the **Hydrodynamic Truth Set**, while the reusable products are the **Subgrid Flood Layers** and **Evaluation Layers**."

> **Dev:** "Can I call a catalog row a 100-year flood?"
> **Domain expert:** "Not before SFINCS: the catalog row has a **Coastal Driver Return Period**; the **Flood Annual Exceedance Probability** is evaluated from the simulated flood response."

## Flagged ambiguities

- "CORA averages across sparse buoys" was used to describe the current artifact. Resolved: this pipeline currently uses one **Boundary CORA Node** selected near the SFINCS boundary centroid.
- "Power system dataset" is outside this repo for now. Resolved: this repo should produce **Evaluation Layers** that downstream power-system work can consume.
- "Mesh" was used for older GNS work. Resolved: new SFINCS setup should start from a generated **Study Area AOI** and **Grid Footprint** unless a task explicitly needs mesh-derived artifacts.
- "Marshfield-only" was used to describe the current code. Resolved: Marshfield is the **Reference Study Location**, while source modules should stay global.
- "Move notebooks by location" was resolved in two phases: additive first, then promoted to **Location Workspace** ownership once Marshfield became the reference end-to-end workflow.
- "Reusable scripts" was resolved as **Workflow Tools** rather than Study Location files unless the script is truly location-specific.
- "Generated artifacts during reorganization" was resolved as a full move into the **Location Workspace** now; avoid compatibility shims that read from the old top-level workflow folders.
- "Location data folder names" was resolved as stage-oriented names: `static/`, `sources/`, `event_catalog/`, `sfincs/`, and `evaluation/`.
- "Split project configs" was resolved as one **Location Configuration** per **Study Location**. A **HydroMT Data Catalog** remains separate only because HydroMT requires its own schema.
- "Where Wflow config lives" was resolved as `wflow.yaml`, a separate
  **Location Detail Configuration**, rather than nesting Wflow setup under
  `sfincs.yaml`.
- "`build_base_data.ipynb` as Marshfield-only static prep" was revised: `01_region_setup.ipynb` now defines the **Study Area AOI** from configured source data; static-source collection and SFINCS build preparation remain downstream steps.
- "Per-location config" was resolved as one `locations/<study_location>/config.yaml` file first; split out HydroMT data catalogs only when the build stage needs the official HydroMT schema.
- "Other Study Location stubs" are allowed as empty placeholder folders, but
  only folders with `config.yaml` are configured **Study Locations**.
- "API data pull" was resolved as direct package use where practical. CORA can
  use the current simple collector; Marshfield rainfall uses **Direct AORC SST
  Collection**. Dewberry StormHub remains a methods reference only.
- "JPM", "vine copula", and "stochastic scenario ensemble" were being used as
  interchangeable options. Resolved: the current deliverable is a
  **Flood-Resilience Scenario Ensemble** built with a **Staged Stochastic
  Scenario Framework**; a **Joint Probability Flood-Frequency Model** is a
  separately named future upgrade.
- "Return period" was being used for both input drivers and flood impacts.
  Resolved: catalog rows carry **Coastal Driver Return Period** before SFINCS;
  post-SFINCS flood maps, exposure, damage, and resilience summaries use
  **Flood Annual Exceedance Probability** and **Average Annual Outcome**.
- "Use Wflow to set up floods" was ambiguous between making Wflow the
  frequency model and using it as a routing bridge. Resolved: for Austin and
  Greensboro, USGS streamgage POT analysis should provide the
  **Streamflow Driver Return Period**, while the **Wflow Hydrologic Bridge**
  stages physically routed SFINCS inflow hydrographs.
- "One streamgage or many" was resolved as a **USGS Streamgage Network**:
  configure multiple hydrologically relevant gages with explicit roles rather
  than assuming the nearest or single primary outlet gage is sufficient.
- "Automatic or manual streamgage selection" was resolved as hybrid:
  automatically discover only **Active USGS Streamgage Candidates**, score them
  by hydrologic suitability, then require a **Reviewed Streamgage Network
  Artifact** before production Wflow-coupled runs.
- "Use discontinued streamgages as backup evidence" was rejected. Resolved:
  inactive or historical-only gages are excluded from the production
  **USGS Streamgage Network**, not retained as validation or evidence rows for
  Austin/Greensboro Wflow-coupled runs.
- "Single primary frequency gage" was rejected. Resolved: Austin and
  Greensboro may use multiple **Frequency-Basis Streamgages**, with each
  streamflow member preserving its gage-specific POT fit and return-period
  provenance.
- "Independent multi-gage design peaks" were rejected. Resolved: multi-gage
  streamflow members are **Coherent Streamgage Network Events** with shared
  physical timing and gage-specific severity descriptors.
- "Synthetic design hydrographs" were deferred. Resolved: first Austin and
  Greensboro design events should scale **Streamgage Network Analogs** toward
  target gage-specific return-period magnitudes while preserving coherent
  network timing and shape.
- "Require full Wflow calibration before production" was rejected as too much
  first-pass overhead. Resolved: uncalibrated/default HydroMT-Wflow is allowed
  for prototype runs, but production Wflow-coupled flood layers require
  **Wflow Readiness Validation** through historical event replay diagnostics.
- "Universal Wflow readiness thresholds" were deferred. Resolved: readiness
  uses quantitative diagnostics with a reviewed **Wflow Readiness Report**
  carrying pass/warn/fail judgments rather than hard-coded cross-location
  thresholds.
- "Wflow inflow only versus local rainfall" was resolved as **Dual
  Fluvial-Pluvial Forcing** for Austin and Greensboro so local pluvial flooding
  around SMART-DS assets is not lost.
- "Use Marshfield logic inland" was resolved as preserving the Marshfield
  Event Catalog/staging structure while replacing coastal water-level/SnapWave
  drivers with streamflow-network/Wflow drivers. The same rainfall member and
  timing should feed both Wflow forcing and direct SFINCS rainfall.
- "Inland event clock" was resolved as the **Dominant Streamgage Network
  Peak**, matching Marshfield's use of the coastal/wave analog peak while
  allowing rainfall to keep its paired lead/lag offset.
- "Inland rainfall pairing" was resolved by priority: use same-storm rainfall
  for historical streamgage-network analogs when available; otherwise fall back
  to Marshfield-style seasonal-window permutation.
- "Inland antecedent soil moisture timing" was resolved as rainfall-relative
  when rainfall is same-storm/coherent, with streamgage-network-peak-relative
  fallback when coherent rainfall is unavailable.
- "Soil moisture as a copula driver" was rejected. Resolved: soil moisture is
  an **Antecedent Moisture State**, an initial-condition attribute conditioned
  on storm context rather than a symmetric **Driver Probability Index**.
- "TC/non-TC conditioning" was accepted as domain language but is not yet a
  production artifact. Resolved: reserve **Storm Type** in the Event Catalog,
  use canonical labels `TC`, `nor_easter`, `other_non_tropical`, and
  `unresolved`, and use month-only Antecedent Moisture State conditioning until
  storm-identification artifacts exist.
- "Marshfield antecedent moisture source" was resolved as the NWM soil-moisture
  artifact for coastal setup; Wflow spinup state is not required for Marshfield's
  first antecedent-moisture implementation.
- "SMART-DS as Wflow watershed" was rejected. Resolved: SMART-DS defines the
  **SMART-DS Evaluation Footprint** and minimum SFINCS flood-layer coverage,
  while upstream Wflow watersheds are allowed to expand into the **Hydrologic
  Modeling Domain** needed for physically coherent routed inflow.
- "One Wflow model per city" was rejected. Resolved: Austin and Greensboro may
  use multiple **Wflow Submodels** when hydrologic partitions require separate
  watershed/routing setups.
- "SFINCS domain must be a watershed" was rejected. Resolved: Austin and
  Greensboro use **SFINCS Coverage Boxes** around SMART-DS evaluation regions,
  with discharge boundary conditions placed at **Stream-Boundary Handoff
  Sources** where Wflow-native streams enter the box. Multiple SFINCS domains
  remain allowed only when evaluation coverage or hydraulic review requires
  separate boxes.
- "Per-domain Event Catalogs" were rejected. Resolved: one **Event Catalog**
  row runs across the whole **Wflow-SFINCS Domain Set**, with domain-specific
  manifests preserving local forcing/output details.
- "Separate exposure outputs per SFINCS domain" were rejected as the primary
  evaluation handoff. Resolved: use **Multi-Domain Evaluation Merge** to produce
  one asset-level layer using max depth per SMART-DS asset while preserving
  source-domain diagnostics.
- "Separate inland notebook workflow" was rejected for the first implementation.
  Resolved: Austin and Greensboro should remain structurally parallel to
  Marshfield's **Flood Notebook Workflow**, with **Wflow-Coupled Notebook
  Sections** added where inland hydrology needs extra review/readiness work.
- "Austin or Greensboro first" was resolved as Greensboro first for the
  implementation slice. Austin now follows the same coverage-box convention
  with the selected `austin_p4u` SFINCS domain, while stream-boundary handoff and
  Wflow production remain review-required until reviewed outlets exist.
- "USGS discovery or Wflow domain planning first" was resolved as the
  **USGS Streamgage Discovery Slice** first because the reviewed active
  streamgage network determines downstream Wflow submodels, SFINCS handoff
  points, readiness validation, and streamflow frequency provenance.
- "Wflow bbox or subbasin watershed setup" was resolved for Greensboro:
  production Wflow builds use HydroMT-Wflow `subbasin` regions derived from
  reviewed SFINCS handoff streamgage outlets plus reviewed drainage-area
  `uparea` evidence. HydroMT-Wflow derives the watershed from the local LDD
  basemap; NHDPlus/3DHP-derived subbasin fabric is optional QA evidence only
  and must not contribute `bounds` or `geom` masks to the default build path.
  A bbox region is only a prototype placeholder until the **Reviewed Streamgage
  Network Artifact** supplies accepted handoff outlets and Wflow Submodel IDs.
- "Nearest SFINCS bbox snap" was rejected. Resolved: SFINCS discharge source
  locations are **Stream-Boundary Handoff Sources** derived from
  **Wflow-Native River Geometry**. If a reviewed outlet's matched Wflow river
  does not have a defensible upstream crossing of the SFINCS coverage boundary,
  the result is review-required rather than hard-forced to the nearest edge.
- "Wflow copied-demo resolution or local 60 m hydrography resolution" was
  resolved for Greensboro: production Wflow builds should use a
  HydroMT-compatible local US hydrography basemap at about 1/1800 degree
  resolution, rather than the 30 arc-second demo resolution copied from the
  larger coupling example. The source basemap and Wflow `setup_basemaps.res`
  must stay aligned; do not downscale a coarser hydrography basemap in the
  Wflow build.
- "MERIT Hydro or US hydrography for Greensboro" was resolved as the
  **USA-First Wflow Source Strategy**. Production Greensboro Wflow builds
  should not default to MERIT Hydro; they should register a
  local DEM-derived HydroMT LDD basemap and let HydroMT-Wflow derive river
  cells from `setup_rivers`, using global hydrography only as a documented
  fallback or prototype path.
- "Are soils already covered?" was resolved as yes for SFINCS infiltration:
  Greensboro already has SSURGO HSG and Ksat rasters. Wflow SBM still needs
  reviewed Wflow-ready soil parameter maps, preferably derived or augmented
  from the SSURGO evidence, while SoilGrids remains a fallback rather than the
  default production source.
- "Reviewed streamgage artifact schema" was resolved as a compact production
  schema carrying site identity, active-record status, drainage area,
  period/completeness, hydrologic roles, model/domain handoff IDs, and review
  status/notes.
- "Streamgage role fields" were resolved as one canonical **Streamgage Role
  List** rather than production booleans for each possible role.
- "`frequency_basis` boolean" was rejected. Resolved: `frequency_basis` is a
  **Streamflow Frequency Basis Group** string so multiple gages can share a
  coherent frequency/provenance grouping.
- "Free-text streamgage review status" was rejected. Resolved:
  **Streamgage Review Status** is one of `candidate`, `accepted`,
  `accepted_with_warning`, or `rejected`.
