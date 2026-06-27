# Marshfield Sandbox Simulated Data Protocol

This protocol defines how the Marshfield Control Sandbox will generate SMART-DS-compatible simulated data for method development. It is a research protocol, not a claim that the sandbox is a SMART-DS region or a utility-grade Eversource model.

## Research Position

**Objective:** produce reproducible simulated data that lets DFT telemetry, belief, failure-injection, and MPC tooling be exercised on the Marshfield sandbox.

**Primary constraint:** the source substrate remains a synthetic OpenDSS network built from public geographic inputs and documented synthetic distribution design assumptions. Compatibility is achieved through shared interfaces and artifact schemas.

**Research standard:** every generated table must have a schema version, input hashes, code version, random seed policy, units, coordinate reference, source provenance, and validation report. This follows FAIR data-management principles and reproducible scientific-computing practice [R1, R2].

## Staged Order

### Stage 0: SMART-DS Static Parity

Before extending operational simulations, the Marshfield sandbox should first match the SMART-DS `base_timeseries/opendss_no_loadshapes` artifact shape: static topology, static loads, bus coordinates, line codes, transformers, lines, loads, and optional regulator/capacitor files. This gate establishes geometry, topology, asset inventory, and static served-load parity before adding full OpenDSS loadshape/profile time series.

**First TDD tracer bullet:** implement a public static-parity validator before adding more generator features:

```text
validate_static_smart_ds_parity(
  marshfield_export,
  reference_profile="greensboro_2016_no_loadshapes",
)
```

The validator passes only when:

- required OpenDSS files exist (`Master.dss`, `Buscoords.dss`, `Lines.dss`, `LineCodes.dss`, `Transformers.dss`, `Loads.dss`)
- `Master.dss` redirect paths resolve
- all buses referenced by lines, loads, and transformers have coordinates or an explicit exemption
- loads have finite `kW` and `kvar`
- line and transformer counts are nonzero
- `assets.parquet` and `control_units.parquet` exist
- stable IDs use the `marshfield:*` namespace
- the validation report records the comparison target and schema version

### Stage A1: Static Data Spine

Generate these static products first:

1. `assets.parquet`
2. `control_units.parquet`

**Gate A1:** no event-conditioned asset-state or telemetry generation begins until Stage A1 passes static referential-integrity, coordinate, schema, provenance, and format checks.

### Stage A2: Event-Conditioned Data Spine

Generate these products only after Gate A1 passes:

1. `asset_states.parquet`
2. `telemetry_observations.parquet`

**Gate A2:** no Stage B work begins until Stage A2 passes event, Monte Carlo, fragility, telemetry-target, unit, provenance, and stochastic-reproducibility checks.

### Stage B: Operational State

Generate DER/controllable-switch/load state only after Stage A is valid:

1. `der_states.parquet`
2. `controllable_switches.parquet`
3. `switch_states.parquet`
4. `load_states.parquet`
5. `load_profile_assignments.parquet`
6. `critical_facilities.parquet`
7. `critical_load_assignments.parquet`
8. OpenDSS `LoadShapes.dss` plus profile CSVs for full `opendss` parity
9. updated `telemetry_observations.parquet` rows for synthesized device telemetry
10. `switch_bounded_load_blocks` rows appended to `control_units.parquet` after
    Controllable Switches exist

**Gate B:** no Stage C work begins until every DER, controllable switch, and controllable load references an existing asset, bus, or Control Unit, and until observability-tier assumptions are recorded.

### Stage C: MPC Trajectories

Generate `mpc_trajectories.parquet` only after Stage B is valid. MPC rows must reference Stage A and Stage B identifiers rather than embedding disconnected copies of network state.

**Gate C:** trajectory generation must record optimizer stack, solver version, objective terms, horizon, time step, infeasibility handling, and OpenDSS/PyDSS validation policy.

## Stable Identifier Scheme

All Stage A identifiers must be deterministic, namespaced, and derived from stable source semantics. Never use row numbers, table sort order, Python object IDs, random UUIDs, or timestamps as persistent identifiers.

Use the sandbox namespace:

```text
marshfield
```

Marshfield artifacts must use the canonical `marshfield:*` namespace.

Required ID patterns:

```text
municipality_id = marshfield:municipality:<municipality_slug>
tile_id         = marshfield:tile:<municipality_slug>:<tile_token>
feeder_id       = marshfield:feeder:<source_anchor>:<tile_token>
asset_id        = marshfield:asset:<source_asset_table>:<source_asset_name>
control_unit_id = marshfield:control_unit:<control_unit_type>:<source_token>
event_id        = <external event ID already used by the SURGE/SFINCS event manifest>
controllable_switch_id = marshfield:asset:controllable_switches:<switch_token>
run_id          = marshfield:run:<protocol_version>:<event_id>:seed_<root_seed>:<short_input_hash>
```

Examples:

```text
marshfield:municipality:marshfield
marshfield:tile:marshfield:t003
marshfield:feeder:marshfield:t003_f017
marshfield:asset:transformers:017_tx_abc12345
marshfield:asset:load_buses:024_bus_9f20
marshfield:control_unit:feeder:017
marshfield:control_unit:source_area:substation_5
marshfield:run:v0.1:riley_90m:seed_20260509:4f3a91c2
```

Normalization rules:

- Lowercase ASCII only.
- Replace whitespace and non-alphanumeric separators with `_`.
- Preserve existing OpenDSS/source names after normalization when they are stable.
- Prefix every asset ID with the source asset table to avoid collisions between a line, load, transformer, source, or bus with the same local name.
- If a source record has a stable upstream UUID, store it in `source_uuid`; do not use it alone as the human-facing `asset_id`.
- If a source name changes because the upstream converter changes, record the old-to-new mapping in the run manifest before treating it as the same asset.

## Artifact Format Policy

Parquet is the canonical format for every generated research table in Stage A, Stage B, and Stage C. CSV exports may be generated for inspection, notebook debugging, and lightweight review, but CSV files are never authoritative inputs to downstream simulation, validation, telemetry synthesis, or MPC.

Required format rules:

- Canonical outputs use `.parquet` filenames and typed schemas.
- Optional debug exports use `.debug.csv` filenames beside the canonical Parquet artifact.
- Debug exports must be regenerated from Parquet, not from independent code paths.
- Debug exports must not be read by production or validation scripts except for explicit debug-only checks.
- The run manifest records the SHA-256 hash of each canonical Parquet artifact and, when present, the hash of each derived debug CSV.
- If Parquet and CSV disagree, Parquet wins and the CSV must be regenerated.

Example layout:

```text
artifacts/sandboxes/marshfield/data/derived/smart_ds_compat/
  assets.parquet
  assets.debug.csv
  control_units.parquet
  control_units.debug.csv
  asset_states.parquet
  telemetry_observations.parquet
  run_manifest.json
  run_manifest_stage_a2.json
  validation_report.json
  validation_report_stage_a2.json
```

## Minimum Schemas

### `assets.parquet`

Required fields:

- `sandbox_id`
- `asset_id`
- `asset_type`
- `source_asset_table`
- `source_asset_name`
- `source_uuid`
- `feeder_id`
- `bus`
- `phases`
- `lon`
- `lat`
- `coordinate_status`
- `coordinate_source`
- `is_flood_relevant`
- `spatial_join_required`
- `coordinate_exemption_reason`
- `rated_kv`
- `rated_kva`
- `source_provenance`
- `schema_version`

### `control_units.parquet`

Required fields:

- `sandbox_id`
- `control_unit_id`
- `control_unit_type`
- `control_unit_stage`
- `source_feeder_id`
- `parent_control_unit_id`
- `member_asset_ids`
- `source_ids`
- `boundary_bus_ids`
- `served_load_kw`
- `critical_load_weight`
- `der_capacity_kw`
- `der_capacity_kwh`
- `candidate_status`
- `candidate_basis`
- `source_provenance`
- `schema_version`

### `asset_states.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `event_id` | string | stable event identifier |
| `mc_draw` | int32 | zero-based Monte Carlo draw index |
| `timestamp` | timestamp, UTC | event sample time |
| `asset_id` | string | must reference Stage A1 `assets.parquet` |
| `state` | string | `available`, `failed` |
| `failure_probability` | float64 | probability in `[0, 1]` |
| `sampled_depth_m` | float64 | meters above local asset exposure datum |
| `failure_model` | string | model family, initially `erad_flood_depth_lognormal` |
| `failure_model_version` | string | ERAD/mapping version token |
| `rng_seed` | int64 | row-level deterministic seed |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_a2_asset_states.v0.1` |

### `telemetry_observations.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `event_id` | string | stable event identifier |
| `mc_draw` | int32 | zero-based Monte Carlo draw index |
| `timestamp_observed` | timestamp, UTC | time represented by the observation |
| `timestamp_delivered` | timestamp, UTC | observed time plus delivery delay |
| `target_type` | string | `asset`, `control_unit` |
| `target_id` | string | must reference Stage A1 asset or Control Unit IDs |
| `observation_source` | string | synthetic source family |
| `measured_quantity` | string | e.g. `asset_failure_state`, `failed_asset_fraction` |
| `value` | float64 | quantity value |
| `unit` | string | explicit unit, e.g. `binary`, `fraction` |
| `noise_model` | string | initial value `none` |
| `delay_model` | string | initial value `deterministic_hash_0_to_15min` |
| `observability_tier` | string | initial values `tier_1_scada_oms`, `tier_2_feeder_summary` |
| `rng_seed` | int64 | row-level deterministic seed |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_a2_telemetry_observations.v0.1` |

### `load_profile_assignments.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `load_asset_id` | string | must reference an `asset_id` in `assets.parquet` |
| `loadshape_id` | string | stable OpenDSS `LoadShape` name or canonical profile ID |
| `municipality_id` | string | `marshfield:municipality:*` |
| `tile_id` | string | `marshfield:tile:*` or null when unavailable |
| `feeder_id` | string | must reference a generated feeder ID |
| `customer_class` | string | `residential`, `commercial`, `industrial_proxy`, `critical_facility`, or `unknown` |
| `profile_source` | string | source family such as `nrel_eulp`, `resstock`, `comstock`, `dsgrid`, or `synthetic_fixture` |
| `profile_source_version` | string | source dataset release/version |
| `source_geography` | string | source geography such as county, PUMA, state, or climate region |
| `source_building_type` | string | source building or archetype class |
| `weather_year` | int16 | weather/profile year when applicable |
| `time_step_minutes` | int16 | profile interval; REopt-ready Stage B critical-facility profiles are stored as 60-minute/8760-hour series after aggregating OEDI 15-minute energy profiles to hourly kW |
| `npts` | int32 | number of profile points |
| `p_scale_factor` | float64 | multiplier mapping source profile to load `kW` |
| `q_scale_factor` | float64 | multiplier or power-factor-derived mapping to `kvar` |
| `annual_energy_kwh` | float64 | generated active energy |
| `peak_kw` | float64 | generated active-power peak |
| `power_factor_policy` | string | e.g. `static_load_pf`, `class_default_pf`, `profile_qmult` |
| `diversity_group_id` | string | deterministic group used for profile diversity and correlation control |
| `rng_seed` | int64 | deterministic seed for profile selection |
| `source_provenance` | string | JSON provenance payload including selected OEDI building ID/profile URL when `synthetic_placeholder = false`, or schedule-overlay fixture rationale when `synthetic_placeholder = true` |
| `schema_version` | string | `stage_b_load_profile_assignments.v0.1` |

### `load_states.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `timestamp` | timestamp, UTC | profile sample time |
| `load_asset_id` | string | must reference `load_profile_assignments.parquet` |
| `loadshape_id` | string | profile used for this row |
| `p_kw` | float64 | active load in kW |
| `q_kvar` | float64 | reactive load in kvar |
| `served_fraction` | float64 | `[0, 1]` before MPC, normally `1.0` |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_b_load_states.v0.1` |

### `der_inventory.parquet`

`der_inventory.parquet` is the static placement-and-sizing table that records
which DERs are interconnected at which buses, along with the capacities used
for grid-forming reachability and supply-adequacy validation of SSAP-derived
Switch-Bounded Load Blocks. It is a separate artifact from
`der_states.parquet`, which records time-varying DER state during a scenario.
Methodology and citations are recorded in
[`der_placement_methodology.md`](der_placement_methodology.md).

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `der_id` | string | `marshfield:asset:der:<facility_token>:<resource_token>` |
| `facility_id` | string | must reference `critical_facilities.parquet` for Layer 1 and Layer 2 rows; nullable only for Layer 3 supplemental synthetic rows |
| `load_asset_id` | string | preferred service-point binding via `critical_load_assignments.parquet`; nullable when the DER is bound to a feeder proxy or a Layer 3 supplemental synthetic bus |
| `bus` | string | existing OpenDSS bus the DER is interconnected at |
| `block_id` | string | Switch-Bounded Load Block containing the DER or served load once block assignment is resolved; nullable while `assignment_status = unassigned` |
| `assignment_status` | string | `assigned` or `unassigned`; unassigned DERs are evidence inventory and are not operational voltage sources |
| `unassigned_reason` | string | machine-readable reason for an unassigned row, such as `pending_critical_load_assignment`; null only when `assignment_status = assigned` |
| `phases` | string | OpenDSS phase string |
| `nominal_voltage_kv` | float64 | interconnection nominal voltage |
| `resilience_asset_type` | string | `pv`, `bess`, `genset`, `microgrid`, `resilience_hub`, or `composite` |
| `pv_kw` | float64 | recommended PV capacity from REopt (Layer 2); nullable for non-PV rows |
| `bess_kw` | float64 | battery power capacity from REopt (Layer 2); nullable for non-BESS rows |
| `bess_kwh` | float64 | battery energy capacity from REopt (Layer 2); nullable for non-BESS rows |
| `genset_kw` | float64 | generator capacity from REopt (Layer 2); nullable for non-genset rows |
| `gfm_capable` | bool | `true` when the resource can operate grid-forming (BESS with GFM inverter, synchronous genset); `false` for PV-only rows |
| `placement_rule` | string | `evidence_anchored_mhmp`, `reopt_resilience_sizing`, or `arefifar_2013_supply_adequacy` |
| `evidence_rank` | string | `town_plan_primary`, `planned_authorized`, `massgis_primary`, `hifld_secondary`, `osm_supplemental`, `manual_fixture`, or `synthetic_supply_sufficiency` |
| `confidence` | string | `high`, `medium`, or `low` |
| `outage_duration_hours` | int16 | REopt input recorded for provenance; default `72` for FEMA Community Lifelines Stabilization horizon |
| `critical_load_fraction` | float64 | REopt input recorded for provenance; tier-derived per `der_placement_methodology.md` (1.00 / 0.50 / 0.25 by tier) |
| `reopt_feasible` | bool | REopt feasibility flag for Layer 2 rows; `null` for Layer 1 rows not yet sized and for Layer 3 supplemental rows |
| `source_provenance` | string | JSON provenance payload including REopt status, run UUID, payload digest, ResStock/ComStock archetype, utility tariff, weather/load year, seed, and outage-start-hour ensemble; full redacted REopt responses live in the sandbox `reopt_cache/` keyed by payload digest |
| `schema_version` | string | `stage_b_der_inventory.v0.1` |

Placement rule allowed values:

- `evidence_anchored_mhmp`: Layer 1 rows seeded from Marshfield's 2023 Multi-Hazard Mitigation Plan generator-backed facility list.
- `reopt_resilience_sizing`: Layer 2 capacities returned by an NREL REopt resilience sizing call against the corresponding Layer 1 facility row.
- `arefifar_2013_supply_adequacy`: Layer 3 supplemental synthetic GFM-DER rows added to close residual supply-adequacy gaps only after downstream block validation reports infeasibility for the Layer 1 + Layer 2 inventory.

Assignment completeness:

- Every row must either reference an existing `bus` or `block_id` with
  `assignment_status = assigned`, or carry `assignment_status = unassigned`
  plus a non-null `unassigned_reason`.
- Rows with `assignment_status = unassigned` remain evidence-backed inventory
  but must not be used as grid-forming voltage sources in
  PowerModelsONM/DNMG restoration cases.

### `controllable_switches.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `switch_id` | string | `marshfield:asset:controllable_switches:*` |
| `opendss_element` | string | `Line.<name>` |
| `source_id` | string | source identifier used by ONM events/settings, e.g. `line.<name>` |
| `from_bus` | string | existing OpenDSS bus |
| `to_bus` | string | existing OpenDSS bus |
| `phases` | string | OpenDSS phase string |
| `nominal_voltage_kv` | float64 | line-to-line nominal voltage where available |
| `switch_role` | string | `sectionalizing`, `tie`, `isolation`, or `critical_boundary` |
| `normal_state` | string | `open` or `closed` |
| `initial_state` | string | `open` or `closed` |
| `dispatchable` | bool | true when ONM may operate the switch |
| `status` | string | `enabled` or `disabled` |
| `thermal_rating_kva` | float64 | apparent-power limit, nullable when unknown |
| `current_rating_a` | float64 | current limit, nullable when unknown |
| `lon` | float64 | WGS84 longitude |
| `lat` | float64 | WGS84 latitude |
| `coordinate_source` | string | `device_point`, `line_midpoint`, `from_bus`, `to_bus`, or `synthetic_tie_midpoint` |
| `is_flood_relevant` | bool | true for Stage B Controllable Switches |
| `spatial_join_required` | bool | true for Stage B Controllable Switches |
| `placement_rule` | string | auditable synthesis rule |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_b_controllable_switches.v0.1` |

OpenDSS export rules:

- Each row must export as an OpenDSS `Line` with `Switch=Yes`; PowerModelsDistribution parses OpenDSS lines with `switch=y` into native switch objects.
- A companion `SwtControl` may be emitted to record the present and normal switch state for OpenDSS workflows, but ONM compatibility must not depend on `SwtControl`.
- ONM settings must preserve `dispatchable` and `status`; `status = disabled` means unavailable, while `initial_state = open` only means currently open.
- Tie switches default to `normal_state = open`, `initial_state = open`, `dispatchable = true`, and `status = enabled`.

### `critical_facilities.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `facility_id` | string | `marshfield:facility:*` |
| `facility_name` | string | public facility name when available |
| `lifeline` | string | FEMA Community Lifeline name |
| `lifeline_component` | string | FEMA component or project-specific component mapped to FEMA taxonomy |
| `facility_class` | string | normalized facility class |
| `criticality_tier` | string | `tier_0_life_safety`, `tier_1_response`, `tier_2_lifeline_support`, `tier_3_standard` |
| `criticality_weight` | float64 | nonnegative MPC service-priority weight |
| `lon` | float64 | WGS84 longitude |
| `lat` | float64 | WGS84 latitude |
| `municipality_id` | string | `marshfield:municipality:*` |
| `source_dataset` | string | e.g. `massgis_acute_care_hospitals`, `massgis_fire_stations`, `hifld_emergency_services` |
| `source_url` | string | public source URL |
| `source_date` | string | source update or publication date when available |
| `source_record_id` | string | source-layer identifier |
| `evidence_rank` | string | `town_plan_primary`, `massgis_primary`, `hifld_secondary`, `osm_supplemental`, `manual_fixture` |
| `confidence` | string | `high`, `medium`, `low` |
| `backup_power_status` | string | `documented_present`, `planned_or_recommended`, `not_documented`, or `unknown` |
| `resilience_asset_type` | string | `generator`, `battery`, `microgrid`, `resilience_hub`, `none`, or `unknown` |
| `hazard_exposure_summary` | string | optional JSON summary from official plans or spatial joins |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_b_critical_facilities.v0.2` |

### `critical_load_assignments.parquet`

Required fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `sandbox_id` | string | `marshfield` |
| `assignment_id` | string | `marshfield:critical_load_assignment:*` |
| `facility_id` | string | must reference `critical_facilities.parquet` |
| `load_asset_id` | string | must reference a load asset in `assets.parquet`; nullable only when `assignment_status = unmatched` |
| `matched_asset_id` | string | asset or Control Unit actually used for the match |
| `matched_asset_type` | string | `load`, `load_bus`, `transformer`, `control_unit`, or `feeder_proxy` |
| `matched_bus` | string | existing OpenDSS bus for the matched electrical service proxy; required when `assignment_status = assigned` |
| `phases` | string | phase count/string inherited from the matched load-bus load metadata when available |
| `nominal_voltage_kv` | float64 | nominal load-bus voltage inherited from Asset Registry `loads.csv` metadata when available |
| `control_unit_id` | string | nearest or serving Control Unit when known |
| `match_method` | string | `direct_service_point`, `nearest_load_bus`, `nearest_transformer`, `feeder_proxy`, or `manual_fixture` |
| `match_distance_m` | float64 | distance from facility to matched grid asset |
| `assignment_confidence` | string | `high`, `medium`, `low` |
| `criticality_tier` | string | inherited or adjusted tier |
| `criticality_weight` | float64 | nonnegative MPC service-priority weight |
| `assignment_status` | string | `assigned`, `unmatched`, `needs_review`, or `excluded_duplicate` |
| `source_provenance` | string | JSON provenance payload |
| `schema_version` | string | `stage_b_critical_load_assignments.v0.2` |

Assignment rules:

- The first operational rule is `nearest_valid_stage_a_load_bus`: match each
  public critical-facility point to the nearest valid Stage A `load_bus` asset
  only when it lies inside the declared maximum distance.
- Facilities outside the match radius must remain in the artifact with
  `assignment_status = unmatched`; their nearest candidate is recorded in
  `source_provenance` for review.
- `der_inventory.parquet` may inherit `bus`, `load_asset_id`, `phases`, and
  `nominal_voltage_kv` only from rows where `assignment_status = assigned`.

### `switch_bounded_load_blocks` in `control_units.parquet`

Switch-Bounded Load Blocks are Stage B Candidate Control Unit rows, not a new
physical asset table. They are derived by opening every synthesized
Controllable Switch and computing connected components in the resulting graph.
This follows the DNMG block abstraction where blocks remain static for a given
switch set and connected components/microgrids change as switches close.

Required values and fields:

| Field | Type | Unit / allowed values |
|---|---|---|
| `control_unit_id` | string | `marshfield:control_unit:switch_bounded_load_block:*` |
| `control_unit_type` | string | `switch_bounded_load_block` |
| `control_unit_stage` | string | `stage_b` |
| `parent_control_unit_id` | string | feeder or source-area Control Unit when a unique parent exists |
| `member_asset_ids` | list[string] | assets inside the block when all Controllable Switches are open |
| `boundary_bus_ids` | list[string] | buses incident to boundary switches |
| `source_ids` | list[string] | source/substation or DER voltage-source candidates inside the block |
| `served_load_kw` | float64 | static load inside the block |
| `critical_load_weight` | float64 | sum of assigned critical-load weights, zero before assignments validate |
| `candidate_status` | string | `active`, `needs_review`, or `deferred` |
| `candidate_basis` | string | `opened_controllable_switch_components` |

Validation rules:

- Every boundary switch must reference a known `controllable_switches.parquet`
  row.
- Every member asset must reference `assets.parquet`.
- If only cross-feeder tie switches exist, block derivation may be feeder-sized;
  nontrivial DNMG blocks require intra-feeder sectionalizing or isolation
  Controllable Switches.
- Critical-load-boundary switches must not be used to derive blocks until
  `critical_facilities.parquet` and `critical_load_assignments.parquet`
  validate.

## Provenance Requirements

Each run must write a companion manifest:

- `run_id`
- generation timestamp in UTC
- git commit or dirty-worktree marker
- input file paths and SHA-256 hashes
- source event path and event manifest hash, where applicable
- versions of Python, OpenDSSDirect.py, OpenDSS engine, PyDSS if used, numpy, pandas, pyarrow, xarray, geopandas, shapely
- random-number generator family and seed hierarchy
- schema versions for every emitted table
- Parquet artifact hashes and any derived debug CSV hashes
- citations used for model assumptions

## Coordinate Validation Policy

Stage A is strict for assets that participate in flood exposure, asset-state inference, telemetry targeting, Control Unit spatial membership, or any other geospatial join. Those rows are Flood-Relevant Assets and must have valid `lon` and `lat` values before Gate A passes.

Required coordinate fields in `assets.parquet`:

- `coordinate_status`: one of `valid`, `missing_exempt`, `invalid`
- `coordinate_source`: source of the coordinate, such as `buscoords.csv`, `derived_midpoint`, `source_bus`, `non_spatial_metadata`, or `unknown`
- `is_flood_relevant`: boolean
- `spatial_join_required`: boolean
- `coordinate_exemption_reason`: non-empty only when `coordinate_status = missing_exempt`

Rules:

- If `is_flood_relevant = true`, then `coordinate_status` must be `valid`.
- If `spatial_join_required = true`, then `coordinate_status` must be `valid`.
- If `coordinate_status = valid`, then `lon` and `lat` must be finite WGS84 longitude/latitude values.
- If `coordinate_status = missing_exempt`, then `is_flood_relevant = false`, `spatial_join_required = false`, and `coordinate_exemption_reason` must explain why the row is non-spatial.
- `coordinate_status = invalid` always fails Gate A.
- Missing coordinates are never silently filled with zero, centroids, source-bus coordinates, or feeder-level defaults. Any imputation must be represented as a new coordinate source and justified in the manifest.

## Flood-Relevant Asset Classification

Stage A classifies flood relevance for flood-depth fragility and spatial hazard sampling only. Wind, tree contact, pole-foundation damage, and washout mechanisms are optional future hazards and are not part of the current flood-depth classification.

Required classification:

| Asset class | `is_flood_relevant` | Required coordinate basis | Rationale |
|---|---:|---|---|
| `transformer` | true | transformer location, preferably LV-side/pad bus | Pad-mounted and service equipment can be flood-exposed. |
| `source` | true | source/substation bus | Substation/source availability can be flood-exposed. |
| `load_bus` | true | load bus | Load service state and exposure are spatially joined to flood depth. |
| `underground_line_proxy` | true | endpoint, midpoint, or explicit splice/vault proxy | Underground vaults, splice boxes, and terminations can be flood-exposed. |
| `der` | true | DER point of interconnection or physical site | DER availability and telemetry can depend on site exposure. |
| `fuse_proxy` | true | Stage A line midpoint or documented protective-device proxy | SHIFT-derived fuse-like line equipment can be flood-exposed and observed in fixtures, but it is not a controllable switch action. |
| `controllable_switch` | true | Stage B switch device location or associated line/bus proxy | Switching availability, state, and telemetry can depend on device exposure. |
| `sensor` | true | sensor device or measured asset location | Telemetry availability/observation state can depend on exposure. |
| `critical_load` | true | critical facility or matched bus | Served critical load and consequence depend on facility/bus exposure. |
| `overhead_line` | false | optional geometry/proxy for visualization | Excluded from flood-depth fragility unless a later wind/tree/pole-foundation model is added. |
| `line` | false | optional endpoint/midpoint proxy for topology | Generic line rows are retained for topology but excluded from flood-depth fragility until classified more specifically. |
| `metadata_only` | false | none | Non-spatial records may be coordinate-exempt with an explicit reason. |

Rules:

- `underground_line_proxy` rows must state their proxy method in `coordinate_source`, such as `line_midpoint`, `from_bus`, `to_bus`, or `splice_vault_inventory`.
- `overhead_line` rows may have coordinates for mapping and topology, but `is_flood_relevant = false` under the current flood-depth-only protocol.
- Stage A must include overhead lines in `assets.parquet` as topology-only assets when they exist in the Asset Registry.
- Overhead-line rows must use `asset_type = overhead_line`, `is_flood_relevant = false`, and `spatial_join_required = false`.
- Overhead-line rows with valid geometry or endpoint-derived proxy coordinates use `coordinate_status = valid`; overhead-line rows without coordinates may use `coordinate_status = missing_exempt` with `coordinate_exemption_reason = topology_only_overhead_line`.
- Overhead-line rows must remain available to Control Unit derivation, network topology, visualization, and future hazard extensions even though current flood-depth asset-state generation ignores them.
- If the Optional Wind Extension is added later, overhead-line classification must be revisited in a new protocol section and ADR.
- Any class not listed here defaults to `is_flood_relevant = true` when it participates in asset-state inference, telemetry targeting, Control Unit spatial membership, or a spatial join.

## Stage A Control Unit Policy

Stage A1 emits feeder-level Control Units only. These are deterministic from the current Asset Registry and do not depend on synthesized controllable switches, DERs, controllable loads, or critical-load blocks.

Required Stage A1 values:

```text
control_unit_type  = feeder
control_unit_stage = stage_a
candidate_status   = active
candidate_basis    = asset_registry_feeder
control_unit_id    = marshfield:control_unit:feeder:<feeder_id>
source_feeder_id   = <feeder_id>
parent_control_unit_id = null
```

Rules:

- Every `feeder_id` present in `feeders.csv` must produce exactly one Feeder Control Unit.
- Stage A1 Feeder Control Units include all Stage A1 assets whose `feeder_id` matches the unit's `source_feeder_id`.
- `member_asset_ids` must reference existing `assets.parquet` rows only.
- `source_ids` should include the source/substation assets associated with the feeder when available.
- `boundary_bus_ids` may be empty in Stage A1 because cross-feeder tie and island-boundary candidates are not yet synthesized.
- `served_load_kw` is the sum of load or load-bus demand available from the Asset Registry.
- `critical_load_weight`, `der_capacity_kw`, and `der_capacity_kwh` may be zero in Stage A1 until critical loads and DERs are synthesized.
- Island candidates, DER clusters, load blocks, and controllable-switch-defined restoration areas are Candidate Control Units and must not be emitted in Stage A1.
- The schema includes `parent_control_unit_id`, `candidate_status`, and `candidate_basis` so later Candidate Control Units can point back to their feeder baseline without changing the table shape.

## Stage A2 Event And Monte Carlo Policy

Stage A2 uses 50 Monte Carlo draws per storm scenario by default. This is the minimum accepted draw count for normal sandbox exports because it gives downstream belief, telemetry, and MPC code a stochastic ensemble while keeping artifacts small enough for rapid iteration. Research runs may raise the count to 100 or more when reporting uncertainty-sensitive results.

Stage A2 depth inputs are asset-keyed event samples:

```text
asset_id,timestamp,sampled_depth_m
```

The preferred production path is to sample SURGE/SFINCS water depth at every Flood-Relevant Asset coordinate. Stage A2 supports two input forms:

- `--sfincs-event-dir <run>` reads a completed `sfincs_map.nc` and samples nearest-cell peak depth at Stage A1 Flood-Relevant Asset coordinates.
- `--event-depth-csv <table>` consumes a pre-sampled asset-keyed table from an external SURGE/SFINCS workflow.

HydroMT/SFINCS model construction and simulation are treated as an optional outside-model workflow; they must not be imported by the base exporter or required by core unit tests.

If no sampled event-depth table is supplied, Stage A2 may emit a clearly marked `synthetic_coordinate_profile` event fixture. This fixture is allowed only for pipeline validation, software testing, and interface development. It is not a physical Marshfield flood simulation and must not be used as evidence for scientific claims.

Initial Stage A2 telemetry targets:

- Tier 1 direct asset observations for Stage A1 `source`, `transformer`, and `fuse_proxy` assets.
- Tier 2 feeder-summary observations for every Stage A1 Feeder Control Unit.
- `load_bus` and `underground_line_proxy` assets participate in asset-state generation but are not directly observed in the initial telemetry fixture.

## Stage B Load Profile Method

Stage B load time series must use public, auditable building-stock load-profile sources as the primary basis. The default public basis is NREL End-Use Load Profiles, ResStock, ComStock, and dsgrid data, selected at the most defensible available geography for Marshfield, such as county, PUMA, state, ISO/RTO, or climate zone [R9, R10, R11]. SMART-DS profiles are validation and comparison references for distributional similarity; they are not the primary source for Marshfield load shapes.

The generation procedure is:

1. Classify each load asset into a customer class using the synthetic network metadata and available public land-use, parcel, building, or critical-facility evidence.
2. Select candidate profile families from the public source data by customer class, source geography, building type, weather year, and time step. For Marshfield critical facilities, the first live data channel is OEDI End-Use Load Profiles by Plymouth County (`G2500230`), with ComStock/ResStock individual-building parquet files selected by building type and deterministic facility token.
3. Assign diverse profiles to nodal loads using deterministic seeded sampling within municipality, tile, feeder, and customer-class groups. Directly scaling one feeder-level curve to all nodal loads is not allowed because it removes load diversity and variability needed for QSTS and MPC studies [R12].
4. Scale active and reactive power so generated profiles are consistent with the static OpenDSS `Loads.dss` `kW/kvar` values and any documented feeder-level calibration targets.
5. Export OpenDSS `LoadShape` objects and profile CSVs. OpenDSS applies load-shape multipliers to base load values, or actual `kW/kvar` values when explicitly using actual-value shapes [R13].
6. Emit `load_profile_assignments.parquet` and `load_states.parquet` with source dataset, geography, building type, weather year, seed, scale factor, diversity group, power-factor policy, and schema version.

Allowed profile sources:

- `nrel_eulp`: preferred public source for calibrated end-use load profiles.
- `resstock`: preferred residential source when querying or subsetting residential building-stock runs directly.
- `comstock`: preferred commercial source when querying or subsetting commercial building-stock runs directly.
- `dsgrid`: preferred source for broader demand-side grid projections or county/sector/end-use aggregates.
- `synthetic_fixture`: allowed only for unit tests and software plumbing; not valid for research claims.

Calibration targets should be applied in this order when available:

1. Static OpenDSS load `kW/kvar` for each load.
2. Feeder-level peak and annual energy derived from the generated synthetic feeder model.
3. Municipality or county class mix inferred from public building/land-use evidence.
4. Public planning or aggregate load references if later added and documented.

Validation summaries must compare the generated Marshfield profiles against both source-profile statistics and SMART-DS reference-profile statistics:

- per-class annual energy and peak-to-average ratio
- feeder coincidence factor and diversity factor
- daily and seasonal peak timing
- distribution of nodal profile correlations within feeders
- reactive-power or power-factor policy consistency
- OpenDSS `LoadShape` file completeness and `Loads.dss` assignment resolution

## Stage B Critical Facility Method

Stage B critical-load synthesis must use public emergency-management and infrastructure data sources. FEMA Community Lifelines provide the taxonomy for facility classes and service-priority categories [R14, R15]. Marshfield's official Multi-Hazard Mitigation Plan is the town-specific critical-facility source; MassGIS public facility layers are the primary structured geometry source for Massachusetts-specific facilities; HIFLD is the secondary source for national critical-infrastructure categories not covered cleanly by MassGIS; OSM is supplemental context only and must not override higher-rank public agency sources [R16, R17, R18, R19, R20, R21, R22, R23].

Criticality is represented as tiered MPC service weights:

| Tier | Meaning | Typical facilities |
|---|---|---|
| `tier_0_life_safety` | immediate life-safety and emergency command functions | acute hospitals with emergency departments, trauma centers, 911/dispatch, emergency operations, major water/wastewater facilities |
| `tier_1_response` | response-force continuity | fire, police, EMS, shelters, key responder communications |
| `tier_2_lifeline_support` | community lifeline support and restoration logistics | schools used as shelters, public works, fuel, transportation nodes, selected municipal facilities |
| `tier_3_standard` | ordinary load without documented critical-facility evidence | residential/commercial load not matched to a critical facility |

The source hierarchy is:

1. Marshfield's official Multi-Hazard Mitigation Plan for town-vetted critical-facility inclusion and local hazard context.
2. MassGIS facility layers when a relevant statewide Massachusetts layer exists, especially for reproducible geometry and source IDs.
3. HIFLD public layers for relevant national categories missing or incomplete in MassGIS.
4. OSM tags only for supplemental context, QA, or fixture discovery.
5. Manual scenario fixtures only when explicitly marked as `manual_fixture` and excluded from research claims unless independently sourced.

The generation procedure is:

1. Clip public facility sources to `power_extent.geojson`.
2. Normalize facility classes to FEMA Community Lifeline categories and project criticality tiers.
3. Deduplicate overlapping source records by source hierarchy, name, address, and geometry.
4. Assign each facility to the nearest or directly associated synthetic load asset, transformer, feeder, or Control Unit.
5. Emit unmatched facilities with `assignment_confidence = low` rather than silently dropping them.
6. Write `critical_facilities.parquet`, `critical_load_assignments.parquet`, and a validation summary.

Validation summaries must report:

- facility counts by lifeline, class, source dataset, municipality, and criticality tier
- unmatched facility count and percentage
- match-distance distribution by facility class
- duplicate-resolution counts by source hierarchy
- tier-weight table used by MPC
- all source URLs and source update dates available from upstream metadata

## PowerModelsONM Event-Window Bundle (MVP, Moring et al. 2025)

The Marshfield sandbox feeds PowerModelsONM not only a static network but a
per-event, per-Monte-Carlo-draw bundle of time-series inputs. The minimum
viable bundle implements the two-stage Robust Partitioning and Operation
Problem (RPOP) of Moring et al. (2025) [R37]; PV irradiance time series and
storage dynamics are intentionally out of scope because [R37] treats DGs as
controllable injections (Section II, eqn 1) and marks storage as future work
(Section VI).

### Slice 1: Nodal demand assignment for non-critical loads

Implemented in [`dft.power.nodal_load_profiles`](../../src/dft/power/nodal_load_profiles.py).

- One `load_profile_assignments.parquet` v0.1 row is produced for every
  non-critical load in `loads.csv`, so the union of Stage B critical-facility
  rows and Slice 1 nodal rows covers every load in the network.
- Customer-class classification uses the **Eversource Massachusetts MDPU
  1310 rate-class kW thresholds**: residential when `peak_kw < 10`,
  commercial when `10 <= peak_kw < 200`, industrial_proxy when `peak_kw >= 200`.
  This matches the tariff selector already cited in
  [`dft.power.tariffs`](../../src/dft/power/tariffs.py).
- Diversity is enforced by deterministic seeded sampling within
  `(feeder_id, customer_class)` buckets, satisfying Zhu & Mather [R12].

### Slice 2: Event-window slicing

Implemented in [`dft.power.event_window`](../../src/dft/power/event_window.py).

- Default horizon is 72 hours, matching the FEMA Community Lifelines
  Stabilization horizon already used by
  [`dft.power.der_inventory.FEMA_COMMUNITY_LIFELINES_OUTAGE_HOURS`](../../src/dft/power/der_inventory.py)
  and REopt sizing defaults.
- Each load's 8760-hour annual ResStock / ComStock / EULP profile is sliced
  to the contiguous window starting at the SURGE/SFINCS event start
  timestamp, wrapping across the year boundary when needed.

### Slice 3: Load uncertainty bounds (paper's set 𝒰)

Implemented in [`dft.power.load_uncertainty`](../../src/dft/power/load_uncertainty.py).

- Emits one `(event_id, mc_draw, timestep, load_asset_id, cluster_id,
  nominal_kw, lower_kw, upper_kw, band_fraction)` row per (load, timestep)
  pair, matching the per-load uncertainty band in eqn (1) of [R37].
- The cluster id is the Switch-Bounded Load Block id from
  `control_units.parquet`, so the cluster set Γ in eqn (12) of [R37] reuses
  the Stage B Candidate Control Units rather than inventing a new spatial
  grouping.
- Default symmetric band is ±20%, matching the [R37] Section V case study
  ("20% load uncertainty," Fig. 5; T=1..60 at 90%, T=61..120 at 100%,
  T=121..180 at 120% of nominal).

### Slice 4: PowerModelsONM `events.json` serializer

Implemented in [`dft.power.restoration_events`](../../src/dft/power/onm_events.py).

- Converts `asset_states.parquet` rows for one `(event_id, mc_draw)` slice
  into PMONM event entries `{"timestep", "event_type": "breaker",
  "affected_asset", "event_data": {"status": "OPEN"}}`.
- One breaker-open event is emitted per `available → failed` transition,
  indexed by the 1-indexed PMONM timestep (eqn (8a)-(8j) loop "forall
  k = 1, 2, ..." in [R37]).
- Assets without an `asset_to_dss_element` mapping are recorded in the
  result's `skipped_asset_ids` so they cannot silently disappear from the
  optimizer's view.

### Bundle composition

Implemented in
[`dft.power.marshfield_onm_run.build_marshfield_onm_event_window_artifacts`](../../src/dft/power/marshfield_onm_run.py).
For each `(event_id, mc_draw)` the orchestrator writes:

```text
artifacts/sandboxes/marshfield/data/derived/onm_runs/<event_id>/draw_<mc_draw>/
  events.json
  load_uncertainty.json
  nominal_load_window.json
  load_profile_assignments.json
  manifest.json
```

`manifest.json` records the protocol version, schema versions of every
slice, root seed, horizon hours, uncertainty band fraction, and per-slice
citations so each run is reproducible against the same RPOP solver build.

## Randomness And Reproducibility

Use deterministic, hierarchical seeds:

```text
root_seed
  -> event_seed(event_id)
  -> draw_seed(event_id, mc_draw)
  -> asset_state_seed(event_id, mc_draw, asset_id, timestamp)
  -> observation_seed(event_id, mc_draw, target_id, timestamp)
```

All stochastic products must be exactly reproducible when the same inputs, code revision, dependency versions, and root seed are used. If dependency drift changes output, the manifest must make that visible.

## Validation Gates

### Gate A1 Static Checks

- Every `asset_id` is unique and stable.
- No `asset_id`, `control_unit_id`, or `run_id` is derived from row number, sort order, timestamp alone, random UUID, or Python object ID.
- Every `asset_id` starts with `marshfield:asset:`.
- Every `control_unit_id` starts with `marshfield:control_unit:`.
- Every source-derived ID token is normalized according to the Stable Identifier Scheme.
- Canonical Stage A1 files are Parquet artifacts.
- Any CSV files use the `.debug.csv` suffix and are derived from matching Parquet artifacts.
- Validation and downstream simulation read Parquet, not CSV.
- Every Flood-Relevant Asset has `coordinate_status = valid`.
- Every asset class follows the Flood-Relevant Asset Classification table.
- Every underground line proxy records its coordinate proxy method.
- Every overhead line excluded from flood relevance is excluded only for flood-depth fragility, not because it is unimportant to grid operation.
- Every overhead line present in the Asset Registry is represented in `assets.parquet` as a topology-only `overhead_line` row.
- Every asset requiring a spatial join has `coordinate_status = valid`.
- Every asset with `coordinate_status = valid` has finite WGS84 longitude and latitude.
- Every coordinate exemption is explicitly marked `missing_exempt` with a non-empty `coordinate_exemption_reason`.
- No missing coordinate is silently imputed or defaulted.
- Every `control_unit_id` is unique.
- Every feeder in `feeders.csv` has exactly one Stage A1 Feeder Control Unit.
- Every Stage A1 Control Unit has `control_unit_type = feeder` and `control_unit_stage = stage_a`.
- No island, DER-cluster, controllable-switch-defined, or load-block Candidate Control Unit appears in Stage A1.
- Every Control Unit member points to an existing `asset_id`.

### Gate A2 Event Checks

- Gate A1 has passed for the `assets.parquet` and `control_units.parquet` inputs referenced by the run manifest.
- Asset-state rows reference known assets only.
- Telemetry target IDs reference known assets or Control Units.
- Units are explicit for every numeric telemetry quantity.
- Failure probabilities are in `[0, 1]`.
- Re-running with the same seed reproduces row counts and sampled binary states.

### Gate B Checks

- Every DER, controllable switch, and controllable load state references known assets, buses, or Control Units.
- Every Stage B Switch-Bounded Load Block references known assets and known
  boundary Controllable Switches.
- Every `load_profile_assignments.parquet` row references a known load asset.
- Every generated `LoadShape` referenced by `Loads.dss` exists and has the expected `npts` and interval.
- Load profiles use public building-stock sources or are explicitly marked `synthetic_fixture`.
- Non-fixture load profiles record source geography, customer class, source version, weather year, seed, scale factor, and diversity group.
- Nodal loads are diversity-preserving; a single directly scaled feeder curve must not be assigned to every load in a feeder.
- Feeder aggregate `load_states.parquet` totals are consistent with static `Loads.dss` `kW/kvar` and recorded calibration tolerances.
- Every `critical_facilities.parquet` row records FEMA lifeline, source dataset, source URL, evidence rank, confidence, and criticality tier.
- Every `critical_load_assignments.parquet` row references a known facility and known load asset or records an explicit unmatched/low-confidence status in the validation summary.
- Criticality tiers use documented MPC weights and do not collapse to a binary critical/non-critical flag.
- OSM-derived evidence is supplemental only and cannot override MassGIS or HIFLD source records.
- Device states have physically valid bounds.
- Telemetry observations distinguish full observability, sampled observability, and unobserved state.
- Delay and noise assumptions are recorded in machine-readable fields.

### Gate C Checks

- Every MPC action references known controllable-switch, DER, and load identifiers.
- Solver status is recorded for every solve.
- Infeasible solves emit explicit diagnostic rows instead of disappearing.
- Objective components sum to the reported objective value within tolerance.
- OpenDSS/PyDSS validation outcomes are linked to the trajectory rows they validate.

## Source And Method Citations

- SMART-DS is the canonical v0 regional substrate because it provides standardized synthetic distribution models and scenarios suitable for algorithm testing [R3].
- SHIFT remains the Marshfield sandbox source because it generates synthetic distribution feeders from open street/building data and distribution design principles [R4].
- OpenDSS is the power-flow engine for distribution-system simulation in this sandbox [R5].
- PyDSS may be used for time-series organization, controller integration, Monte Carlo studies, and automated result export around OpenDSS [R6].
- ERAD-derived fragility curves are used only for failure/impact modeling, not as the resilience scorer or MPC objective [R7, R8].
- Marshfield load profiles use public calibrated building-stock load-profile data from NREL EULP, ResStock, ComStock, or dsgrid as the primary source basis; SMART-DS profiles are comparison references rather than copied source profiles [R9, R10, R11].
- Distribution QSTS load synthesis must preserve nodal diversity and variability rather than assigning a single scaled feeder curve to every load [R12].
- OpenDSS `LoadShape` is the simulator export mechanism for active/reactive load profiles in full `opendss` parity [R13].
- Marshfield critical-load synthesis uses FEMA Community Lifelines as the taxonomy, MassGIS as the primary public Massachusetts facility evidence, HIFLD as secondary national evidence, and OSM only as supplemental context [R14, R15, R16, R17, R18, R19, R20, R21, R22].
- FAIR and reproducible-computing references justify the manifest, versioning, provenance, seed, and validation requirements [R1, R2].

## References

[R1] Wilkinson, M. D., Dumontier, M., Aalbersberg, I. J., et al. "The FAIR Guiding Principles for scientific data management and stewardship." *Scientific Data* 3, 160018 (2016). https://doi.org/10.1038/sdata.2016.18

[R2] Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., and Teal, T. K. "Good enough practices in scientific computing." *PLOS Computational Biology* 13(6): e1005510 (2017). https://doi.org/10.1371/journal.pcbi.1005510

[R3] NREL. "SMART-DS: Synthetic Models for Advanced, Realistic Testing: Distribution Systems and Scenarios." https://www.nrel.gov/grid/smart-ds

[R4] NREL. "SHIFT: Simple Synthetic Distribution Feeder Generation Tool." https://www.nrel.gov/research/software/shift-simple-synthetic-distribution-feeder-generation-tool

[R5] EPRI. "Introduction to OpenDSS." https://opendss.epri.com/IntroductiontoOpenDSS.html

[R6] NREL. "PyDSS documentation." https://nrel.github.io/PyDSS/index.html

[R7] NREL. "ERAD: Equitable Resiliency Analysis Tool for Distribution System." https://www.nrel.gov/research/software/erad--equitable-resiliency-analysis-tool-for-distribution-system

[R8] Duwadi, K., Palmintier, B., Latif, A., Sedzro, K. S. A., and Abraham, S. A. "Energy Resilience Analysis for electric Distribution systems (ERAD)." Zenodo software record. https://zenodo.org/records/17640811

[R9] NREL. "End-Use Load Profiles for the U.S. Building Stock." https://www.nrel.gov/buildings/end-use-load-profiles.html

[R10] NREL. "BuildStockQuery: Python library for querying datasets generated by ResStock and ComStock." https://github.com/NREL/buildstock-query

[R11] NREL. "Demand-Side Grid Toolkit (dsgrid)." https://www.nrel.gov/analysis/dsgrid

[R12] Zhu, X., and Mather, B. "Data-Driven Load Diversity and Variability Modeling for Quasi-Static Time-Series Simulation on Distribution Feeders." NREL publication record. https://research-hub.nrel.gov/en/publications/data-driven-load-diversity-and-variability-modeling-for-quasi-sta-3

[R13] EPRI. "OpenDSS LoadShape." https://opendss.epri.com/LoadShape.html

[R14] FEMA. "Community Lifelines." https://www.fema.gov/emergency-managers/practitioners/lifelines

[R15] FEMA. "Community Lifelines Implementation Toolkit." https://www.fema.gov/emergency-managers/practitioners/lifelines-toolkit

[R16] MassGIS. "MassGIS Data Layers." https://www.mass.gov/info-details/massgis-data-layers

[R17] MassGIS. "MassGIS Data: Acute Care Hospitals." https://www.mass.gov/info-details/massgis-data-acute-care-hospitals

[R18] MassGIS. "MassGIS Data: Fire Stations." https://www.mass.gov/info-details/massgis-data-fire-stations

[R19] MassGIS. "MassGIS Data: Police Stations." https://www.mass.gov/info-details/massgis-data-police-stations

[R20] MassGIS. "MassGIS Data: MassDEP Estimated Sewer System Service Area Boundaries." https://www.mass.gov/info-details/massgis-data-massdep-estimated-sewer-system-service-area-boundaries

[R21] HIFLD Open. "Emergency Services datasets." https://maps.nccs.nasa.gov/mapping/rest/services/hifld_open/emergency_services/MapServer

[R22] EPA. "FEMA's Community Lifelines Construct." https://www.epa.gov/waterresilience/femas-community-lifelines-construct

[R23] Town of Marshfield. "Final Marshfield Multi-Hazard Mitigation Plan." https://www.marshfield-ma.gov/Documents/Departments/Town%20Hall/Planning/Marshfield%20Multi%20Hazard%20Mitigation%20Plan/marshfield_mhmp_final_w_appendices_compressed_120434.pdf

[R24] MassGIS. "MassGIS Data: Massachusetts Schools (Pre-K through High School)." https://www.mass.gov/info-details/massgis-data-massachusetts-schools-pre-k-through-high-school

[R25] MassGIS. "MassGIS Data: Public Water Supplies." https://www.mass.gov/info-details/massgis-data-public-water-supplies

[R26] MassDEP. "Water Utility Resilience Program." https://www.mass.gov/info-details/water-utility-resilience-program

[R27] MassCEC. "Energy Resilience." https://www.masscec.com/energy-resilience

[R28] MassCEC. "Clean Energy and Resilience (CLEAR)." https://www.masscec.com/program/clean-energy-and-resilience-clear

[R29] Massachusetts DOER. "CCERI Program Goals." https://www.mass.gov/info-details/cceri-program-goals

[R30] PowerModelsONM. "ONM Workflow." https://lanl-ansi.github.io/PowerModelsONM.jl/stable/manual/onm_workflow.html

[R31] PowerModelsONM. "Settings Schema." https://lanl-ansi.github.io/PowerModelsONM.jl/v3.3/schemas/input-settings.schema.iframe.html

[R32] PowerModelsDistribution. "External Data Formats." https://lanl-ansi.github.io/PowerModelsDistribution.jl/stable/manual/external-data-formats.html

[R33] PowerModelsDistribution. "Conversion to Mathematical Model." https://lanl-ansi.github.io/PowerModelsDistribution.jl/stable/manual/eng2math.html

[R34] PowerModels. "Switch Model." https://lanl-ansi.github.io/PowerModels.jl/stable/switch/

[R35] EPRI. "OpenDSS SwtControl." https://opendss.epri.com/SwtControl.html

[R36] NatLabRockies. "NRELDynaGrid." https://github.com/NatLabRockies/NRELDynaGrid/tree/main

[R37] Moring, H., Poolla, B. K., Nagarajan, H., Mathieu, J. L., Bernstein, A., and Fobes, D. M. "Reconfiguration and Real-Time Operation of Networked Microgrids Under Load Uncertainty." arXiv:2504.15084v2 (2025). Local copy: `docs/Reconfiguration and Real-Time Operation of Networked Microgrids Under Load Uncertainty - 2504.15084v2.pdf`
