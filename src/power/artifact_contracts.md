# Artifact contracts

The package treats files as interfaces. Each stage may be re-run independently as long as its input artifacts are present.

| Artifact | Producer | Consumer | Claim |
|---|---|---|---|
| `asset_registry/*.csv` | `registry.build_registry` | all later stages | deterministic OpenDSS/GDM baseline topology normalization |
| `assets.parquet` | `dataset.export_base` | event states, resilience layers, ONM mapping | static SMART-DS-like asset table |
| `control_units.parquet` | `dataset.export_base` | telemetry, load matches, audits | feeder/control hierarchy |
| `asset_states.parquet` | `dataset.export_stage_a2` | ONM events, impact audit | Monte Carlo asset availability under hazard depth |
| `telemetry_observations.parquet` | `dataset.export_stage_a2` | audit and scenario replay | synthetic SCADA/OMS observations |
| `critical_facilities.parquet` | `DistributionCase.add_resilience_layers` | load matching, DER | reviewed public-facility evidence |
| `critical_load_assignments.parquet` | `facilities.build_load_matches` | profiles, DER | service proxy, not utility service-record truth |
| `load_profile_assignments.parquet` | `profiles.load_inputs` | ONM loadshapes, DER sizing | synthetic or OEDI 8760 profile assignment |
| `der_inventory.parquet` | `der.build_der_inventory` / `der.size_der` | ONM export, readiness | DER assignment and sizing provenance |
| `controllable_switches.parquet` | `ssap.write_switches` | ONM export, blocks | SSAP-derived switch candidates |
| `switch_bounded_load_blocks.parquet` | `blocks.build_blocks` | ONM settings, load uncertainty | block partition after opening controllable switches |
| `network.dss` + `settings.json` | `onm.build_powermodels_onm_export` | PMONM/DynaGrid | solver-facing export |
| `events/<event>/draw_<m>/run_manifest.json` | `onm.materialize_onm_run_bundle` | PMONM/DynaGrid runs | reproducible event-conditioned run bundle |
| validation reports | `audit` / `readiness` | stakeholders, CI | evidence ledger and blockers |

