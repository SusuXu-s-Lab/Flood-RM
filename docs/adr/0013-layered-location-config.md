# Layered Location Configuration

Accepted.

Inland and coastal locations now share a setting-level base config
(`locations/_shared/inland_base.yaml`, `locations/_shared/coastal_base.yaml`)
that each location inherits via a top-level `extends:` key in its `config.yaml`;
the location's own `sfincs.yaml`/`wflow.yaml` carry only its deltas (data_root,
soil rasters, domain ids, review flags, structures, runup transects) and override
the base. `define_location` resolves `extends:` by deep-merging the base under the
location (override wins). We chose this over self-contained per-location files
because the location YAMLs are the user-facing surface for standing up a new
location, and ~95% of austin/greensboro was duplicated copy-paste; the base makes
the per-location edit surface small and obvious (~10 lines) and makes
austin↔greensboro methodology parity structural instead of manual. The trade-off
is one level of indirection — a reader sees a short `sfincs.yaml` and must follow
`extends:` to the base for the full picture. CRS is single-sourced from
`project.model_crs` (consumers already fall back to it), so it is set once per
location. The merged config is pinned by a characterization snapshot test
(`tests/flood_rm/region_config_snapshot_test.py`) that proved the restructure
behavior-preserving. Extends to ADR 0007 (Location Workspace Interface).
