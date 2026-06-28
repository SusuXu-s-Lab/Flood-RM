# HydroMT Model Recipe YAML

## Status

Superseded by ADR-0018.

## Context

ADR-0013 made Austin, Greensboro, and Marshfield model YAMLs very small by
putting most repeated SFINCS and Wflow settings in shared inherited bases. That
reduced copy-paste, but it made the stakeholder-facing Location Workspace harder
to audit: `sfincs.yaml` and `wflow.yaml` no longer showed the HydroMT model
steps that actually build or update the models.

The coupling reference under `code/coupling/hydromt-wflow-sfincs/src` is easier
to inspect because its YAML files read as direct HydroMT recipes.

## Decision

Use self-contained **Model Recipe Configuration** files for HydroMT-facing model
setup and update syntax:

- `sfincs_build.yml`
- `sfincs_update_forcing.yml`
- `wflow_build.yml`
- `wflow_update_forcing.yml`

These files use top-level HydroMT step names such as `setup_basemaps`,
`setup_rivers`, `setup_grid_from_region`, `write_forcing`, and installed
component writes such as `forcing.write`. Wflow remains inland-only for Austin
and Greensboro; Marshfield declares SFINCS recipes, including the wave-coupled
recipe path, but no Wflow recipes.

Shared runtime defaults may still support existing code paths during migration,
but model setup choices should be visible in the recipe files or directly in
the notebooks when they are extra-model policy.

## Consequences

- This supersedes ADR-0013 for model setup YAML. `_shared` bases must not be
  used to hide HydroMT model build/update syntax from Location Workspaces.
- Notebook cells should load local recipe includes instead of reaching into the
  coupling example directory.
- Wflow code normalizes readable top-level recipes into the `steps` list that
  HydroMT-Wflow model objects consume internally.
- `data_sources.yaml` should stay source-facing: source identifiers, collection
  windows, and artifact paths belong there; review policy, dependence policy,
  sampling choices, and handoff choices belong in readable notebook code or
  focused tests.
