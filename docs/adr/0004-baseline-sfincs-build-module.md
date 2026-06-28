# Baseline SFINCS Build Module

## Status

Accepted.

## Context

The baseline SFINCS workflow needs to be reusable across Study Locations while
preserving the decision in ADR-0001: regular-grid and wave-coupled builds use
separate notebooks and produce different artifact layouts.

Notebooks remain useful as readable, educational runbooks. They should not own
the decision about which build path applies to a Study Location.

## Decision

Use a **Baseline Build Plan** as the interface for selecting the SFINCS baseline
build path.

The plan owns:

- Study Location name
- regular-grid versus wave-coupled build kind
- Hydrodynamic Truth Set versus Wave-Coupled Truth Set classification
- notebook path
- base model output root
- grid footprint source
- required Source Artifacts for the build path

The notebook template should load runtime config, build the plan, display the
summary rows, and then run the selected build cells.

## Consequences

- Marshfield keeps using the wave-coupled notebook while ADR-0001 remains in
  force.
- Inland or surge-only Study Locations can use the regular-grid notebook
  without editing notebook logic.
- Future source requirements for baseline builds can be added to the plan module
  and tested without running HydroMT or SFINCS.
