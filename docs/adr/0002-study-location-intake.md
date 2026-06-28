# Study Location Intake

## Status

Accepted.

## Context

The global workflow should start from a reusable **Study Location** description,
not from Marshfield-specific notebooks or scattered path edits. A bounding box is
useful for bootstrapping a **Grid Footprint**, but it is not enough to describe a
runnable **Flood-Resilience Scenario Ensemble**. A runnable location also needs
source settings, CRS choices, wave-coupling selection, output roots, and event
driver availability.

Marshfield remains the **Reference Study Location**. San Francisco, Austin, and
Greensboro may exist as placeholder folders while Marshfield is proven end to
end.

## Decision

Use `locations/<study_location>/config.yaml` as the external interface for a
configured **Study Location**.

Placeholder folders under `locations/` are not runnable until they contain a
`config.yaml`. Shared pipeline code may list placeholder folders, but
collection, SFINCS build, Event Catalog, and cluster stages should require a
configured location.

Notebooks should become thin templates that consume this location interface:

- data collection template: builds **Source Artifacts**
- baseline SFINCS template: builds the first **Hydrodynamic Truth Set** or
  **Wave-Coupled Truth Set**
- Event Catalog template: builds event recipes and records the **Forcing Pairing
  Policy**
- single-use-case template: runs one event before cluster submission

## Consequences

- Marshfield work can continue without inventing location-specific forks.
- Future locations start by filling in one location config rather than editing
  many notebooks.
- Empty placeholder folders do not imply acquisition readiness.
- Bbox-first workflows should write or update `config.yaml`, then call the
  normal location-aware modules.
