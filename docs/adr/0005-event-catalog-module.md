# Event Catalog Module

## Status

Accepted.

## Context

The Event Catalog workflow combines synthetic coastal event members with
configured rainfall, streamflow, and soil-moisture member tables. The builder
already writes and audits the catalog, but callers had to infer which files,
pairing policies, required forcings, and wave-coupling constraints apply.

For wave-coupled Study Locations, ADR-0001 requires non-historical events to
use the same historical analog for both water-level forcing and SnapWave
Boundary Forcing.

## Decision

Use an **Event Catalog Plan** as the interface for building an Event Catalog.

The plan owns:

- Study Location name
- scenario name
- event-summary input
- event-member input
- Event Catalog and audit outputs
- configured forcing-member tables
- per-forcing pairing policies
- audit-required forcings
- wave analog policy

The notebook template should load runtime config, build the plan, display
summary rows, then call `build_event_catalog(config, paths)`.

## Consequences

- Event Catalog notebooks stay location-agnostic.
- Tests can verify pairing inputs without reading large datasets.
- Required forcings flow into the catalog audit through one module seam.
- The wave-analog requirement is visible before SFINCS runs are prepared.
