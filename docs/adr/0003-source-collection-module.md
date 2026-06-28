# Source Collection Module

## Status

Accepted.

## Context

The data-collection workflow needs to be reusable across Study Locations. A
notebook is useful as a readable runbook, but it should not own source ordering,
time-window validation, source-specific settings, or provider choices.

The codebase already has individual source adapters for CORA, NWM, Direct AORC
SST rainfall, ERA5 waves, and SSURGO-style supporting pulls. The friction is
that callers need to know which source gets which config keys and which dates
are valid.

## Decision

Use a **Source Collection Plan** as the interface for deciding which source
steps run for a configured **Study Location**.

The plan owns:

- configured source ordering
- base collection window validation
- source-specific date-window validation
- source-specific settings passed to collection adapters

The collection notebook template should load a Study Location, build the plan,
display the planned sources, then call the source collection module. Individual
source adapters remain behind the module seam.

## Consequences

- Adding a source changes the plan module and one adapter, not every notebook.
- The data-collection notebook stays educational and location-agnostic.
- Tests can verify the plan interface without network access.
- Source adapters can still vary by provider, such as CDS versus Earth Data Hub
  for ERA5 waves.
