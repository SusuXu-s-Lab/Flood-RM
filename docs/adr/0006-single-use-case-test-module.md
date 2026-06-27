# Single-Use-Case Test Module

## Status

Accepted.

## Context

Before cluster submission, the workflow needs one local SFINCS check that uses
the same configured Study Location, baseline model, Event Catalog, and scenario
staging code as the batch run. This test should not write into the normal batch
scenario folders.

When possible, the selected event should be an actual historical extreme event
from the last 20 years. If the Event Catalog does not contain such a row, the
module may fall back to the most extreme recent historical-template proxy, and
then to the first catalog row.

## Decision

Use a **Single-Use-Case Test Plan** as the interface for the pre-cluster local
test.

The plan owns:

- selected event id
- selection reason
- isolated scenario, run-stage, run-output, and stats folders
- required inputs
- commands for scenario creation, dry run staging, actual run, and stats

Default event selection is:

1. most extreme actual historical Event Catalog row within the last 20 years
2. most extreme recent historical-template proxy
3. first Event Catalog row

An explicit event id overrides this selection.

## Consequences

- The local pre-cluster check is reproducible and inspectable.
- The check does not overwrite batch scenario folders.
- The notebook can show why a particular event was selected.
- Marshfield can still be tested end to end even before the Event Catalog has
  actual historical rows.
