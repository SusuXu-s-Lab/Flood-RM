# Staged Stochastic Scenario Framework

## Status

Accepted.

## Context

Marshfield needs a community-defensible compound-flood ensemble that includes
coastal water level, wave forcing, rainfall, and antecedent wetness while still
using standard flood-risk language. Woods Hole Group's Marshfield coastal
resilience work and the Massachusetts Coast Flood Risk Model communicate flood
hazard with annual-chance / return-period checkpoints, expected losses, and
planning horizons. This project should be equally plain, but it must not call
an input-driver return period a flood-map return period before hydrodynamic
response has been simulated.

JPM/JPM-OS and vine-copula models are credible methods, but they require a
stronger joint event-rate model, storm parameterization, response-surface
validation, and uncertainty analysis than the current location workflow has
committed to.

## Decision

Present the Event Catalog as a **Staged Stochastic Scenario Framework**:
source records and manifests first, driver libraries and marginals second,
dependence-aware member construction third, scenario shifts fourth, SFINCS
response simulation fifth, and weighted resilience summaries last.

Use **Coastal Driver Return Period** for pre-SFINCS catalog rows. Use
**Flood Annual Exceedance Probability** only for post-SFINCS flood-response
outputs such as depth, extent, asset exposure, damage, service disruption, and
resilience metrics. Use **Benchmark Design Slices** for clear public
communication at standard annual-chance checkpoints such as 10%, 2%, 1%, and
0.2% AEP. Keep the **Probability Catalog** as the full weighted ensemble for
annualized flood/power summaries. Add a separate **Resilience Stress/Training
Set** for scarce high-fidelity SFINCS and dynamic-microgrid simulations,
enriched around consequence thresholds rather than dominated by mild rows. Use
**Average Annual Outcome** summaries only after probability weights are defined
for the evaluated response. Reserve JPM/JPM-OS or vine-copula claims for
separately named future model families.

## Consequences

- The method is not heuristic-drawn: every row must trace to observed or reanalysis source records, fitted marginals or SST members, an explicit Forcing Pairing Policy, and a SFINCS response.
- Marshfield can cite compound-flood, joint-probability, SST, CORA, JPM, and
  Massachusetts coastal-flood-risk practice while being honest that this
  workflow is a scenario ensemble, not a FEMA flood-map production method.
- A 100-year catalog row means a 1% annual-chance coastal water-level driver
  until SFINCS response has been evaluated; it is not yet a 1% annual-chance
  flood map, asset impact, or resilience outcome.
- Marshfield sampling extends the design-driver ceiling to the 500-year
  coastal-driver benchmark so the 0.2% annual-chance checkpoint is available
  before SFINCS response evaluation.
- The high-fidelity stress/training budget should include some nuisance/mild
  events but concentrate on rare coastal drivers, rainfall-heavy SST members,
  wet antecedent states, wave/overtopping-sensitive analogs, first-wet grid
  assets, and near-threshold islanding decisions.
- Weighted expected annual summaries require response-level probability
  weights, not just a count of how SFINCS simulation budget was allocated.
- Future upgrades can add a vine copula, conditional extremes model, or JPM-OS response surface without renaming the current Event Catalog or weakening its provenance.
