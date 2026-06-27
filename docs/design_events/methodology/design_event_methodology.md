# Design Event Methodology

This workflow builds a flood-resilience scenario ensemble, not a single
canonical 100-year map. Each row in the Event Catalog must say which driver
sources were used, how driver members were paired, and how the row should be
weighted. Return-period language must distinguish pre-SFINCS driver frequency
from post-SFINCS flood-response frequency: catalog rows carry a Coastal Driver
Return Period; flood depths, exposure, damages, and resilience summaries use
Flood Annual Exceedance Probability after hydrodynamic response is evaluated.

## Staged framework

Use the Event Catalog as a staged stochastic compound-flood scenario ensemble:

1. Build source records and manifests for each Study Location.
2. Fit or assemble driver marginals from historical/reanalysis records.
3. Construct event members with explicit dependence rules and provenance.
4. Add scenario families such as fixed mean-sea-level shifts after stationary
   event-member construction.
5. Simulate hydrodynamic response with SFINCS/SnapWave.
6. Report annual-chance benchmark slices and weighted stochastic summaries
   separately.

This is the defensible claim for community presentation: the ensemble is
grounded in observed or reanalysis source records, published marginal and
compound-event methods, explicit driver-pairing assumptions, and hydrodynamic
response simulations. Use standard flood-risk communication where the evaluated
products support it: 10%, 2%, 1%, and 0.2% annual-chance checkpoints correspond
to 10-, 50-, 100-, and 500-year return periods. Before SFINCS, those
checkpoints describe the coastal water-level driver; after SFINCS, response
metrics can be summarized by Flood Annual Exceedance Probability. The workflow
is not a FEMA regulatory JPM/JPM-OS flood-frequency study, and it is not yet a
fitted vine-copula joint AEP model. Those are valid future upgrades when the
data volume, event-rate model, storm parameterization, response-surface
validation, and uncertainty analysis justify the extra complexity.

The closest published analogue for the full ordering is Olbert et al. (2023):
individual driver analysis, dependence assessment, multivariate scenario
generation, hydrodynamic modelling, and inundation mapping. For Marshfield, use
that as the reference pattern, while keeping the present workflow simpler and
more transparent.

## Driver sources

- Coastal water levels: use CORA for Atlantic and Gulf study locations where it
  is available. CORA provides more than 40 years of water-level, wave, and
  atmospheric fields and is intended to support frequency analysis between tide
  gauges; Rose et al. (2024) provide the current validation reference for CORA
  water levels. For Pacific locations such as SFO, use NOAA CO-OPS tide-gage
  water levels until a comparable CORA source exists there.
- Coastal waves: for wave-coupled Study Locations, use the same historical
  coastal analog to supply both the water-level hydrograph and SnapWave boundary
  forcing. This preserves observed water-level/wave timing without fitting a
  fragile high-dimensional dependence model; Hawkes et al. (2002) and Masina et
  al. (2015) are the water-level/wave joint-probability references to cite when
  explaining why the joint matters.
- Rainfall: use a direct AORC stochastic-storm-transposition scan over a
  documented transposition region. The production collector opens AORC yearly
  Zarr groups, computes rolling storm depths, declusters top/POT events, and
  writes `storm-stats.csv`, `ranked-storms.csv`, rainfall members, and selected
  event-window NetCDFs. StormHub remains a methodological reference for SST,
  not a runtime dependency. Wright, Smith, and Baeck (2014) and Wright, Yu, and
  England (2020) are the main SST references.
- Streamflow: prefer USGS gage data where a relevant gage exists; otherwise use
  documented NWM streamflow for the location. Record the source in the Event
  Catalog rather than hiding it in the run folder.
- Antecedent soil moisture: use NWM retrospective hydrologic states to initialize
  simple infiltration treatments. This keeps antecedent wetness explicit instead
  of burying it in one fixed loss parameter; Pathiraja, Westra, and Sharma
  (2012) support the importance of antecedent moisture in design flood
  estimation.

## Marginal and compound sampling

1. Build each driver marginal from historical or reanalysis data with source
   manifests.
2. Use POT/extreme-value fits for coastal peaks after detrending; keep threshold
   and model sensitivity outputs as review artifacts.
3. Use direct AORC SST rainfall members to retain observed storm structure while
   expanding the set of plausible locations over the study domain.
4. Combine drivers through an explicit forcing pairing policy. The current
   Marshfield policies are same-historical-analog wave pairing, seasonal-window
   rainfall pairing, and 24-hour antecedent soil-moisture pairing relative to
   the paired rainfall member. Fully independent rainfall permutation remains
   useful as a baseline sensitivity case.
5. Keep an empirical body and oversample the fitted tail for resilience
   evaluation. The empirical body preserves observed common-storm behavior;
   the enriched tail gives the Probability Catalog enough support in damaging
   low-probability states. Write `sampling_weight` so the proposal enrichment
   remains auditable, and write `probability_weight` so post-SFINCS annualized
   summaries can be computed without treating the simulation budget as natural
   frequency.
6. Split the handoff into two products. The **Probability Catalog** remains the
   full weighted ensemble for expected outage hours, expected load unserved,
   expected islanding operations, annual damage, and response exceedance
   curves. The **Resilience Stress/Training Set** is a consequence-enriched
   subset for high-fidelity SFINCS and dynamic microgridding, selected around
   10-, 50-, 100-, and 500-year coastal-driver benchmarks, rainfall-heavy SST
   members, wet antecedent soil states, wave-sensitive coastal analogs, first
   wetting of grid/road/critical-load assets, and near-threshold islanding
   decisions.

For wave-coupled coastal locations, water level and wave forcing must not be
assembled independently. Use the Historical Coastal Analog required by ADR-0001:
the same storm window supplies both coastal water-level and wave time series.
Rainfall and antecedent wetness can then be attached through a documented
seasonal or conditional policy. Jane et al. (2020), Maduwantha et al. (2024),
and Naseri and Hummel (2022) are useful references for explaining dependence,
copula options, and why driver-by-driver return periods are not enough for
compound flooding.

For coastal rainfall/NTR dependence, do not change the co-occurrence pairing
window from the production default without a citation or a written
location-specific validation record. The current Marshfield default is 3 days
(72 hours), following Maduwantha et al. (2026), who condition on NTR or RF POT
events, select the partner maximum within a 3 d window, and report that this
window generally captured both peaks in their observed event checks. Shorter
windows, such as 24-36 hours, are useful sensitivity diagnostics for detecting
stale timestamps or unrelated partner peaks, but they are not the production
method unless separately justified.

Within that 3-day paired-observation sample, assign synthetic RF/NTR lag by
conditional empirical analogue sampling rather than a fitted lag distribution:
stratify by storm type when supported, draw a weighted local analogue using peak
NTR, peak rainfall, and season, and inherit that analogue's observed peak lag.
This follows the field-preserving logic of Maduwantha et al. (2026): use the
copula for magnitude extrapolation, but keep timing and forcing-shape attributes
anchored to observed events. A parametric conditional lag model is reserved for a
documented fallback if reuse/ESS diagnostics show that the observed analogue pool
is too sparse.

## Return-period and AEP language

Annual exceedance probability is the common comparison axis across Marshfield,
Austin, SFO, and Greensboro, but it is not a claim that each location floods by
the same mechanism. Each location gets its own marginal fit, source
availability, and driver recipe. The shared severity bands (`mild`, `common`,
`significant`, `rare`, `extreme`, `beyond_design`) are probability bands; the
local data decide which water level, rainfall, streamflow, and soil-moisture
states fall into those bands.

Use standard flood-risk checkpoints for communication: 10%, 2%, 1%, and 0.2%
annual chance, also described as 10-, 50-, 100-, and 500-year return periods.
In the Event Catalog these are Coastal Driver Return Periods from the fitted
coastal water-level marginal. They should be described as "a 1% annual-chance
coastal water-level driver paired with plausible compound forcings," not as "a
1% annual-chance flood map." After SFINCS/SnapWave runs, evaluate flood depth,
extent, asset exposure, damage, disruption, and resilience metrics against the
same annual-chance checkpoints. At that point the project may report Flood
Annual Exceedance Probability for response metrics.

Use benchmark slices for comparability, for example 10-, 50-, 100-, and
500-year checkpoints. For Marshfield, the active coastal-driver design ceiling
is 500 years so the 0.2% annual-chance checkpoint is in scope. Use the weighted
Probability Catalog for resilience evaluation because asset failures often
depend on storm shape, antecedent wetness, rainfall placement, driver timing,
and the full spatio-temporal SFINCS output, not just a maximum flood-depth map
or a single marginal return-period label. Expected annual metrics such as
Average Annual Outcome must be computed from post-SFINCS response metrics and
response-level Probability Weights; body/tail `sampling_weight` alone is not
enough.

## Joint-model boundary

Do not describe the current Event Catalog as a JPM, JPM-OS, vine-copula model,
or regulatory flood-frequency study. FEMA coastal statistical guidance and
Yang, Paramygin, and Sheng (2019) are the right references when explaining what
JPM/JPM-OS would require: storm parameter distributions, response simulations or
response surfaces, event rates, and probability integration for flood elevations.

Vine copulas and Bayesian copulas are credible research upgrades for dependence
modelling, but they should become separate scenario/model families after the
location has enough paired extremes and a validation plan. Until then, the
preferred low-complexity path is empirical/coherent pairing: same-storm coastal
analogs for water level and waves, seasonal-window pairing for rainfall,
antecedent hydrologic-state pairing for soil moisture, explicit weights, and
sensitivity cases.

## Seasonal-window pairing

Seasonal-window pairing constrains the shuffle without claiming a fully fitted
joint distribution. A coastal event whose historical template peaked in winter
should preferentially draw rainfall members from the same part of the year,
because storm type and rainfall intensity are seasonally structured. The current
rainfall policy samples members whose day-of-year is within a configured
window, defaulting to 45 days, and records the selected member time plus the
window width in the Event Catalog.

Antecedent soil moisture is then paired conditionally on the selected rainfall
member, not as a second independent seasonal shuffle. For Marshfield the catalog
selects the nearest NWM soil-moisture state 24 hours before the rainfall member
time and records both the reference time and lag. This keeps antecedent wetness
tied to the rainfall realization while avoiding a claim that a full
rainfall-runoff joint distribution has been fitted.

This is a realism constraint, not a guarantee that every compound event has the
correct joint exceedance probability. Keep named benchmark annual-chance slices
for comparability, report weighted stochastic summaries for risk only after
Probability Weights are defined, and add stronger dependence models only as
separately named scenario families once the data support them.

## Audit rules

- Every Event Catalog row should include source, member id/file, pairing policy,
  pairing seed, sampling region, sampling weight, and infiltration treatment.
- Tail-enriched counts are for coverage of damaging states, not natural event
  frequency. A defensible Tail-Enriched Design Ensemble must include both
  empirical-body and fitted-tail rows. Use `sampling_weight` to audit the
  body/tail design enrichment, and use Probability Weight for expected
  annualized or distributional flood-response summaries.
- Report both unweighted and weighted event distributions. Unweighted plots show
  how much model budget was spent in the tail; probability-weighted plots show
  the implied distribution after correcting for the full sampling proposal.
- Compare forcing variables both marginally and jointly before SFINCS handoff:
  coastal water-level return period, wave height/period/direction, rainfall
  depth, antecedent soil moisture, and any active streamflow driver.
- Compare flood-response variables after SFINCS using the same annual-chance
  checkpoints: depth, extent, exposure, damage, disruption, and resilience
  metrics.
- Validation historical events should stay distinct from design and stochastic
  members so calibration and stress-testing do not leak into each other.
- SFINCS run manifests should copy the Event Catalog row and source-artifact
  references used for the run.

## Reference basis

- [Olbert et al. (2023), *Journal of Hydrology*, doi:10.1016/j.jhydrol.2023.129383](https://doi.org/10.1016/j.jhydrol.2023.129383):
  combined statistical and hydrodynamic modelling of compound coastal flooding;
  closest published reference pattern for the staged workflow.
- [Zscheischler et al. (2018), *Nature Climate Change*](https://www.nature.com/articles/s41558-018-0156-3):
  compound-event framing and the risk of underestimating hazards when drivers
  are assessed one at a time.
- [Wahl et al. (2015), *Nature Climate Change*, doi:10.1038/nclimate2736](https://doi.org/10.1038/nclimate2736):
  U.S. storm-surge and heavy-precipitation co-occurrence.
- [Hawkes et al. (2002), *Journal of Hydraulic Research*](https://eprints.lancs.ac.uk/id/eprint/19269/):
  joint probability of waves and water levels for coastal defence design.
- [Masina, Lamberti, and Archetti (2015), *Coastal Engineering*, doi:10.1016/j.coastaleng.2014.12.010](https://doi.org/10.1016/j.coastaleng.2014.12.010):
  copula-based joint probability of water levels and waves for coastal flood
  hazard estimation.
- [Jane et al. (2020), *Natural Hazards and Earth System Sciences*, doi:10.5194/nhess-20-2681-2020](https://doi.org/10.5194/nhess-20-2681-2020):
  multivariate statistical modelling of compound flood drivers, including
  marginal extremes, dependence checks, and joint-probability framing.
- [Maduwantha et al. (2024), *Natural Hazards and Earth System Sciences*, doi:10.5194/nhess-24-4091-2024](https://doi.org/10.5194/nhess-24-4091-2024):
  multivariate compound-flood analysis with mixed storm types.
- [Maduwantha et al. (2026), *Hydrology and Earth System Sciences*, doi:10.5194/hess-30-401-2026](https://doi.org/10.5194/hess-30-401-2026):
  probabilistic boundary-condition generation using RF/NTR POT pairs, a 3 d
  partner-maximum pairing window, observed hydrograph/rainfall-field analogs,
  and observed lag times between peak NTR and peak basin-average RF.
- [Woods Hole Group, *Marshfield Long-Term Coastal Resiliency Plan*](https://www.nantucket-ma.gov/DocumentCenter/View/47237/Marshfield-Long-Term-Coastal-Resiliency-Plan---Final):
  local Massachusetts reference for communicating annual-chance flood
  checkpoints, flood depth, exposed buildings, losses, and average annual loss
  by planning horizon.
- [Woods Hole Group, *Massachusetts Coast Flood Risk Model FAQ*](https://truro-ma.gov/DocumentCenter/View/571/Woods-Hole-Group-Massachusetts-Coast-Flood-Risk-Model-PDF):
  Massachusetts coastal-flood-risk reference for Coastal Flood Exceedance
  Probability and return-period language.
- [Naseri and Hummel (2022), *Journal of Hydrology*, doi:10.1016/j.jhydrol.2022.128005](https://doi.org/10.1016/j.jhydrol.2022.128005):
  Bayesian copula framework for coastal-pluvial compound flood risk.
- [Wright, Smith, and Baeck (2014), *Water Resources Research*, doi:10.1002/2013WR014224](https://doi.org/10.1002/2013WR014224):
  stochastic storm transposition as a flood-frequency framework using spatial
  rainfall fields.
- [Wright, Yu, and England (2020), *Journal of Hydrology*, doi:10.1016/j.jhydrol.2020.124816](https://doi.org/10.1016/j.jhydrol.2020.124816):
  review of stochastic storm transposition, including rainfall/flood frequency
  analysis, regionalization, and design-storm applications.
- [Dewberry StormHub technical summary](https://stormhub.readthedocs.io/en/latest/tech_summary.html):
  reference implementation for SST concepts; not used as a runtime package in
  this workflow.
- [NOAA CORA](https://tidesandcurrents.noaa.gov/cora.html):
  coastal reanalysis water-level and wave source for Atlantic/Gulf study
  locations.
- [Rose et al. (2024), *Frontiers in Marine Science*, doi:10.3389/fmars.2024.1381228](https://doi.org/10.3389/fmars.2024.1381228):
  assessment of 43 years of NOAA CORA water levels for Gulf and East Coast
  locations.
- [NOAA CO-OPS data API](https://api.tidesandcurrents.noaa.gov/api/dev):
  tide-gage observation source for locations outside CORA coverage, including
  SFO.
- [NOAA AORC on AWS](https://registry.opendata.aws/noaa-nws-aorc/):
  hourly gridded meteorological forcing used by the Direct AORC SST Collection
  and by NWM retrospective simulations. The confirmed AORC v1.1 S3 bucket is
  `noaa-nws-aorc-v1-1-1km`, organized as yearly Zarr groups such as
  `s3://noaa-nws-aorc-v1-1-1km/2022.zarr`.
- [NOAA National Water Model overview](https://water.noaa.gov/about/nwm):
  source context for streamflow and surface/near-surface hydrologic states.
- [NOAA National Water Model retrospective on AWS](https://registry.opendata.aws/nwm-archive/):
  retrospective streamflow and land-surface output archive used for historical
  hydrologic context. Version 3.0 spans February 1979 through January 2023, so
  the current ensemble window aligns CORA, direct AORC SST, and NWM on
  1979-02-01 through 2022-12-31. Confirmed CONUS Zarr groups are
  `s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr` for streamflow
  and `s3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr` for land
  surface states including `SOIL_M`.
- [Pathiraja, Westra, and Sharma (2012), *Water Resources Research*, doi:10.1029/2011WR010997](https://doi.org/10.1029/2011WR010997):
  role of antecedent moisture in design flood estimation.
- [USGS Bulletin 17C](https://www.usgs.gov/publications/guidelines-determining-flood-flow-frequency-bulletin-17c):
  U.S. guidance for flood-frequency estimates from streamgage records.
- [Beck and Zuev (2015), *Rare Event Simulation*, doi:10.48550/arXiv.1508.05047](https://doi.org/10.48550/arXiv.1508.05047):
  rationale for oversampling rare regions and carrying weights for probability
  summaries.
- [FEMA (2023), *Guidance for Flood Risk Analysis and Mapping: Coastal Statistical Simulation Methods*](https://www.fema.gov/sites/default/files/documents/Coastal_Statistical_Simulation_Methods_Nov_2023.pdf):
  reference boundary for regulatory-style coastal statistical simulation,
  including JPM, EST, and Monte Carlo methods.
- [Yang, Paramygin, and Sheng (2019), *Natural Hazards*, doi:10.1007/s11069-019-03807-w](https://doi.org/10.1007/s11069-019-03807-w):
  objective JPM-OS method for probabilistic coastal inundation hazards.

## Location Exceptions

- Marshfield baseline: streamflow is marked unavailable. Nearby small NWM
  reaches exist, but there is no meaningful streamflow boundary or gage record
  for the coastal SFINCS grid, so streamflow should not be treated as a compound
  flood driver here. NWM remains useful for antecedent soil moisture.
