# Compound Flood Event Characterization: Rainfall–NTR Timing, Event Pairing, and Lag Sampling in Stochastic Design-Event Catalogues

## TL;DR
- **A ~40-hour rainfall-leads-coastal modal lag is almost certainly a pairing/sampling artifact, not a physical signal, for a nor'easter-dominated New England open-coast site like Marshfield.** Peer-reviewed hourly analyses of the closest analogous mid-latitude site (NYC; Chen et al. 2025, HESS 29:3101–3117) find rain–surge lags of only a few hours — e.g., a "2–6 h lag from the time of peak rain to peak surge" — and even the widest pairing windows in the literature (±1.5 d) cannot physically support a 40-h mode. The horizontal banding you observe (a few analogue events reused across many synthetic rows) is the diagnostic signature of the artifact.
- **Best practice is to define discrete RF and NTR events by POT + declustering (3–5 d windows), pair them within a narrow ±24–36 h window centered on the conditioned peak, model the statistical dependence on non-tidal residual (NTR, not total water level), and sample the lag conditionally** — jointly with magnitude and stratified by storm type/season — rather than as a deterministic nearest-neighbor pick or an independent marginal.
- **Move from a single nearest-neighbor draw to a weighted kNN analogue resample (Maduwantha et al. 2026 inverse-difference weighting is the direct template), and add explicit diversity diagnostics** (effective number of analogues / reuse counts, observed-vs-synthetic lag CDF, magnitude–lag dependence, sign-fraction preservation). Keep a separate deterministic lag-sweep stress test (à la Xu et al. 2024) distinct from the probabilistic catalogue.

## Key Findings

1. **Your reference paper (Maduwantha et al. 2026, HESS 30:401–420) uses exactly the architecture you are moving toward, and its sign/pairing conventions matter.** Events are identified by two-sided conditional POT sampling — the generated set provides "5,000 combinations of peak NTR and RF ... 1,000 years' worth of extreme events (5 events per year on average)." They select the max of the partner variable within a 3-day window, decluster with a 5-day window (2.5 d either side), and stratify TC vs non-TC events (TC = circulation center within 350 km within a 3-day window). To build hydrographs they randomly sample observed events with probability proportional to the **inverse of the difference** between target and observed peak (separately for RF and NTR) — a weighted-analogue draw designed expressly to avoid reusing a single nearest neighbor. The lag is then assigned by randomly picking one of the observed lag times (between peak hourly NTR and peak hourly basin-average RF) from the selected events.

2. **Empirical rain–surge lags in the literature are small — hours, not days — for extratropical/mid-latitude US coasts.** Chen et al. 2025 report lags of "a few hours," with TCs smallest and histograms displayed only within ±15 h. A 24–48 h rainfall-leading-surge mode is not supported as a physical central tendency at such sites.

3. **NTR (not total water level) is the correct variable for the statistical dependence and lag model.** Using TWL aliases the meteorological timing with the deterministic astronomical tide (the ~12.42 h M2 cycle and spring–neap modulation), which is severe at a strongly tidal site like Marshfield. Best practice (Serafin & Ruggiero 2014; Maduwantha et al. 2026) is to model dependence/lag on NTR, then recombine NTR + tide + MSL anomaly to produce the TWL hydraulic boundary condition.

4. **Lag should be sampled conditionally, not as an independent marginal.** Lag depends on magnitude (Chen et al.: rain intensity negatively correlated with absolute lag during TCs), storm type (TC < ETC < convective spread), and season. A weighted empirical kNN draw conditioned on (NTR magnitude, RF magnitude, storm class, season) is the recommended approach.

5. **Inundation is sensitive to lag, and that sensitivity should be explored with a separate deterministic stress test.** Xu et al. 2024 (HESS 28:3919–3930) found that in Shanghai "the peak rainfall occurs 2 h before the peak storm surge would cause the deepest average cumulative inundation depth ... surge is the primary flood driver." Keep this lag-sweep separate from the probabilistic catalogue.

## Details

### 1. Event definition and pairing

**Discrete event identification.** The dominant approach is peaks-over-threshold (POT) with declustering, preferred over block maxima because block maxima of two variables can occur at different times and produce spurious "simultaneous" pairs that never co-occurred (a point made explicitly in the Qiantang/Xu literature). Typical choices:
- **Thresholds** set to yield an average of ~3–5 events/yr per variable (Maduwantha et al. use ~5/yr; Jane et al. similar). Percentile thresholds (95th–99th) are common in global/regional work (Ward et al. 2018; Bevacqua et al. 2019).
- **Declustering windows** of 3 days are standard for larger-domain studies (Bevacqua et al. 2019; Ward et al. 2018; Haigh et al. 2016 found UK storms affect sea level ~3.5 days). Maduwantha et al. use 5 days; Chen et al. use 5 days ("to account for the typical maximum duration of cyclonic storm events"). The window should reflect the typical storm duration at the site.
- **Rainfall events** are often defined by a dry-spell separation (Maduwantha use a 6 h continuous dry period); the RF accumulation duration is swept (1–48 h) and the duration maximizing correlation with NTR is selected for the dependence analysis (Maduwantha; Chen et al.). Maduwantha et al. 2024 found the largest NTR–RF dependence at 18-h rainfall accumulation for the Delaware estuary catchment.

**Pairing/concurrence criteria.** Two-sided ("conditional") sampling is the field standard: condition on the primary driver being extreme, then take the partner's peak within a window (Wahl et al. 2015; Ward et al. 2018; Jane et al. 2022). The pairing window choice directly controls the lag distribution:
- Maduwantha et al. 2026: partner max within a **3-day window**.
- Chen et al. 2025: secondary driver within **±1.5 d** of the primary peak for storm-duration maxima, but only **±1 h** for the joint-return-period sampling because 90th-percentile rain drops to ~40% of peak within an hour of peak NTR.
- Daily-data studies (much of the older literature) pair on the same calendar day with no sub-daily lag, which cannot resolve timing at all.

**Warning re: wide windows and spurious lags — directly relevant to your –40 h mode.** A wide pairing window (e.g., ±3 days = ±72 h) will, whenever a clearly extreme partner peak does not exist close to the conditioned peak, "reach out" and grab a secondary, physically unrelated peak elsewhere in the storm envelope (or in an adjacent storm). This manufactures large apparent lags. The Jane et al. (ASCE J. Hydrol. Eng. 2022) sensitivity study on the Texas Gulf Coast explicitly shows compound-event statistics are "highly sensitive to the setup of the statistical model," including the time-lags considered between drivers. Your –40 h mode exceeds the central tendency of every hourly study and approaches/exceeds the pairing-window half-width itself — the classic signature of an artifact. The banding (few analogues reused) compounds this: a handful of observed storms with an incidental secondary peak ~40 h away are being repeatedly matched.

### 2. Empirical lag findings and the New England question

| Study | Site / type | Reported lag |
|---|---|---|
| Chen et al. 2025 (HESS 29:3101) | NYC, TC/ETC/Neither, hourly | Median \|abs\| lag "a few hours"; e.g., Bronx River "2–6 h lag from the time of peak rain to peak surge"; near-simultaneous at The Battery; histograms shown within ±15 h |
| Xu et al. 2024 (HESS 28:3919) | Shanghai, TC | Max inundation when "peak rainfall occurs 2 h before the peak storm surge" |
| Zheng et al. / Hawkes-lineage (J. Hydrol. 2013) | Australia | Strongest dependence at 0 lag for >6 h bursts; up to ±10 h for short bursts |
| Maduwantha et al. 2026 | Gloucester City NJ (Delaware estuary) | Lags drawn empirically from observed event pairs (hours) |

**Sign convention.** Chen et al. define lag = T_peak_surge − T_peak_rain, so positive lag = surge after rain (rain-led). Your convention (lag = t_rainfall_peak − t_coastal_peak) is the opposite sign: your –40 h means rainfall peaks 40 h *before* the coastal peak, which in Chen's convention would be +40 h (rain-led). Be explicit about this in your documentation, since the two literatures use opposite signs.

**Storm-type association radii differ by study and matter for stratification.** Chen et al. 2025 associate an event with a TC if it occurs "within 500 km of the center of a TC or within 1000 km of the center of an ETC"; Maduwantha et al. use a 350 km / 3-day TC rule. Maduwantha et al. 2024 found that, conditioning on NTR, "38 events are identified as TCs ... 43 when conditioned on RF, while the rest are non-TCs" — i.e., non-TC (ETC + convective) events dominate the sample, which is exactly the New England nor'easter regime.

**Is a ~24–48 h rain-leads-surge mode plausible for Marshfield?** No, not as a *modal* physical lag. For nor'easter/ETC-dominated open-coast New England:
- The surge and the heaviest rain are driven by the same cyclone and its fronts, typically within hours of each other; ETCs have longer duration but the *peak-to-peak* offset is still hours, not 1–2 days.
- A genuine multi-day lead would require the rainfall mechanism to be physically decoupled from the surge mechanism (e.g., a separate frontal passage well before the coastal storm), which is not the modal case.
- A multi-day lag is more characteristic of large-catchment *fluvial* discharge responding to antecedent rain (Petroliagkis et al. found a few-days surge–discharge lag for some European rivers; Kew et al. found ~6 days for the Rhine, the time to route excess precipitation to the river outlet). Marshfield is a small, flashy open-coast setting where pluvial response is fast — not a large basin.
- Therefore a –40 h mode for an *NTR–rainfall* pairing almost certainly reflects (a) a too-wide pairing window, (b) daily-resolution or mismatched peak-picking, and/or (c) reuse of a few analogue storms with incidental secondary peaks.

### 3. NTR vs total water level

The dependence structure and lag must be characterized on **NTR (the meteorological residual)**, because:
- TWL = astronomical tide + MSL anomaly + NTR (+ wave setup). The tide is deterministic and phase-locked to the moon, not to the storm. If you compute lag from TWL peaks, the "coastal peak" snaps to the nearest high tide, injecting a quasi-periodic ~12.42 h (M2) structure and a spring–neap beat into the lag distribution that has nothing to do with meteorology. At a strongly tidal site like Marshfield (large tidal range), this aliasing dominates and is a prime suspect for a spurious multi-hour-to-multi-day mode.
- Best practice (Serafin & Ruggiero 2014 TWL full-simulation model; Maduwantha et al. 2026): build the statistical/dependence/lag model on NTR; then for the hydraulic boundary condition, recombine the scaled NTR hydrograph with a sampled tide and MSL anomaly, **preserving the observed peak-NTR-to-high-tide timing offset** to retain tide–surge interaction. Maduwantha et al. show that "m.s.l. anomalies and tidal conditions alone can lead to differences in flood depths exceeding 1 and 1.2 m, respectively, in parts of Gloucester City" — i.e., the tide must be handled as a separate, explicitly sampled component, not baked into the dependence model.

### 4. Sampling methodology for stochastic catalogues

**(a) Empirical kNN / analogue resampling (recommended core).** The Lall & Sharma (1996, WRR 32:679–693) k-nearest-neighbor bootstrap is the foundational method: resample with replacement from the k nearest neighbors in a conditioning feature space, with a decreasing kernel weight by rank. Mehrotra & Sharma (2006, Adv. Water Resour.) extended it to conditional resampling on multiple predictors with feature scaling weights. Maduwantha et al. 2026 implement a closely related idea: sample observed events with probability ∝ the inverse of the distance between target and observed peak. Your planned weighted empirical kNN draw conditioned on (coastal magnitude, rainfall magnitude, storm class, season) is the right generalization. Key design choices:
- Use a sensible k (Lee & Ouarda 2011, J. Hydrol., give selection guidance; common heuristic k ≈ √n).
- Standardize/scale predictors and consider feature weights reflecting physical importance.
- Stratify by storm type *before* the draw (TC vs ETC vs convective) so analogues are drawn from the physically correct population.

**(b) Copula / parametric joint models.** Copulas (Bevacqua et al. 2017 pair-copula construction, Ravenna; Ward et al. 2018; Moftakhari et al. 2019; Couasnon et al. 2018 copula-Bayesian network, Houston) are best for the *magnitude* dependence (peak NTR, peak RF) and for extrapolating beyond observed combinations. They are weaker for *lag*, which is bounded, often multimodal, and conditionally dependent. A hybrid is best practice: copula for magnitudes (extrapolation) + empirical conditional resampling for the lag and hydrograph shapes (realism). This is essentially the Maduwantha architecture (copula for peaks; observed-event resampling for time series + lag).

**(c) Event-based vs continuous.** For a design-event catalogue, event-based is appropriate and far cheaper. Continuous simulation (long synthetic NTR+RF series) avoids event-definition artifacts but is overkill unless you need duration/clustering statistics. Note that Maduwantha et al. advocate a "response-based" approach: simulate many (5,000) events through the flood model and derive flood-depth return levels from the *response*, rather than assuming a single design event's driver-AEP equals its flood-AEP. The companion NHESS paper (Santamaria-Aguilar, Maduwantha et al. 2026, Gloucester City NJ) found compound events "with return periods less than 20 years can produce the 100-year flood depths in large areas," underscoring why a diverse catalogue (not a single design event) is needed.

**Sampling the lag variable — the central question.** Lag should be sampled **jointly with / conditioned on magnitudes and storm type**, not as an independent marginal:
- Independent marginal sampling destroys the magnitude–lag and type–lag dependence and will faithfully reproduce the artifactual –40 h mode if it is present in the empirical marginal.
- The cleanest method: within each storm-type stratum, draw the analogue event by weighted kNN on magnitudes + season, and inherit that event's *observed* lag (Maduwantha approach). This automatically conditions lag on everything.
- Alternatively, fit a conditional lag distribution (lag | magnitude, type) and sample from it, but the empirical inherit-the-analogue's-lag method is simpler and preserves realism.

**Finite-catalogue diversity / analogue reuse.** This is your banding problem. Controls:
- Use *weighted* draws (inverse-difference or kNN kernel) rather than nearest-neighbor, exactly as Maduwantha do, to spread draws across multiple analogues. They adopt weighting precisely because "selecting only the nearest event would result in utilizing a single or small number of observed events for all the nearby target scenarios, thereby restricting the diversity of the generated events."
- Compute an **effective sample size / effective number of analogues**, e.g., the inverse Herfindahl index of empirical reuse frequencies (ESS = 1 / Σ p_i², where p_i is the fraction of synthetic rows drawn from observed event i). Low ESS relative to the number of candidate analogues signals over-reuse.
- Consider a **reuse penalty** (down-weight an analogue's selection probability as it accumulates draws) or jitter/perturbation of inherited lags within a physically plausible band to add diversity without distorting the conditional structure.
- Widen the conditioning bandwidth (larger k / softer kernel) where the observed sample is sparse, so a single nearest event does not dominate a region of parameter space.

### 5. Magnitude-dependent and conditional lag structure

Evidence that lag is not constant:
- **Magnitude:** Chen et al. find a significant negative correlation between extreme rainfall intensity and absolute lag during TCs — the most intense rains have the shortest offsets to peak surge (promoting pluvial-coastal compounding).
- **Storm type:** TCs produce the smallest, most simultaneous lags; ETCs and convective events have larger and more spread lags. Spatial propagation also matters — at Kings Point peak NTR lags peak rain by hours "due to surge propagation along Long Island Sound," vs near-simultaneous at The Battery. This is why stratifying TC vs non-TC (as Maduwantha do) is essential before modeling lag.
- **Season:** storm-type mix is seasonal (TCs late summer/fall; nor'easters cool season), so season is a useful proxy that should be retained as a conditioning variable.

To preserve this in a synthetic catalogue: stratify by storm type, condition the analogue draw on magnitude and season, and inherit the observed lag — which carries the conditional structure forward automatically. Then validate (Section 7) that the synthetic magnitude–lag scatter matches the observed.

### 6. Hydraulic sensitivity to lag

Xu et al. 2024 used a "same-frequency amplification" method to build design surge and rainfall hydrographs and ran a hydrodynamic model across a sweep of relative timings (e.g., −2 h and +12 h cases), finding the deepest cumulative inundation when "peak rainfall occurs 2 h before the peak storm surge," with surge the dominant driver in Shanghai. The key best-practice point: **separate the deterministic lag-sweep stress test from the probabilistic catalogue.**
- The probabilistic catalogue answers "what is the flood-depth return level given realistic joint driver behavior?" — lag must be sampled per its conditional distribution.
- The deterministic lag-sweep answers "how sensitive is *this* design event to timing, and what is the worst-case alignment?" — fix the magnitudes, sweep the lag (e.g., −24 h to +24 h in 1–3 h steps), and report the inundation envelope. This identifies the critical timing without contaminating the probabilistic frequency estimates.

### 7. Validation / diagnostics for the catalogue

Recommended diagnostic suite:
1. **Analogue reuse / effective number of analogues:** histogram of how many synthetic rows draw from each observed event; ESS = 1/Σp_i². Target ESS well above the number of distinct observed events, with no single event dominating. This directly diagnoses your banding.
2. **Lag CDF comparison (observed vs synthetic),** within each storm-type stratum; a KS or Cramér–von Mises distance. The synthetic lag distribution should match the *observed* (hours-scale) distribution, not a wide artifactual one.
3. **Magnitude–lag dependence preservation:** scatter / rank correlation (Kendall τ) of |lag| vs RF intensity and vs NTR, observed vs synthetic — should reproduce the negative magnitude–lag correlation.
4. **Sign-fraction preservation:** fraction of rain-led vs surge-led events, by storm type, observed vs synthetic.
5. **Marginal and joint magnitude checks:** synthetic peak NTR/RF marginals and copula dependence (Kendall τ) vs observed, per stratum.
6. **Physical plausibility filters:** reject synthetic events with non-physical lags (e.g., |lag| beyond the declustering window) and scaling factors beyond a reasonable cap (Maduwantha reviewers flagged that large scaling factors can make events unrealistic).
7. **Seasonality / storm-type proportion checks:** monthly frequency and TC/non-TC split of synthetic events vs observed.

## Recommendations

**Stage 1 — Fix the artifact first (before any kNN refinement).**
- Recompute lag on **NTR peaks**, never TWL peaks, at hourly resolution. If you are currently picking the coastal peak from TWL, this alone may explain the –40 h mode at a strongly tidal site like Marshfield.
- **Tighten the pairing window** to ±24–36 h (Chen et al. use ±1.5 d for storm maxima; ±1 h for tight JRP work). Re-plot the lag histogram. If the –40 h mode collapses toward 0 ± a few hours, it was a window artifact — the expected outcome.
- Verify peak-picking: ensure you are matching the *primary* partner peak within the window, not the largest secondary peak in a multi-peak storm.
- **Benchmark:** the corrected modal lag should fall within roughly ±10–15 h, consistent with NYC hourly findings. If it does not, investigate further before proceeding.

**Stage 2 — Replace nearest-neighbor with weighted kNN analogue draw.**
- Stratify by storm type (TC via HURDAT2 radius/time rule as Maduwantha; ETC/convective otherwise).
- Within stratum, draw analogues by inverse-difference / kNN kernel weighting on standardized (NTR peak, RF peak, season). Inherit each analogue's observed lag and hydrograph shape.
- **Benchmark:** the effective number of analogues (ESS) should rise substantially vs the nearest-neighbor baseline; no observed event should supply more than a small fraction of synthetic rows.

**Stage 3 — Add diversity controls if banding persists.**
- Apply a reuse penalty and/or small lag jitter within a physically bounded band.
- Soften the kernel bandwidth in sparse regions of parameter space.
- **Threshold to change approach:** if, after weighting + penalties, ESS is still low (a handful of events dominate), the observed record is too short for empirical resampling in that region — switch to a parametric conditional-lag model or pool storm types more aggressively for that stratum.

**Stage 4 — Validate and document.**
- Run the full diagnostic suite (Section 7). Require lag-CDF, magnitude–lag correlation, and sign-fraction matches per stratum before accepting the catalogue.
- Keep a **separate deterministic lag-sweep** (fix design magnitudes, sweep −24 to +24 h) to report the inundation sensitivity envelope and critical timing, distinct from the probabilistic return-level catalogue.
- Document the sign convention explicitly (yours is opposite to Chen et al.'s).

## Caveats
- The exact per-storm-type median lag numbers in Chen et al. 2025 are embedded in their Figure 6 raster and could not be extracted as text; the verified anchor values are the Bronx River / TC "2–6 h" range and the qualitative ordering TC < ETC < "Neither" with The Battery < Kings Point. The robust conclusion (lags are hours, not days) is firmly supported.
- Marshfield-specific compound-flood lag statistics do not appear in the peer-reviewed literature; the New England inference is by analogy to NYC (Chen et al.), the Delaware estuary (Maduwantha et al.), and general nor'easter physics. Marshfield differs from NYC in being a more exposed open-coast, wave-influenced, strongly tidal site, so wave setup may matter for TWL — though it does not affect the NTR–RF lag question.
- Maduwantha et al. 2026 is applied to Gloucester City, an estuarine site where NTR blends fluvial and coastal signals; at an open-coast site like Marshfield the NTR is more purely meteorological surge, which should make the rain–surge lag *tighter*, reinforcing the conclusion that a 40-h mode is artifactual.
- The Lall & Sharma kNN bootstrap preserves only dependence present in the observed record and cannot generate genuinely unprecedented driver combinations or lags; pair it with a copula for magnitude extrapolation where tail behavior matters.
- Some sources consulted for background physical context (news reports on nor'easters; general IDW interpolation pages) are not peer-reviewed and were used only for context, not for any quantitative claim.