# Marshfield SST Transposition Region

The Marshfield AORC SST transposition region should not be a hand-drawn
convenience polygon. For production, use a documented candidate region and a
homogeneity check.

## Recommendation

Use an Atlas-14-informed candidate region for smoke testing and diagnostics,
then finalize only after checking the AORC storm sample. The current generated
candidate uses counties in NOAA Atlas 14 Volume 10 states whose centroids are
within 250 km of the Marshfield SFINCS grid centroid. This is reproducible and
traceable, but it is still a candidate because it includes some inland counties.
The AORC homogeneity check should either confirm that this broader region is
acceptable or reduce it to a coastal southern New England subset.

This is the defensible starting point because NOAA Atlas 14 Volume 10 uses a
regional frequency-analysis framework for Connecticut, Maine, Massachusetts, New
Hampshire, New York, Rhode Island, and Vermont. Marshfield's important rainfall
generators include coastal lows, tropical-remnant rain, frontal systems, and
warm-season convection. A larger domain increases storm count, but it also risks
mixing inland, orographic, and coastal precipitation regimes that are not freely
transposable to a low coastal Massachusetts SFINCS domain.

## Required Checks

Before production Direct AORC SST collection:

1. Build a candidate polygon from traceable boundaries, such as coastal
   counties, coastal climate divisions, or an AORC-derived similarity mask.
2. Compute AORC 72-hour annual/block maxima over the Marshfield grid footprint
   and over candidate subregions.
3. Compare storm seasonality, 72-hour depth distributions, and storm centroids
   across the candidate domain.
4. Run a sensitivity check with at least one smaller coastal-only domain and one
   wider southern New England domain.
5. Store the selected polygon, the diagnostics, and the rejected alternatives as
   source artifacts.

## Current Status

The first AORC diagnostic shows the 250 km candidate is too broad. Use the
100 km Atlas-14-informed candidate for direct AORC SST collection and
multi-year AORC diagnostics. See
`docs/design_events/methodology/marshfield_aorc_homogeneity_results.md`.

## Reference Basis

- USACE HEC-HMS SST guidance says the transposition domain should be a
  meteorologically homogeneous region, large enough to expand the storm sample,
  but not so large that the homogeneity assumption is stretched.
- USACE also recommends AORC for storm catalog development because it provides
  hourly, high-resolution precipitation over a long period of record.
- NOAA Atlas 14 Volume 10 documents precipitation-frequency methods and regional
  frequency-analysis ideas for the northeastern states, including Massachusetts.
- Recent SST literature identifies transposition-domain selection as a major
  source of subjectivity and supports objective homogeneity testing.
