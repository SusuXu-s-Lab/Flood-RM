# Marshfield AORC Homogeneity Results

This note records the first AORC-based diagnostics for the Marshfield
AORC SST transposition region. The goal is to avoid treating an arbitrary
polygon as meteorologically homogeneous.

## Method

- Source: AORC v1.1 yearly Zarr, `s3://noaa-nws-aorc-v1-1-1km/2018.zarr`.
- Variable: `APCP_surface`.
- Statistic: maximum rolling 72-hour precipitation depth at sample points.
- Target point: Marshfield SFINCS grid centroid.
- Candidate points: centroids of counties in NOAA Atlas 14 Volume 10 states
  selected by distance from the Marshfield SFINCS grid centroid.
- Tested regions: 100 km, 125 km, 150 km, and 250 km county-centroid radii.

## Results

| candidate | counties | target max 72h mm | min ratio | median ratio | max ratio | peak months | review |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 100 km | 12 | 76.1 | 0.81 | 1.06 | 1.57 | 1, 2, 3, 9, 11 | required |
| 125 km | 18 | 76.1 | 0.81 | 1.08 | 1.57 | 1, 2, 3, 8, 9, 10, 11 | required |
| 150 km | 20 | 76.1 | 0.74 | 1.08 | 1.86 | 1, 2, 3, 8, 9, 10, 11 | required |
| 250 km | 48 | 76.1 | 0.61 | 1.05 | 2.18 | 1, 2, 3, 4, 7, 8, 9, 10, 11, 12 | required |

## Interpretation

The 250 km Atlas-14 candidate is too broad for production. Its 2018 diagnostic
includes southwest Connecticut points with more than twice the Marshfield target
72-hour maximum, and peak months span most of the year. That is a strong warning
that the region mixes rainfall regimes.

The 100 km candidate is the best of the tested radii. It keeps the max ratio
below 1.6 in the 2018 diagnostic and stays closer to eastern Massachusetts,
Rhode Island, and nearby coastal southern New England. It is still marked
`review_required` because peak months span five months in a one-year diagnostic.

## Direct AORC SST Collector Status

- 250 km single-polygon hull: rejected for production because it mixes rainfall
  regimes in the 2018 diagnostic.
- 100 km single-polygon hull: retained as the current direct AORC SST
  transposition candidate. Production collection should run the repo-owned AORC
  SST collector over the full 1979-2022 window with checkpointed yearly stats.

## Current Recommendation

Use the 100 km Atlas-14-informed candidate for the next multi-year AORC
diagnostics. Do not use the 250 km candidate for production.

Before production, rerun diagnostics for multiple years or the full 1979-2022
period, and compare at least:

- 72-hour annual maxima ratios.
- Peak month/season distributions.
- Storm centroid distribution.
- Sensitivity to 100 km versus a coastal-only county subset.

## Artifacts

- `locations/marshfield/data/sources/aorc_sst/transposition_regions/transposition_region_100km.geojson`
- `locations/marshfield/data/sources/aorc_sst/transposition_regions/transposition_region_counties_100km.geojson`
- `locations/marshfield/data/sources/aorc_sst/homogeneity_100km/aorc_homogeneity_summary_2018.json`
- `locations/marshfield/data/sources/aorc_sst/homogeneity_100km/aorc_homogeneity_samples_2018.csv`
- `locations/marshfield/data/sources/aorc_sst/homogeneity_250km/aorc_homogeneity_summary_2018.json`
- `locations/marshfield/data/sources/aorc_sst/homogeneity_250km/aorc_homogeneity_samples_2018.csv`
