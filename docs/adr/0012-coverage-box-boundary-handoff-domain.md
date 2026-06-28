# Coverage Box Boundary Handoff Domain

## Status

Accepted.

## Context

Austin and Greensboro need Wflow-coupled inland flood modeling over SMART-DS
evaluation regions. Earlier notebook language treated the SFINCS region as if it
should be selected by reviewed gage count or shaped like a watershed. That
creates clipped or misleading domains: SFINCS is the hydraulic coverage model,
while Wflow is the upstream hydrologic routing model.

The Wflow model may need to be much larger than the SMART-DS region so that
flow direction, elevation, upstream area, and routed hydrographs are physically
coherent. Boundary forcing for SFINCS should enter where a Wflow stream crosses
the SFINCS coverage boundary, not where the reviewed USGS gage happens to sit.

## Decision

Use **Coverage Box + Boundary Handoff** for Austin and Greensboro.

- SFINCS domains are coverage boxes around SMART-DS Evaluation Footprints or
  reviewed evaluation components. They are not required to be watersheds.
- Wflow submodels are outlet-delineated HydroMT-Wflow `subbasin` regions built
  from reviewed handoff outlet gages and reviewed drainage-area `uparea`
  evidence.
- HydroMT-Wflow `setup_basemaps` and `setup_rivers` derive the authoritative
  Wflow LDD, river mask, and `staticgeoms/rivers.geojson` from the local
  DEM/LDD basemap.
- SFINCS discharge source points are placed at upstream stream intersections
  with the SFINCS coverage-box boundary using Wflow-native river geometry.
- NHDPlus/3DHP polygons and lines are review/preflight evidence and fallback
  geometry only when Wflow-native river geometry is unavailable.
- Ambiguous or missing stream-boundary intersections are review-required; the
  workflow should not hard-force a nearest boundary point.

## Consequences

- `01_region_setup` writes the SMART-DS SFINCS coverage bbox separately from
  the larger Wflow/static collection review envelope.
- `02_collect_sources` no longer selects a primary SFINCS modeling region by
  reviewed gage count.
- `04/build_coupled_model.ipynb` builds or reuses Wflow submodels first to get
  HydroMT-Wflow-native river geometry, writes SFINCS boundary handoff sources,
  then rebuilds Wflow gauges at those boundary source points.
- Austin remains review-required until a reviewed streamgage network with
  handoff outlets exists; no fake bbox watershed should be created. Its SFINCS
  domain set is split by SMART-DS AUS subregion into `austin_p1r`,
  `austin_p1u`, `austin_p2u`, `austin_p3u`, `austin_p4u`, and `austin_p5u`.
- Greensboro models only the selected `greensboro_east` SFINCS coverage box;
  its larger encompassing Wflow HUC basin still retains the accepted gages
  inside that basin and feeds SFINCS through stream-boundary handoff sources.
