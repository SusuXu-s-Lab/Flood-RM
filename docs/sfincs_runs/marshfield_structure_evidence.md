# Marshfield SFINCS Structure Evidence

This note records the evidence hierarchy for Marshfield SFINCS structure
layers. It is not a design deliverable; surveyed/as-built geometry and
hydraulic ratings still govern any design or regulatory use.

## Source Hierarchy

1. WHG/USACE/Town plan sheets, final CAD/as-built drawings, and certified
   survey provide the design-grade crest elevations and as-built geometry for
   each structure reach.
2. Woods Hole Group and USACE design/resilience studies provide design-grade
   parameters only where they include surveyed elevations, sections,
   overtopping controls, or project geometry.
3. Green Harbor River tide-gate, sluiceway, and hydraulics studies for
   drainage-structure behavior.
4. Town engineering, DPW, Harbormaster, and historic plan sheets for jetties,
   seawall repairs, nourishment templates, and navigation-project geometry.
5. Massachusetts Coastal Infrastructure Inventory and Assessment records,
   including MassGIS/MORIS linework, structure type, material, condition,
   visible-height bands, `PositionZ`, photographs, and document references.

## Current Model Decision

Use the MassGIS/CZM public shoreline stabilization inventory as the active GIS
baseline for Marshfield **SFINCS Structure Layer** candidates. The old
`marshfield_sfincs_structures_screening/` package has been deleted after
generating and verifying MassGIS-derived source files. MassGIS supersedes the
hand-screening geometry; WHG/USACE/Town plan sheets supersede MassGIS for
design-grade crest elevations and as-built geometry.

Derived local files:

- Raw evidence:
  `locations/marshfield/data/static/structures/evidence/massgis_public_structures_2015_marshfield.geojson`
- Evidence table:
  `locations/marshfield/data/static/structures/evidence/massgis_public_structures_2015_marshfield_summary.csv`
- SFINCS source weirs:
  `locations/marshfield/data/static/structures/sources/weirs_marshfield_massgis_public_2015.geojson`
- SFINCS source thin dams:
  `locations/marshfield/data/static/structures/sources/thin_dams_marshfield_massgis_public_2015.geojson`
- Derivation summary:
  `locations/marshfield/data/static/structures/sources/massgis_public_2015_sfincs_derivation_summary.json`

Apply MassGIS-derived weirs and thin dams:

- Weirs represent overtoppable oceanfront seawall and road-crest overflow
  controls where the DEM/grid may under-resolve crests. The converter uses
  MassGIS `PositionZ` as a provisional elevation and writes `z` in meters for
  HydroMT-SFINCS. Design-grade `z` values must come from WHG/USACE/Town plan
  sheets, final CAD/as-builts, or certified survey.
- Thin dams represent MassGIS `Groin/ Jetty` features where grid leakage could
  otherwise short-circuit flow paths.
- `PrimaryHei` is retained as visible-height evidence, but it is not used as
  a crest elevation because MassGIS defines it as estimated visible structure
  height, not a continuous top-of-structure survey.

Drop drainage structures by default:

- Tide gates and culverts are omitted from `locations/marshfield/config.yaml`.
- They are not applied to SFINCS until invert elevation, hydraulic dimensions or
  rating, and upstream-to-downstream orientation are confirmed.

## Design-Grade Gap

We are closer to design-grade than the original hand-screening package because
the active linework and classification now come from an authoritative public
inventory with structure IDs, material, condition, height bands, FEMA zone, and
inspection notes. We are not design-grade yet.

Required before design/regulatory use:

- WHG/USACE/Town plan-sheet or as-built crest polyline/top outshore edge for
  each weir reach.
- Crest elevations in a confirmed vertical datum, preferably NAVD88, sampled
  along the crest rather than a single lidar-derived `PositionZ` attribute.
  These should be taken from WHG/USACE/Town plan sheets, final CAD/as-builts,
  or certified survey.
- As-built or permit drawings for repaired/raised reaches, especially Brant
  Rock, Fieldston, Rexhame, Bay Avenue, and Green Harbor.
- DEM/topobathy reconciliation showing whether a SFINCS structure is needed or
  whether terrain correction already resolves the crest/jetty.
- For tide gates and culverts: invert elevations, opening dimensions, gate
  operation/orientation, loss coefficients or rating curves, and upstream-to-
  downstream flowline order.
- QA plots comparing MassGIS-derived lines, SFINCS grid resolution, DEM crest
  elevations, and known overtopping/overflow controls from the reports.

## Evidence Targets Pursued

- MassGIS/CZM public structures, 2015 update: pulled 37 Marshfield public
  features. The active converter produced 25 weir features from seawall/
  revetment records with `PositionZ` and 6 thin-dam features from groin/jetty
  records. Six records were omitted from SFINCS source files because they were
  non-structure beach records or lacked a usable weir elevation.
- MassGIS metadata: confirms `PrimaryHei` is estimated visible height in feet,
  `PositionZ` is a lidar elevation field, and the data are approximate/planning
  inventory data rather than legal/design survey.
- Woods Hole Group Bay Avenue/Gurnet Road plans: contain on-the-ground October
  2023 survey information, NAVD88 vertical datum, concrete seawall elevations
  near 15.5-16.0 ft, nourishment berm crest elevation 8 ft, and final/as-built
  survey requirements.
- USACE/Town Brant Rock and Fieldston report: useful for documented overflow
  controls, seawall/revetment project reaches, and the Brant Rock cobble berm
  concept; it is not a full structure survey.
- Woods Hole Group Marshfield Long-Term Coastal Resilience Plan: useful for
  overtopping transect context and why seawall/runup representation matters in
  sea-level-rise simulations; it explicitly treats wave overtopping and seawall
  failure as a major risk pathway.
- USACE Green Harbor navigation-project page and Town Harbormaster historic
  plan sheets: better than MassGIS for jetty provenance and navigation-project
  geometry. Jetties remain thin-dam candidates unless topobathy/subgrid
  reliably captures the crest and slopes.
- Marshfield Green Harbor River Studies: the public document center includes
  survey points/spot elevations, tide gauge locations, tide levels, a sluiceway
  inspection report, and a hydraulics report. These are the upgrade path for
  Dyke Road tide gates, but drainage structures remain out of the base run
  until invert, opening geometry, coefficients/rating, and gate operation are
  converted into SFINCS-ready parameters.

## Structure Upgrade Plan

- Keep MassGIS-derived linework as the GIS baseline and visual inventory.
- Split long derived weir reaches into named design reaches as better evidence
  is assigned: Bay Avenue/Gurnet Road, Ocean Bluff, Fieldston, Brant Rock,
  Ocean Street overflow, Dyke Road/Town Pier overflow, Green Harbor.
- Replace provisional MassGIS `PositionZ` values with segment-specific NAVD88
  crest elevations from WHG/USACE/Town plan sheets, final CAD/as-builts, or
  certified survey.
- Treat Green Harbor jetties as thin dams for numerical flow blocking at coarse
  grid resolution, then verify against topobathy/subgrid elevations.
- Treat Dyke Road tide gates as drainage-structure candidates, not as weirs,
  but only after the Green Harbor sluiceway/hydraulics evidence is converted
  into invert elevations, dimensions, orientation, and operating assumptions.

## Useful Public Sources

- MassGIS/CZM Coastal Infrastructure Inventory and Assessment Project:
  https://www.mass.gov/info-details/inventories-of-seawalls-and-other-coastal-structures
- South Shore public coastal infrastructure metadata:
  https://maps.massgis.digital.mass.gov/czm/moris/metadata/moris_csi_public_south_shore_arc.htm
- 2015 South Shore-South inventory update record:
  https://archives.lib.state.ma.us/entities/publication/5617c747-d602-4e97-9959-9907fad201df
- USACE Brant Rock and Fieldston feasibility report:
  https://www.marshfield-ma.gov/Documents/Departments/Public%20Works/Forms%20Documents/hurricane_and_coastal_storm_damage_reduction_repor1_scm_gwrb-20160510.pdf
- Woods Hole Group Marshfield Long-Term Coastal Resilience Plan:
  https://www.nantucket-ma.gov/DocumentCenter/View/47237/Marshfield-Long-Term-Coastal-Resiliency-Plan---Final
- Marshfield Multi-Hazard Mitigation Plan appendices:
  https://www.marshfield-ma.gov/Documents/Departments/Town%20Hall/Planning/Marshfield%20Multi%20Hazard%20Mitigation%20Plan/marshfield_mhmp_draft_030323_appendices_compressed.pdf
- Marshfield Green Harbor Jetty Documents:
  https://www.marshfield-ma.gov/departments/public_safety/harbormaster/green_harbor_jetty_documents.php
- USACE Green Harbor Navigation Project:
  https://www.nae.usace.army.mil/Missions/Civil-Works/Navigation/Massachusetts/Green-Harbor/
- Woods Hole Group Bay Avenue/Gurnet Road nourishment plans:
  https://www.town.duxbury.ma.us/sites/g/files/vyhlif10506/f/pages/2018-0231-07_gurnet-bay_duxbury_11-07-2023_stamped_and_reduced.pdf
- Marshfield Green Harbor River Studies:
  https://www.marshfield-ma.gov/departments/town_hall/conservation/green_harbor_river_studies.php
