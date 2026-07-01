import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from coupling import domain_set as crossings
from sfincs_runs import infiltration as infiltration
from sfincs_runs import inland_base as inland_base
from sfincs_runs import plan as plan
from sfincs_runs import region_notebook as region_notebook
from sfincs_runs import static_catalog as static_catalog
from sfincs_runs import static_intake as static_intake
from sfincs_runs import structures as structures
from sfincs_runs.plan import (
    BaselineBuildPlan,
    StaticIntakePlan,
    build_baseline_build_plan,
    build_static_intake_plan,
)
from sfincs_runs.static_catalog import build_static_data_catalog
from sfincs_runs.inland_base import (
    InlandSfincsBasePlan,
    InlandSfincsDomainSetPlan,
    add_inland_outflow_boundary,
    build_inland_sfincs_base,
    build_domains,
    create_handoffs,
    is_built_sfincs_base,
    is_built_wflow_base,
    meaningful_model_files,
    plan_inland_sfincs_domain_set,
    plan_inland_sfincs_base,
    plot_sfincs_handoff_basemap,
    sfincs_grid_resolution_matches,
    sfincs_rivers_inflow_geoms,
    set_observations,
    write_inland_sfincs_domain_set_manifest,
    write_inland_sfincs_handoff_locations,
    write_inland_sfincs_handoff_locations_from_wflow_rivers,
)
from sfincs_runs.infiltration import validate_physics
from sfincs_runs.static_intake import (
    RegionSetup,
    build_region_setup,
    clip_dem_and_landcover_to_bbox,
    collect_ssurgo_infiltration_inputs,
    collect_static_region_inputs,
    collect_wflow_static_region_inputs,
    download_file,
    fetch_usgs_3dep_dem,
    fetch_worldcover_landcover,
    worldcover_tile_urls,
)
from sfincs_runs.region_notebook import (
    build_sfincs_coverage_and_wflow_preflight,
    collect_coastal_terrain_landcover_inputs,
    collect_inland_static_inputs,
    ensure_usgs_3dep_dem_covers_bbox,
    plot_coastal_static_input_qa,
    plot_static_input_qa,
)
from sfincs_runs.structures import (
    StructureLayer,
    StructurePlan,
    apply_sfincs_structures,
    derive_massgis_sfincs_structure_layers,
    plot_structure_layers,
    prepare_structure_layers,
)

__all__ = [
    "BaselineBuildPlan",
    "RegionSetup",
    "StructureLayer",
    "StructurePlan",
    "StaticIntakePlan",
    "add_inland_outflow_boundary",
    "apply_sfincs_structures",
    "build_baseline_build_plan",
    "build_inland_sfincs_base",
    "build_domains",
    "create_handoffs",
    "build_region_setup",
    "build_sfincs_coverage_and_wflow_preflight",
    "build_static_data_catalog",
    "build_static_intake_plan",
    "clip_dem_and_landcover_to_bbox",
    "collect_coastal_terrain_landcover_inputs",
    "collect_inland_static_inputs",
    "collect_ssurgo_infiltration_inputs",
    "collect_static_region_inputs",
    "collect_wflow_static_region_inputs",
    "download_file",
    "derive_massgis_sfincs_structure_layers",
    "ensure_usgs_3dep_dem_covers_bbox",
    "fetch_usgs_3dep_dem",
    "fetch_worldcover_landcover",
    "InlandSfincsBasePlan",
    "InlandSfincsDomainSetPlan",
    "is_built_sfincs_base",
    "is_built_wflow_base",
    "meaningful_model_files",
    "plan_inland_sfincs_domain_set",
    "plan_inland_sfincs_base",
    "plot_structure_layers",
    "plot_coastal_static_input_qa",
    "plot_static_input_qa",
    "plot_sfincs_handoff_basemap",
    "prepare_structure_layers",
    "sfincs_grid_resolution_matches",
    "sfincs_rivers_inflow_geoms",
    "set_observations",
    "write_inland_sfincs_domain_set_manifest",
    "write_inland_sfincs_handoff_locations",
    "write_inland_sfincs_handoff_locations_from_wflow_rivers",
    "validate_physics",
    "worldcover_tile_urls",
]
