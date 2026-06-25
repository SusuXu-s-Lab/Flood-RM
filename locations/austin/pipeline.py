"""Notebook order for the Austin location pipeline."""

GRID_NOTEBOOKS = (
    "01_grid/sds_plot.ipynb",
)

FLOOD_NOTEBOOKS = (
    "02_flood/01_region_setup.ipynb",
    "02_flood/02_collect_sources.ipynb",
    "02_flood/03_build_event_catalog.ipynb",
    "02_flood/04/a_build_coupled_model.ipynb",
    "02_flood/04/b_prepare_wflow_dynamic_handoff.ipynb",
    "02_flood/04/c_run_example.ipynb",
    "02_flood/05_create_scenarios.ipynb",
    "02_flood/05b_calibrate_wshed.ipynb",
    "02_flood/05c_ship_calibrated.ipynb",
    "02_flood/06_evaluate.ipynb",
)

PIPELINE_NOTEBOOKS = {
    "grid": GRID_NOTEBOOKS,
    "flood": FLOOD_NOTEBOOKS,
    "all": (*GRID_NOTEBOOKS, *FLOOD_NOTEBOOKS),
}
