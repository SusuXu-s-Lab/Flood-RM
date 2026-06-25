"""Notebook order for the Marshfield location pipeline."""

GRID_NOTEBOOKS = (
    "01_grid/01_base_network.ipynb",
    "01_grid/02_augment_network/01_der_inventory.ipynb",
    "01_grid/02_augment_network/02_load_profiles.ipynb",
    "01_grid/02_augment_network/03_switch_synthesis.ipynb",
    "01_grid/02_augment_network/04_load_blocks.ipynb",
    "01_grid/02_augment_network/05_onm_export.ipynb",
    "01_grid/03_audit_network.ipynb",
    "01_grid/psds_plot.ipynb",
)

FLOOD_NOTEBOOKS = (
    "02_flood/01_region_setup.ipynb",
    "02_flood/02_collect_sources.ipynb",
    "02_flood/03_build_event_catalog.ipynb",
    "02_flood/04/a_build_waves.ipynb",
    "02_flood/04/b_example_waves.ipynb",
    "02_flood/05_create_scenarios.ipynb",
    "02_flood/06_evaluate.ipynb",
    "02_flood/07_risk_fiat.ipynb",
)

PIPELINE_NOTEBOOKS = {
    "grid": GRID_NOTEBOOKS,
    "flood": FLOOD_NOTEBOOKS,
    "all": (*GRID_NOTEBOOKS, *FLOOD_NOTEBOOKS),
}
