from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from design_events.collect_sources.reservoir_conditions import (
    ACRE_FT_TO_M3,
    ACRE_TO_M2,
    apply_reservoir_condition_summary,
    parse_twdb_reservoir_csv,
    summarize_twdb_reservoir_history,
)


TWDB_CSV = """# disclaimer
# Units:
#     date: YYYY-MM-DD
#     water_level: feet above vertical datum
date,water_level,surface_area,reservoir_storage,conservation_storage,percent_full,conservation_capacity,dead_pool_capacity
2025-01-01,10,100,1000,900,45,2000,100
2025-01-02,12,120,1440,1340,67,2000,100
2025-01-03,14,140,1960,1860,93,2000,100
"""


def test_twdb_reservoir_history_updates_wflow_area_volume_and_depth():
    history = parse_twdb_reservoir_csv(TWDB_CSV)
    record = summarize_twdb_reservoir_history(
        history,
        waterbody_index=0,
        waterbody_name="Lake Example",
        slug="example",
        url="https://waterdatafortexas.org/reservoirs/individual/example-1year.csv",
        statistic="median",
    )

    assert record["condition_status"] == "matched"
    assert record["Area_avg"] == pytest.approx(120 * ACRE_TO_M2)
    assert record["Vol_avg"] == pytest.approx(1440 * ACRE_FT_TO_M3)
    assert record["Depth_avg"] == pytest.approx((1440 * ACRE_FT_TO_M3) / (120 * ACRE_TO_M2))

    reservoirs = gpd.GeoDataFrame(
        {
            "waterbody_id": [1],
            "waterbody_name": ["Lake Example"],
            "Area_avg": [100.0],
            "Depth_avg": [5.0],
            "Vol_avg": [500.0],
            "Dis_avg": [1.0],
            "reservoir_parameter_source": ["bootstrap"],
            "review_status": ["review_required_public_waterbody_estimates"],
        },
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])],
        crs="EPSG:4326",
    )
    updated = apply_reservoir_condition_summary(reservoirs, pd.DataFrame([record]))

    assert updated.loc[0, "Area_avg"] == pytest.approx(record["Area_avg"])
    assert updated.loc[0, "Vol_avg"] == pytest.approx(record["Vol_avg"])
    assert updated.loc[0, "Depth_avg"] == pytest.approx(record["Depth_avg"])
    assert updated.loc[0, "review_status"] == "review_required_twdb_storage_no_control_outflow_fallback"
