from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from fiat_runs import diagnostics as fiat_diagnostics
from sfincs_runs import diagnostics as sfincs_diagnostics


def test_annual_rate_and_partial_coverage_are_not_renormalized():
    weights = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "probability_weight": [0.2, 0.3, 0.5],
            "event_origin": ["synthetic_body", "synthetic_body", "synthetic_tail"],
        }
    )
    rates = sfincs_diagnostics.annual_rate_table(weights, total_rate=10.0)

    assert rates.loc[rates["event_id"] == "e2", "annual_rate"].iloc[0] == 3.0

    outcomes = pd.DataFrame({"event_id": ["e1", "e2"], "probability_weight": [0.2, 0.3]})
    coverage = sfincs_diagnostics.outcome_coverage(outcomes, weights)

    assert coverage["completed_outcome_events"] == 2
    assert coverage["catalog_weighted_events"] == 3
    assert coverage["covered_probability_weight"] == 0.5
    assert coverage["weight_coverage"] == 0.5


def test_poisson_exceedance_probability_from_annual_rates():
    rates = np.array([0.0, 0.5, 1.0])
    probability = sfincs_diagnostics.poisson_exceedance_probability(rates)

    np.testing.assert_allclose(probability, 1.0 - np.exp(-rates))


def test_building_risk_frames_preserves_geometry_and_weights():
    outcomes = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "design_scenario": ["base", "base"],
            "probability_weight": [0.25, 0.75],
            "annual_rate": [0.5, 1.5],
        }
    )
    exposure = pd.DataFrame(
        {
            "object_id": ["b1", "b2"],
            "primary_object_type": ["RES", "COM"],
            "aggregation_label:Census Blockgroup": ["bg1", "bg2"],
        }
    )
    event_1 = gpd.GeoDataFrame(
        {
            "object_id": ["b1", "b2"],
            "event_id": ["e1", "e1"],
            "design_scenario": ["base", "base"],
            "total_damage": [100.0, 0.0],
            "inun_depth": [1.0, np.nan],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )
    event_2 = gpd.GeoDataFrame(
        {
            "object_id": ["b1", "b2"],
            "event_id": ["e2", "e2"],
            "design_scenario": ["base", "base"],
            "total_damage": [200.0, 50.0],
            "inun_depth": [3.0, 2.0],
        },
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )

    risk = fiat_diagnostics.building_risk_frames([event_1, event_2], outcomes, exposure=exposure)
    b1 = risk.set_index("object_id").loc["b1"]
    b2 = risk.set_index("object_id").loc["b2"]

    assert isinstance(risk, gpd.GeoDataFrame)
    assert risk.crs == "EPSG:4326"
    assert b1.geometry.equals(Point(0, 0))
    assert b1["annual_damage"] == 350.0
    assert math.isclose(b1["damage_aep"], 1.0 - math.exp(-2.0))
    assert b1["weighted_mean_inun_depth_ft"] == 2.5
    assert b2["annual_damage"] == 75.0
    assert b2["primary_object_type"] == "COM"


def test_weighted_standardized_associations_detect_monotonic_driver():
    data = pd.DataFrame(
        {
            "driver": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "noise": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "outcome": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
            "probability_weight": [1.0 / 8.0] * 8,
            "storm_type": ["nor_easter"] * 8,
        }
    )

    assoc = sfincs_diagnostics.weighted_standardized_associations(
        data,
        drivers=["driver", "noise"],
        outcomes=["outcome"],
        min_rows=4,
    )
    driver_row = assoc[(assoc["storm_type"] == "all") & (assoc["driver"] == "driver")].iloc[0]

    assert driver_row["standardized_wls_coefficient"] > 0.9
    assert driver_row["weighted_correlation"] > 0.99
