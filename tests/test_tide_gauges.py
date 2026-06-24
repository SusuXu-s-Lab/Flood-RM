from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from sfincs_runs import diagnostics as sfincs_diagnostics
from sfincs_runs import tide_gauges


def _candidates() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "candidate_id": ["good", "bad"],
            "candidate_source": ["synthetic", "synthetic"],
            "candidate_label": ["good", "bad"],
            "nearby_annual_damage": [90.0, 10.0],
            "x": [0.0, 1000.0],
            "y": [0.0, 0.0],
        },
        geometry=[Point(0, 0), Point(1000, 0)],
        crs="EPSG:26919",
    )


def test_candidate_scoring_prefers_known_damage_signal():
    response = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"] * 2,
            "design_scenario": ["base"] * 8,
            "candidate_id": ["good"] * 4 + ["bad"] * 4,
            "nearby_max_depth_ft": [0.0, 1.0, 2.0, 3.0, 3.0, 0.0, 2.0, 1.0],
            "total_damage": [0.0, 100.0, 200.0, 300.0] * 2,
            "probability_weight": [0.25] * 8,
        }
    )

    scores = tide_gauges.score_sensor_candidates(response, _candidates())

    ranked = scores.set_index("candidate_id")
    assert scores.iloc[0]["candidate_id"] == "good"
    assert ranked.loc["good", "weighted_damage_correlation"] > 0.99
    assert ranked.loc["good", "score"] > ranked.loc["bad", "score"]


def test_greedy_sensor_selection_skips_nearby_duplicates():
    scores = gpd.GeoDataFrame(
        {
            "candidate_id": ["best", "duplicate", "far"],
            "candidate_source": ["synthetic"] * 3,
            "candidate_label": ["best", "duplicate", "far"],
            "score": [1.0, 0.99, 0.5],
            "nearby_annual_damage": [10.0, 9.0, 1.0],
            "x": [0.0, 50.0, 1000.0],
            "y": [0.0, 0.0, 0.0],
        },
        geometry=[Point(0, 0), Point(50, 0), Point(1000, 0)],
        crs="EPSG:26919",
    )

    selected = tide_gauges.greedy_sensor_selection(scores, sensor_count=2, min_distance_m=350.0)

    assert list(selected["candidate_id"]) == ["best", "far"]
    assert list(selected["selected_rank"]) == [1, 2]


def test_partial_probability_coverage_is_reported_not_renormalized():
    weights = pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "probability_weight": [0.2, 0.3, 0.5],
        }
    )
    outcomes = pd.DataFrame({"event_id": ["e1", "e2"], "probability_weight": [0.2, 0.3]})

    coverage = sfincs_diagnostics.outcome_coverage(outcomes, weights)

    assert coverage["covered_probability_weight"] == pytest.approx(0.5)
    assert coverage["catalog_probability_weight"] == pytest.approx(1.0)
    assert coverage["weight_coverage"] == pytest.approx(0.5)


def test_candidate_geometry_and_crs_are_preserved():
    building_risk = gpd.GeoDataFrame(
        {
            "object_id": ["b1", "b2"],
            "annual_damage": [100.0, 50.0],
        },
        geometry=[Point(0, 0), Point(500, 0)],
        crs="EPSG:26919",
    )

    candidates = tide_gauges.candidate_points_from_building_risk(
        building_risk,
        top_n=2,
        min_distance_m=100.0,
        crs="EPSG:26919",
    )

    assert isinstance(candidates, gpd.GeoDataFrame)
    assert candidates.crs == "EPSG:26919"
    assert candidates.geometry.iloc[0].equals(Point(0, 0))
    assert np.isfinite(candidates["nearby_annual_damage"]).all()
