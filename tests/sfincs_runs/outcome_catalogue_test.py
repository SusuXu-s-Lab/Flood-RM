import pandas as pd

from sfincs_runs.scenarios.outcome_catalogue import _rainfall_metrics


def test_rainfall_metrics_prefer_catalog_target_metric_over_raw_member_lookup():
    catalog = pd.DataFrame(
        {
            "rainfall_metric_mm": [55.0],
            "rainfall_member_file": ["missing-ranked-storms.csv"],
            "rainfall_member_time": ["1999-01-01T00:00:00"],
            "rainfall_scale_factor": [6.0],
        }
    )

    assert _rainfall_metrics(catalog) == [55.0]


def test_rainfall_metrics_scale_raw_member_metric_when_target_missing(tmp_path):
    ranked = tmp_path / "ranked-storms.csv"
    pd.DataFrame(
        {
            "storm_date": ["1999-01-01T00:00:00"],
            "mean": [50.0],
        }
    ).to_csv(ranked, index=False)
    catalog = pd.DataFrame(
        {
            "rainfall_member_file": [str(ranked)],
            "rainfall_member_time": ["1999-01-01T00:00:00"],
            "rainfall_scale_factor": [1.2],
        }
    )

    assert _rainfall_metrics(catalog) == [60.0]
