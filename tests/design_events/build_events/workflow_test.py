import pandas as pd

from design_events.build_events.catalog import build_event_catalog
from design_events.build_events.workflow import _replay_columns


def test_replay_columns_keep_wflow_forcing_pointers():
    catalog = pd.DataFrame(
        columns=[
            "event_id",
            "streamflow_member_id",
            "streamflow_member_file",
            "streamflow_member_time",
            "streamflow_scale_factor",
            "rainfall_member_id",
            "rainfall_member_file",
            "rainfall_member_time",
            "rainfall_scale_factor",
            "soil_moisture_member_id",
            "soil_moisture_member_file",
            "soil_moisture_member_time",
            "wflow_event_dir",
        ]
    )

    assert _replay_columns(catalog) == list(catalog.columns)


def test_build_event_catalog_uses_event_catalog_plan(tmp_path):
    summary_path = tmp_path / "event_summary.csv"
    catalog_path = tmp_path / "event_catalog.csv"
    members_path = tmp_path / "event_members.nc"
    summary = pd.DataFrame(
        {
            "event_id": ["design_0001"],
            "sample_rp_years": [100.0],
            "sampling_region": ["tail"],
            "sampling_weight": [1.0],
            "probability_weight": [0.01],
            "template_peak_time": ["2020-01-01T00:00:00"],
            "peak": [1.2],
            "absolute_peak_m": [2.0],
            "valid_start_hour": [-36],
            "valid_end_hour": [36],
        }
    )
    summary.to_csv(summary_path, index=False)

    catalog = build_event_catalog(
        {"project": {"name": "marshfield"}},
        {
            "repo_root": tmp_path,
            "location_name": "marshfield",
            "event_summary_csv": summary_path,
            "event_members_nc": members_path,
            "event_catalog_csv": catalog_path,
            "event_catalog_audit_json": None,
            "scenario": {"name": "base"},
        },
    )

    assert catalog_path.exists()
    assert catalog.loc[0, "event_id"] == "design_0001"
    assert catalog.loc[0, "study_location"] == "marshfield"
    assert catalog.loc[0, "coastal_member_file"] == str(members_path)
