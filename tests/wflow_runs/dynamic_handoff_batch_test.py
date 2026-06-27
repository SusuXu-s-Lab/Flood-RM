import json
from pathlib import Path

import pandas as pd
import xarray as xr

from wflow_runs.dynamic_handoff_batch import dynamic_handoff_batch_worklist


def test_dynamic_handoff_batch_worklist_selects_blocked_events(tmp_path):
    location_root = tmp_path / "locations/greensboro"
    catalog_path = location_root / "data/event_catalog/catalog/scenario_catalog.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"event_id": "accepted_evt", "event_reference_time": "2020-01-01T00:00:00"},
            {
                "event_id": "blocked_evt",
                "event_reference_time": "2020-01-01T00:00:00",
                "streamflow_member_id": "02000000_20200101T000000",
            },
            {
                "event_id": "incompatible_evt",
                "event_reference_time": "2020-01-01T00:00:00",
                "streamflow_member_id": "03000000_20200101T000000",
            },
        ]
    ).to_csv(catalog_path, index=False)
    members_path = location_root / "data/sources/usgs_streamgages/streamflow_members.csv"
    members_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "member_id": "02000000_20200101T000000",
                "site_no": "02000000",
                "event_time": "2020-01-01T00:00:00",
                "peak_flow_cfs": 100.0,
                "contributing_site_nos": "",
            },
            {
                "member_id": "03000000_20200101T000000",
                "site_no": "03000000",
                "event_time": "2020-01-01T00:00:00",
                "peak_flow_cfs": 100.0,
                "contributing_site_nos": "",
            },
        ]
    ).to_csv(members_path, index=False)
    gauges_path = location_root / "data/wflow/domain_set_gauges/greensboro_rural_observation_gauges.geojson"
    gauges_path.parent.mkdir(parents=True, exist_ok=True)
    gauges_path.write_text(
        """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"site_no": "02000000"},
      "geometry": {"type": "Point", "coordinates": [-79.0, 36.0]}
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    accepted_root = location_root / "data/wflow/events/accepted_evt"
    accepted_root.mkdir(parents=True, exist_ok=True)
    discharge = accepted_root / "sfincs_discharge.nc"
    xr.Dataset(coords={"time": pd.date_range("2020-01-04T00:00:00", periods=1, freq="h")}).to_netcdf(discharge)
    (accepted_root / "sfincs_discharge.dynamic_handoff.json").write_text(
        json.dumps(
            {
                "event_id": "accepted_evt",
                "status": "accepted",
                "discharge_source": "wflow_dynamic",
                "discharge_nc": str(discharge),
                "checks": [
                    {"check": "event_peak", "status": "passed", "message": ""},
                    {"check": "source_ids", "status": "passed", "message": ""},
                    {"check": "zero_rain_peak_fraction", "status": "passed", "message": ""},
                ],
                "metadata": {"streamflow_realization": "wflow_external_river_inflow"},
            }
        ),
        encoding="utf-8",
    )
    config = {
        "wflow": {
            "events_root": "data/wflow/events",
            "domain_set": {"submodels": [{"wflow_submodel_id": "greensboro_rural"}]},
        },
        "inland_coupling": {"discharge_forcing": {"source": "wflow_dynamic"}},
    }

    blocked = dynamic_handoff_batch_worklist(
        config,
        location_root,
        catalog_path=catalog_path,
        status="blocked",
    )
    accepted = dynamic_handoff_batch_worklist(
        config,
        location_root,
        catalog_path=catalog_path,
        status="accepted",
    )
    all_events = dynamic_handoff_batch_worklist(
        config,
        location_root,
        catalog_path=catalog_path,
        status="all",
    )

    assert blocked["event_id"].tolist() == ["blocked_evt"]
    assert accepted["event_id"].tolist() == ["accepted_evt"]
    assert all_events.set_index("event_id").loc["incompatible_evt", "status"] == "incompatible"
