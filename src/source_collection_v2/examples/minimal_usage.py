from pathlib import Path

import pandas as pd

from design_events.stochastic_boundary import plan, run
from design_events.stochastic_boundary.members import empirical_measure, event_members
from design_events.stochastic_boundary.rainfall import build_aorc_sst

paths = {
    "repo_root": Path("."),
    "location_root": Path("locations/example"),
    "location_name": "example",
    "source_artifacts_root": Path("locations/example/data/sources/source_artifacts"),
    "aorc_sst_rainfall_members_csv": Path("locations/example/data/sources/aorc_sst/rainfall_members.csv"),
    "waterlevel_csv": Path("locations/example/data/sources/cora/waterlevel.csv"),
    "nwm_soil_moisture_csv": Path("locations/example/data/sources/nwm/soil_moisture.csv"),
    "event_members_csv": Path("locations/example/data/sources/event_members.csv"),
}

config = {
    "collection": {
        "start": "1979-01-01",
        "end": "2022-12-31",
        "aorc_sst": {
            "bbox_wgs84": [-75.0, 35.0, -74.0, 36.0],
            "storm_duration_hours": 72,
            "min_precip_threshold": 100.0,
            "decluster_hours": 72,
        },
    }
}

collection_plan = plan(config, paths)
audit = run(collection_plan, skip_existing=True)
print(audit)

aorc_step = next(step for step in collection_plan.steps if step.name in {"aorc", "aorc_sst"})
build_aorc_sst(aorc_step.settings(config, paths), skip_existing=True)

rain = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
members = event_members(rainfall=rain, output_csv=paths["event_members_csv"])
weights = empirical_measure(members)
print(members.head())
print(weights.head())
