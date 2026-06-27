import numpy as np
import pandas as pd

from design_events.fit_history.storm_type import classify_storm_type


def _tracks_with_a_pass(centroid_time):
    # one TC passing ~20 km from the Marshfield coast at centroid_time, plus two distant
    # offshore tracks in 2001 so the record's coverage spans the non-TC test events
    return pd.DataFrame(
        {
            "storm_id": ["AL012000"] * 3 + ["AL012001", "AL022001"],
            "time": pd.to_datetime(
                [
                    centroid_time - pd.Timedelta(hours=6), centroid_time, centroid_time + pd.Timedelta(hours=6),
                    "2001-02-01 00:00", "2001-09-01 00:00",
                ]
            ),
            "lat": [41.5, 42.0, 42.5, 22.0, 24.0],
            "lon": [-71.0, -70.5, -70.0, -55.0, -58.0],
        }
    )


def test_classify_storm_type_four_way():
    centroid = (-70.7, 42.1)
    landfall = pd.Timestamp("2000-09-15 12:00")
    tracks = _tracks_with_a_pass(landfall)
    events = pd.Series(
        [
            landfall + pd.Timedelta(hours=3),   # TC passing within radius + window
            pd.Timestamp("2001-01-20 00:00"),   # no track, cool season -> nor'easter
            pd.Timestamp("2001-07-04 00:00"),   # no track, warm season -> other non-tropical
            pd.NaT,                              # missing time -> unresolved
        ]
    )
    labels = classify_storm_type(events, tracks, centroid_lonlat=centroid, radius_km=350.0)
    assert list(labels) == ["tc", "nor_easter", "other_non_tropical", "unresolved"]


def test_distant_tropical_track_is_not_tc():
    # a tropical track that stays far offshore must not be labeled TC
    centroid = (-70.7, 42.1)
    t = pd.Timestamp("2005-08-20 00:00")
    far = pd.DataFrame({"storm_id": ["AL022005"], "time": [t], "lat": [25.0], "lon": [-60.0]})
    label = classify_storm_type(pd.Series([t]), far, centroid_lonlat=centroid, radius_km=350.0)
    assert label.iloc[0] == "other_non_tropical"  # warm-season non-tropical, not tc
