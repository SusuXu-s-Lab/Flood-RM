import json

import pandas as pd

from design_events.collect_sources.source_artifacts import (
    read_source_artifact,
    source_artifact_covers,
    source_artifact_path,
    write_source_artifact,
)


def _paths(tmp_path):
    return {
        "repo_root": tmp_path,
        "location_name": "test_location",
        "source_artifacts_root": tmp_path / "data/sources/source_artifacts",
    }


def test_write_source_artifact_preserves_manifest_schema_and_relative_paths(tmp_path):
    paths = _paths(tmp_path)
    output = tmp_path / "data/sources/nwm/streamflow.csv"

    artifact_path = write_source_artifact(
        paths,
        source="nwm",
        kind="retrospective_hydrologic_state",
        start=pd.Timestamp("2020-01-01 00:00"),
        end=pd.Timestamp("2020-01-02 00:00"),
        artifacts={"streamflow_csv": output},
        metadata={"version": "2.1"},
    )

    assert artifact_path == source_artifact_path(paths, "nwm", "retrospective_hydrologic_state")
    manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert manifest == {
        "study_location": "test_location",
        "source": "nwm",
        "kind": "retrospective_hydrologic_state",
        "status": "complete",
        "start": "2020-01-01T00:00:00",
        "end": "2020-01-02T00:00:00",
        "artifacts": {"streamflow_csv": "data/sources/nwm/streamflow.csv"},
        "metadata": {"version": "2.1"},
    }
    assert read_source_artifact(paths, "nwm", "retrospective_hydrologic_state") == manifest


def test_source_artifact_covers_rejects_smoke_and_incomplete_windows(tmp_path):
    paths = _paths(tmp_path)
    write_source_artifact(
        paths,
        source="era5",
        kind="snapwave_boundary_forcing",
        start="2020-01-01",
        end="2020-01-31",
        metadata={"smoke": True},
    )
    assert not source_artifact_covers(paths, "era5", "snapwave_boundary_forcing", "2020-01-01", "2020-01-02")

    write_source_artifact(
        paths,
        source="era5",
        kind="snapwave_boundary_forcing",
        start="2020-01-01",
        end="2020-01-31",
    )
    assert source_artifact_covers(paths, "era5", "snapwave_boundary_forcing", "2020-01-02", "2020-01-30")
    assert not source_artifact_covers(paths, "era5", "snapwave_boundary_forcing", "2019-12-31", "2020-01-30")
    assert not source_artifact_covers(paths, "era5", "snapwave_boundary_forcing", "2020-01-02", "2020-02-01")
