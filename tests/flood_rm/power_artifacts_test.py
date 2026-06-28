def _write_csv(path, rows):
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)


def test_export_base_writes_schema_parquet_without_artifact_io_wrappers(tmp_path):
    import json
    import pandas as pd

    from power.exports.smart_ds_grid import export_base, location_id

    registry = tmp_path / "registry"
    output = tmp_path / "augmented"
    registry.mkdir()
    _write_csv(
        registry / "transformers.csv",
        [
            {
                "transformer_name": "xfmr_1",
                "feeder_id": "Feeder A",
                "location_bus": "bus_1",
                "phases": "abc",
                "location_lon": -70.7,
                "location_lat": 42.1,
                "max_kv": 12.47,
                "max_kva": 500,
            }
        ],
    )
    _write_csv(
        registry / "sources.csv",
        [
            {
                "source_name": "source_1",
                "feeder_id": "Feeder A",
                "bus": "source_bus",
                "phases": "abc",
                "lon": -70.71,
                "lat": 42.11,
                "basekv": 12.47,
            }
        ],
    )
    _write_csv(
        registry / "load_buses.csv",
        [{"bus": "load_bus_1", "feeder_id": "Feeder A", "lon": -70.72, "lat": 42.12}],
    )
    _write_csv(
        registry / "lines.csv",
        [
            {
                "line_name": "line_1",
                "feeder_id": "Feeder A",
                "from_bus": "source_bus",
                "phases": "abc",
                "from_lon": -70.71,
                "from_lat": 42.11,
                "to_lon": -70.72,
                "to_lat": 42.12,
                "line_class": "overhead",
            }
        ],
    )
    _write_csv(registry / "feeders.csv", [{"feeder_id": "Feeder A", "load_kw": 25.0}])

    report = export_base(registry, output, debug_csv=False)

    assets = pd.read_parquet(output / "assets.parquet")
    control_units = pd.read_parquet(output / "control_units.parquet")
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert f"{location_id}:asset:sources:source_1" in set(assets["asset_id"])
    assert control_units["control_unit_id"].tolist() == [
        f"{location_id}:control_unit:feeder:feeder_a"
    ]
    assert manifest["outputs"]["assets.parquet"]["sha256"]
