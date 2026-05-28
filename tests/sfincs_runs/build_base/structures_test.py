import json
import shutil
from pathlib import Path

from sfincs_runs.build_base.structures import (
    apply_sfincs_structures,
    derive_massgis_sfincs_structure_layers,
    plot_structure_layers,
    prepare_structure_layers,
)
from study_location import define_location


def _write_geojson(path, ids):
    features = []
    for i, feature_id in enumerate(ids):
        features.append(
            {
                "type": "Feature",
                "properties": {"id": feature_id, "name": feature_id},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-70.7, 42.1 + i * 0.01], [-70.69, 42.1 + i * 0.01]],
                },
            }
        )
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )


def test_prepare_structure_layers_pulls_only_configured_local_layers(tmp_path):
    source_root = tmp_path / "screening"
    source_root.mkdir()
    _write_geojson(source_root / "weirs.geojson", ["WEIR_A", "WEIR_B"])
    _write_geojson(source_root / "thin_dams.geojson", ["THD_A"])
    _write_geojson(source_root / "drainage.geojson", ["DRN_A"])

    config = {
        "project": {"name": "marshfield"},
        "sfincs_structures": {
            "enabled": True,
            "source_root": str(source_root),
            "output_root": "data/static/structures/screening",
            "layers": [
                {
                    "name": "weirs",
                    "component": "weirs",
                    "source": "weirs.geojson",
                    "dz": None,
                },
                {
                    "name": "thin_dams",
                    "component": "thin_dams",
                    "source": "thin_dams.geojson",
                },
            ],
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations" / "marshfield",
    }

    plan = prepare_structure_layers(config, paths)

    assert [layer.name for layer in plan.applied_layers] == ["weirs", "thin_dams"]
    assert [layer.kind for layer in plan.applied_layers] == ["weir", "thin_dam"]
    assert plan.dropped_layers == ()
    assert plan.applied_layers[0].path == paths["location_root"] / "data/static/structures/screening/weirs.geojson"
    assert json.loads(plan.applied_layers[0].path.read_text(encoding="utf-8"))["features"][0]["properties"]["id"] == "WEIR_A"
    assert not (paths["location_root"] / "data/static/structures/screening/dropped/drainage.geojson").exists()
    assert [row["decision"] for row in plan.summary_rows()] == ["applied", "applied"]


class _Recorder:
    def __init__(self):
        self.calls = []
        self.weirs = self
        self.thin_dams = self
        self.drainage_structures = self

    def create(self, **kwargs):
        self.calls.append(kwargs)


def test_apply_sfincs_structures_uses_hydromt_sfincs_v2_component_calls(tmp_path):
    source_root = tmp_path / "screening"
    source_root.mkdir()
    _write_geojson(source_root / "weirs.geojson", ["WEIR_A"])
    _write_geojson(source_root / "thin_dams.geojson", ["THD_A"])
    _write_geojson(source_root / "drainage.geojson", ["DRN_A"])
    config = {
        "project": {"name": "marshfield"},
        "sfincs_structures": {
            "enabled": True,
            "source_root": str(source_root),
            "layers": [
                {"name": "weirs", "kind": "weir", "component": "weirs", "source": "weirs.geojson", "dz": None},
                {"name": "thin_dams", "kind": "thin_dam", "component": "thin_dams", "source": "thin_dams.geojson"},
                {
                    "name": "drainage",
                    "kind": "drainage_structure",
                    "component": "drainage_structures",
                    "source": "drainage.geojson",
                    "enabled": False,
                    "stype": "valve",
                },
            ],
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations" / "marshfield",
    }

    plan = prepare_structure_layers(config, paths)
    sf = _Recorder()
    apply_sfincs_structures(sf, plan)

    assert sf.calls == [
        {
            "locations": str(paths["location_root"] / "data/static/structures/screening/weirs.geojson"),
            "dz": None,
            "merge": True,
        },
        {
            "locations": str(paths["location_root"] / "data/static/structures/screening/thin_dams.geojson"),
            "merge": True,
        },
    ]


def test_plot_structure_layers_writes_visual_inventory(tmp_path):
    source_root = tmp_path / "screening"
    source_root.mkdir()
    _write_geojson(source_root / "weirs.geojson", ["WEIR_A"])
    config = {
        "project": {"name": "marshfield"},
        "sfincs_structures": {
            "enabled": True,
            "source_root": str(source_root),
            "layers": [
                {"name": "weirs", "kind": "weir", "component": "weirs", "source": "weirs.geojson"},
            ],
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations" / "marshfield",
    }
    plan = prepare_structure_layers(config, paths)

    output = plot_structure_layers(plan, output_path=tmp_path / "figures" / "structures.png")

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_structure_layers_writes_map_context_inventory(tmp_path):
    source_root = tmp_path / "screening"
    source_root.mkdir()
    _write_geojson(source_root / "weirs.geojson", ["WEIR_A"])
    config = {
        "project": {"name": "marshfield"},
        "sfincs_structures": {
            "enabled": True,
            "source_root": str(source_root),
            "layers": [
                {"name": "weirs", "kind": "weir", "component": "weirs", "source": "weirs.geojson"},
            ],
        },
    }
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations" / "marshfield",
    }
    plan = prepare_structure_layers(config, paths)

    output = plot_structure_layers(
        plan,
        output_path=tmp_path / "figures" / "structures_map.png",
        map_context=True,
        single_color="#087dbb",
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_derive_massgis_sfincs_structure_layers_creates_weirs_and_thin_dams(tmp_path):
    source = tmp_path / "massgis.geojson"
    source.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {
                            "STR_ID": "SW-1",
                            "PrimaryTyp": "Bulkhead/ Seawall",
                            "PrimaryHei": "10 to 15 Feet",
                            "PositionZ": "14",
                            "Location": "Ocean Street",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-70.7, 42.1], [-70.69, 42.1]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "STR_ID": "RV-1",
                            "PrimaryTyp": "Revetment",
                            "PrimaryHei": "Over 15 Feet",
                            "PositionZ": " ",
                            "Location": "No usable elevation",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-70.7, 42.11], [-70.69, 42.11]],
                        },
                    },
                    {
                        "type": "Feature",
                        "properties": {
                            "STR_ID": "JT-1",
                            "PrimaryTyp": "Groin/ Jetty",
                            "PrimaryHei": "5 to 10 Feet",
                            "PositionZ": "",
                            "Location": "Green Harbor",
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[-70.7, 42.12], [-70.69, 42.12]],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    outputs = derive_massgis_sfincs_structure_layers(source, tmp_path / "derived")

    weirs = json.loads(outputs["weirs"].read_text(encoding="utf-8"))
    thin_dams = json.loads(outputs["thin_dams"].read_text(encoding="utf-8"))
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert [feature["properties"]["id"] for feature in weirs["features"]] == ["SW-1"]
    assert weirs["features"][0]["properties"]["z"] == 4.267
    assert weirs["features"][0]["properties"]["z_ft_navd88"] == 14.0
    assert weirs["features"][0]["properties"]["primary_visible_height_ft_band"] == "10 to 15 Feet"
    assert weirs["features"][0]["properties"]["design_grade_elevation_source"] == (
        "WHG/USACE/Town plan sheets or as-built survey."
    )
    assert weirs["features"][0]["properties"]["design_grade_geometry_source"] == (
        "WHG/USACE/Town plan sheets or final CAD/as-built drawings."
    )
    assert [feature["properties"]["id"] for feature in thin_dams["features"]] == ["JT-1"]
    assert summary["weir_features"] == 1
    assert summary["thin_dam_features"] == 1
    assert summary["omitted_features"] == 1


def test_marshfield_runtime_structure_config_applies_only_weirs_and_thin_dams(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    config = define_location(repo_root / "locations/marshfield/config.yaml").config
    shutil.copytree(
        repo_root / "locations/marshfield/data/static/structures/sources",
        tmp_path / "locations/marshfield/data/static/structures/sources",
    )
    paths = {
        "repo_root": tmp_path,
        "location_name": "marshfield",
        "location_root": tmp_path / "locations" / "marshfield",
    }

    plan = prepare_structure_layers(config, paths)

    assert [(layer.name, layer.component, layer.kind, layer.feature_count) for layer in plan.applied_layers] == [
        ("weirs", "weirs", "weir", 25),
        ("thin_dams", "thin_dams", "thin_dam", 6),
    ]
    assert all(layer.component != "drainage_structures" for layer in plan.applied_layers)
    assert plan.dropped_layers == ()
    assert sorted(path.name for path in (paths["location_root"] / "data/static/structures/processed").iterdir()) == [
        "thin_dams_marshfield_massgis_public_2015.geojson",
        "weirs_marshfield_massgis_public_2015.geojson",
    ]
