from pathlib import Path
import importlib.metadata
import json
import os
import subprocess
import sys

import pytest

import study_location
from paths import default_location_config_path, resolve_repo_path


def _requires_editable_install():
    try:
        direct_url = importlib.metadata.distribution("flood-rm").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        direct_url = None
    if not direct_url or not json.loads(direct_url).get("dir_info", {}).get("editable"):
        pytest.skip("editable-install import contract only applies to checkout environments")


def test_study_location_facade_preserves_public_imports():
    expected_names = {
        "StudyLocation",
        "LocationDefinition",
        "find_repo_root",
        "default_location_config_path",
        "resolve_repo_path",
        "load_location_config",
        "build_study_area",
        "study_area_bbox",
        "define_location",
        "resolve_study_location",
    }

    assert expected_names <= set(dir(study_location))


def test_default_location_config_path_prefers_explicit_config(tmp_path):
    config = tmp_path / "custom.yaml"
    environ = {
        "FLOOD_RM_LOCATION_CONFIG": "custom.yaml",
        "FLOOD_RM_LOCATION": "ignored",
    }

    assert default_location_config_path(tmp_path, environ=environ) == config.resolve()


def test_resolve_repo_path_is_repo_relative(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    cwd = tmp_path / "work"
    cwd.mkdir()
    cwd_file = cwd / "local.txt"
    cwd_file.write_text("local", encoding="utf-8")
    root_file = root / "local.txt"
    root_file.write_text("repo", encoding="utf-8")
    monkeypatch.chdir(cwd)

    assert resolve_repo_path("local.txt", repo_root=root) == root_file.resolve()


def test_deep_merge_does_not_mutate_inputs():
    left = {"outer": {"a": 1, "b": 2}, "kept": True}
    right = {"outer": {"b": 3, "c": 4}}

    merged = study_location._deep_merge(left, right)

    assert merged == {"outer": {"a": 1, "b": 3, "c": 4}, "kept": True}
    assert left == {"outer": {"a": 1, "b": 2}, "kept": True}
    assert right == {"outer": {"b": 3, "c": 4}}


def test_resolve_study_location_keeps_inland_event_driver_contract(tmp_path):
    repo_root = tmp_path
    location_root = repo_root / "locations" / "demo"
    location_root.mkdir(parents=True)
    (location_root / "config.yaml").write_text("project:\n  name: demo\n", encoding="utf-8")

    location = study_location.resolve_study_location(
        {
            "project": {"name": "demo"},
            "flood_setting": "inland",
            "event_drivers": ["rainfall", "streamflow", "soil_moisture"],
        },
        repo_root,
    )

    assert location.name == "demo"
    assert location.root == location_root
    assert location.event_drivers == ("rainfall", "streamflow", "soil_moisture")
    assert location.uses_coastal_water_level is False
    assert location.is_configured is True


def test_editable_checkout_resolves_top_level_modules_from_src():
    _requires_editable_install()
    repo_root = Path(__file__).resolve().parents[2]
    script = """
from pathlib import Path

import aoi
import paths
import study_location

source_root = Path.cwd().resolve() / "src"
modules = {"aoi": aoi, "paths": paths, "study_location": study_location}
wrong = {
    name: str(Path(module.__file__).resolve())
    for name, module in modules.items()
    if not Path(module.__file__).resolve().is_relative_to(source_root)
}
if wrong:
    raise SystemExit(f"top-level modules did not resolve from src: {wrong}")
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_editable_install_does_not_copy_top_level_modules_into_site_packages():
    _requires_editable_install()
    repo_root = Path(__file__).resolve().parents[2]
    script = """
from pathlib import Path
import site

copied = {}
for site_packages in site.getsitepackages():
    root = Path(site_packages)
    for name in ("aoi.py", "paths.py", "study_location.py"):
        path = root / name
        if path.exists():
            copied[name] = str(path)

if copied:
    raise SystemExit(f"editable install contains copied top-level modules: {copied}")
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
