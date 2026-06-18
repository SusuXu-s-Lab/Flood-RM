"""Characterization snapshot of the merged region config.

`define_location` deep-merges each location's `config.yaml` with its included
`grid/sfincs/wflow/data_sources.yaml` into one dict that `src/` reads key-by-key.
These tests freeze that merged dict per location so any YAML restructuring
(shared inland base, thin overrides, dead-key removal) can be proven
behavior-preserving: the merged dict must stay identical except where we
intentionally change it.

Regenerate the golden after an *intentional* config change with:

    REGEN_REGION_CONFIG_GOLDEN=1 uv run pytest tests/flood_rm/region_config_snapshot_test.py

then review the git diff of tests/flood_rm/golden/ to confirm only the intended
keys changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from study_location import define_location

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).parent / "golden"
LOCATIONS = ("austin", "greensboro", "marshfield")

# Each location's current model CRS, asserted independently of the config keys so
# that deleting the redundant sfincs/wflow model_crs keys (which only duplicate
# project.model_crs) is provably effective-behavior-preserving.
EXPECTED_MODEL_CRS = {
    "austin": "EPSG:32614",
    "greensboro": "EPSG:32617",
    "marshfield": "EPSG:26919",
}


def _merged_config(name: str) -> dict:
    return define_location(REPO_ROOT / "locations" / name / "config.yaml").config


@pytest.mark.parametrize("name", LOCATIONS)
def test_merged_region_config_matches_golden(name):
    config = _merged_config(name)
    golden_path = GOLDEN_DIR / f"{name}.json"
    if os.environ.get("REGEN_REGION_CONFIG_GOLDEN"):
        GOLDEN_DIR.mkdir(exist_ok=True)
        golden_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"regenerated golden snapshot for {name}")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert config == golden


@pytest.mark.parametrize("name", LOCATIONS)
def test_effective_model_crs_resolves_via_project_fallback(name):
    # Mirrors how every consumer reads the CRS: sfincs/wflow override, else project.
    config = _merged_config(name)
    project_crs = config["project"]["model_crs"]
    sfincs_crs = config.get("sfincs", {}).get("model_crs", project_crs)
    wflow_crs = config.get("wflow", {}).get("model_crs", project_crs)
    assert project_crs == EXPECTED_MODEL_CRS[name]
    assert sfincs_crs == EXPECTED_MODEL_CRS[name]
    assert wflow_crs == EXPECTED_MODEL_CRS[name]
