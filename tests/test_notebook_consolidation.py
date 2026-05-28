"""Notebook consolidation contract.

The 04 workflow is split into lightweight base-model build notebooks and
Single-Use-Case example notebooks. Rebuilding the base should not rerun SFINCS
or plotting, while examples still exercise the ADR-0006 event-run path.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCATION_DIR = REPO_ROOT / "locations/marshfield"
NOTEBOOKS_DIR = LOCATION_DIR / "02_flood"
CONFIG_YAML = LOCATION_DIR / "config.yaml"


def _code_text(notebook_path: Path) -> str:
    nb = json.loads(notebook_path.read_text())
    return "\n".join(
        ("".join(c["source"]) if isinstance(c["source"], list) else c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


def test_waves_sfincs_invokes_single_use_case_plan():
    """The wave-coupled example notebook must wire build_single_use_case_plan so
    event selection follows ADR-0006 instead of being hardcoded."""
    source = _code_text(NOTEBOOKS_DIR / "04/b_example_waves.ipynb")
    assert "build_single_use_event" in source, (
        "b_example_waves.ipynb must call build_single_use_event so ADR-0006 plan selection is used"
    )


def test_standard_sfincs_is_end_to_end():
    """b_example_standard.ipynb mirrors the wave-coupled example: Single-Use-
    Case Test Plan + stage + run + plot, without rebuilding the base."""
    source = _code_text(NOTEBOOKS_DIR / "04/b_example_standard.ipynb")
    assert "build_single_use_event" in source, "must wire the Single-Use-Case Test Plan"
    assert "event_forcing = single_use_event.forcing" in source, "must read Event Catalog water level"
    assert "stage_event_run" in source, "must stage SFINCS boundary forcing"
    assert "single_use_case_plan.run_root" in source, "must stage under plan.run_root"
    assert "run_result.map_path" in source, "must report the run's sfincs_map.nc path"
    assert "plot_flood_animation" in source, "must render the result"


def test_waves_sfincs_plots_event_output():
    """End-to-end smoke run is only useful if the notebook reads back
    sfincs_map.nc and produces a plot/animation. Without this cell the user
    can't visually confirm the run."""
    source = _code_text(NOTEBOOKS_DIR / "04/b_example_waves.ipynb")
    assert "run_result.map_path" in source, "must report the run's sfincs_map.nc path"
    assert "plot_flood_animation" in source, "must render the result"


def test_04_build_notebooks_do_not_run_or_plot_example_events():
    """The rebuild notebooks should be fast, base-only runbooks."""
    for notebook_name in ("04/a_build_standard.ipynb", "04/a_build_waves.ipynb"):
        source = _code_text(NOTEBOOKS_DIR / notebook_name)
        assert "run_sfincs_model(run_root" not in source
        assert "build_single_use_event" not in source
        assert "sfincs_map.nc" not in source
        assert "plot_flood_animation" not in source


def test_legacy_04_end_to_end_notebooks_removed_after_split():
    assert not (NOTEBOOKS_DIR / "04/standard_sfincs.ipynb").exists()
    assert not (NOTEBOOKS_DIR / "04/waves_sfincs.ipynb").exists()


def test_config_omits_notebook_paths():
    """The root README owns notebook order; location YAML defines the dataset."""
    import yaml

    config = yaml.safe_load(CONFIG_YAML.read_text())

    assert "notebooks" not in config
    assert config["includes"] == {
        "data_sources": "data_sources.yaml",
        "grid": "grid.yaml",
        "sfincs": "sfincs.yaml",
    }
    for rel_path in [
        "01_grid/01_base_network.ipynb",
        "01_grid/02_augment_network/01_der_inventory.ipynb",
        "01_grid/02_augment_network/02_load_profiles.ipynb",
        "01_grid/02_augment_network/03_switch_synthesis.ipynb",
        "01_grid/02_augment_network/04_load_blocks.ipynb",
        "01_grid/02_augment_network/05_onm_export.ipynb",
        "01_grid/03_audit_network.ipynb",
        "02_flood/01_region_setup.ipynb",
        "02_flood/02_collect_sources.ipynb",
        "02_flood/03_build_event_catalog.ipynb",
        "02_flood/04/a_build_standard.ipynb",
        "02_flood/04/a_build_waves.ipynb",
        "02_flood/04/b_example_standard.ipynb",
        "02_flood/04/b_example_waves.ipynb",
        "02_flood/05_create_scenarios.ipynb",
        "02_flood/06_evaluate.ipynb",
    ]:
        assert (LOCATION_DIR / rel_path).exists(), f"missing conventional notebook: {rel_path}"


def test_legacy_single_use_case_notebook_removed():
    """Once merged into 04/, the standalone 05 notebook is gone — keeping
    it around invites stale references and duplicate work."""
    assert not (NOTEBOOKS_DIR / "05_single_use_case_test.ipynb").exists(), (
        "05_single_use_case_test.ipynb still present; merge into 04/ and delete"
    )


def test_downstream_notebooks_renumbered():
    """After 05's removal, the cluster batch + evaluate-truth-set notebooks
    shift down one number so the sequence stays contiguous."""
    assert (NOTEBOOKS_DIR / "05_create_scenarios.ipynb").exists(), (
        "expected 05_create_scenarios.ipynb"
    )
    assert not (NOTEBOOKS_DIR / "05_prepare_cluster_batch.ipynb").exists(), (
        "old 05_prepare_cluster_batch.ipynb still present"
    )
    assert (NOTEBOOKS_DIR / "06_evaluate.ipynb").exists(), (
        "expected 06_evaluate.ipynb (renamed from 07_)"
    )
    assert not (NOTEBOOKS_DIR / "06_prepare_cluster_batch.ipynb").exists(), (
        "old 06_prepare_cluster_batch.ipynb still present"
    )
    assert not (NOTEBOOKS_DIR / "07_evaluate_truth_set.ipynb").exists(), (
        "old 07_evaluate_truth_set.ipynb still present"
    )


def test_create_scenarios_notebook_builds_wave_coupled_scenarios():
    source = _code_text(NOTEBOOKS_DIR / "05_create_scenarios.ipynb")
    raw = (NOTEBOOKS_DIR / "05_create_scenarios.ipynb").read_text(encoding="utf-8")

    assert "python -m sfincs_runs build_scenarios" in source
    assert "--include-waves" in source
    assert "--include-precip" in source
    assert "--zsini-mode boundary_t0" in source
    assert "cluster/run_sfincs_dsai_wave_coupled.slurm" in source
    assert "per-event SnapWave and precipitation forcing" in raw, (
        "create_scenarios must warn that wave-coupled scenarios need full event forcing"
    )


def test_no_stale_notebook_paths_in_src():
    """src/ must not reference the old (pre-consolidation) notebook
    filenames. The grep target list mirrors the renumber: legacy 04 single
    file, legacy 04 quadtree single file, 05/06/07 standalone."""
    import re

    stale_patterns = [
        re.compile(r"\b04_build_baseline_sfincs\.ipynb\b"),
        re.compile(r"\b04_build_sfincs_quadtree_snapwave\.ipynb\b"),
        re.compile(r"\b05_single_use_case_test\.ipynb\b"),
        re.compile(r"\b06_prepare_cluster_batch\.ipynb\b"),
        re.compile(r"\b07_evaluate_truth_set\.ipynb\b"),
    ]
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(errors="ignore")
        for pat in stale_patterns:
            if pat.search(text):
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {pat.pattern}")
    assert not offenders, "stale notebook paths in src/:\n  " + "\n  ".join(offenders)


def test_waves_sfincs_stages_event_run():
    """End-to-end means: read CORA forcing for the selected event window,
    stage a run folder from the base model, and write the bzs file. Without
    these the smoke run cannot actually execute."""
    source = _code_text(NOTEBOOKS_DIR / "04/b_example_waves.ipynb")
    assert "event_forcing = single_use_event.forcing" in source, (
        "must read Event Catalog water level for the event window"
    )
    assert "stage_event_run" in source, "must write staged SFINCS boundary forcing"
    assert "single_use_case_plan.run_root" in source, (
        "must stage the run folder under plan.run_root (not an ad-hoc path)"
    )


def test_04_notebooks_share_event_ingestion_and_audit_contract():
    """The standard and wave-coupled runbooks should ingest the same Event
    Catalog water-level/rain/soil data and differ only in the wave driver."""
    required_common = [
        "build_single_use_event",
        "resolve_event_hydrology_inputs",
        "stage_event_run",
        "stage_event_precipitation",
        "audit_forcing_manifest",
        "timing_config={\"allow_legacy_inference\": True}",
        "staged_manifest = staged.manifest",
        "precip_manifest = None",
        "staged_manifest.update(precip_manifest)",
        "run_start = pd.Timestamp(staged_manifest[\"run_start\"])",
        "run_stop = pd.Timestamp(staged_manifest[\"run_stop\"])",
        "prepared_precip",
        "plot_forcing_qa",
    ]
    for notebook_name in ("04/b_example_standard.ipynb", "04/b_example_waves.ipynb"):
        source = _code_text(NOTEBOOKS_DIR / notebook_name)
        missing = [token for token in required_common if token not in source]
        assert not missing, f"{notebook_name} missing common ingestion/audit tokens: {missing}"
        assert source.index("audit_forcing_manifest(run_root)") < source.index(
            "run_sfincs_model(run_root"
        ), f"{notebook_name} must audit staged forcing before running SFINCS"


def test_04_notebooks_differ_only_on_wave_coupling_controls():
    standard = _code_text(NOTEBOOKS_DIR / "04/b_example_standard.ipynb")
    waves = _code_text(NOTEBOOKS_DIR / "04/b_example_waves.ipynb")

    assert "include_waves=False" in standard
    assert "include_waves=True" not in standard
    assert "snapwave.bhs" not in standard
    assert "unwrap_direction_degrees" not in standard

    assert "include_waves=True" in waves
    assert "plot_forcing_qa_waves" in waves
    assert "plot_runup_overtopping" in waves


def test_04_notebooks_keep_flood_event_reader_plots_and_video():
    required_plotting = [
        "plot_flood_animation",
        "display(Video(str(out_mp4), embed=True))",
    ]
    for notebook_name in ("04/b_example_standard.ipynb", "04/b_example_waves.ipynb"):
        source = _code_text(NOTEBOOKS_DIR / notebook_name)
        missing = [token for token in required_plotting if token not in source]
        assert not missing, f"{notebook_name} missing flood plotting/video tokens: {missing}"
        assert source.index("run_sfincs_model(run_root") < source.index(
            "plot_flood_animation"
        ), f"{notebook_name} must render flood output after the model run"
