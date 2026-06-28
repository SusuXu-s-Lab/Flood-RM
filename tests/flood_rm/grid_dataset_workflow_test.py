from pathlib import Path


repo_root = Path(__file__).resolve().parents[2]
power_root = repo_root / "src" / "power"


def test_power_layout_uses_stakeholder_facing_modules():
    assert (power_root / "plotting.py").exists()
    assert (power_root / "baseline_network").is_dir()
    assert (power_root / "exports").is_dir()
    assert (power_root / "impact").is_dir()
    assert (power_root / "resilience").is_dir()


def test_power_layout_removed_stage_letter_and_shared_folders():
    for folder in ["kernel", "shared", "stage_a", "stage_b", "viz", "workflow", "onm"]:
        assert not (power_root / folder).exists()
