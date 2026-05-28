from pathlib import Path

from sfincs_runs.scenarios.scenario_stats import parse_args


def suffix(path):
    return Path(path).relative_to(Path(__file__).resolve().parents[3]).as_posix()


def test_scenario_stats_accepts_location_config_for_defaults():
    args = parse_args(["--config", "locations/marshfield/config.yaml"])

    assert suffix(args.scenarios_dir) == "locations/marshfield/data/sfincs/scenarios"
    assert suffix(args.storage_dir) == "locations/marshfield/data/sfincs/run_outputs"
    assert suffix(args.stats_dir) == "locations/marshfield/data/sfincs/stats"
