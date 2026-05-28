from __future__ import annotations

import os
import subprocess
import sys

from sfincs_runs.config import build_paths


stage_modules = {
    "build_scenarios": "sfincs_runs.scenarios.create_events",
    "run_scenarios": "sfincs_runs.scenarios.run_events",
    "stats": "sfincs_runs.scenarios.scenario_stats",
}


def _run_module(module_name, stage_args):
    root = build_paths()["root"]
    env = os.environ.copy()
    parent = str(root.parent)
    env["PYTHONPATH"] = parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(
        [sys.executable, "-m", module_name, *stage_args],
        cwd=root,
        env=env,
        check=False,
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def run_all(stage_args):
    for stage in ["build_scenarios", "run_scenarios", "stats"]:
        print(f"[stage] {stage}")
        _run_module(stage_modules[stage], stage_args)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m sfincs_runs {build_scenarios|run_scenarios|stats|all} [args...]")
        raise SystemExit(2)

    stage, stage_args = argv[0], argv[1:]
    if stage == "all":
        run_all(stage_args)
        return
    if stage not in stage_modules:
        raise SystemExit(f"Unknown stage: {stage}")

    _run_module(stage_modules[stage], stage_args)


if __name__ == "__main__":
    main()
