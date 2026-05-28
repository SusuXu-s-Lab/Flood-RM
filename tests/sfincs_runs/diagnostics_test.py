import json

import pandas as pd

from sfincs_runs.diagnostics import _resolve_run_start


def test_resolve_run_start_prefers_forcing_manifest(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "forcing_manifest.json").write_text(
        json.dumps({"run_start": "2018-01-01 17:00:00"}),
        encoding="utf-8",
    )
    (run_root / "sfincs.inp").write_text("tstart = 20000101 000000\n", encoding="utf-8")

    assert _resolve_run_start(run_root, fallback="1999-01-01") == pd.Timestamp("2018-01-01 17:00:00")


def test_resolve_run_start_falls_back_to_sfincs_inp(tmp_path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "sfincs.inp").write_text("tstart = 20180101 170000\n", encoding="utf-8")

    assert _resolve_run_start(run_root, fallback="1999-01-01") == pd.Timestamp("2018-01-01 17:00:00")
