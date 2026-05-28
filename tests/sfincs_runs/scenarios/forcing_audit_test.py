import json

import numpy as np

from sfincs_runs.scenarios import audit_forcing_manifest


def test_audit_forcing_manifest_flags_stale_forcing_artifacts(tmp_path):
    run_root = tmp_path / "evt_0003"
    run_root.mkdir()
    (run_root / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "event_id": "evt_0003",
                "rainfall_member_id": "aorc_20180102",
                "prepared_precip": "aorc_precip_for_sfincs.nc",
                "netamprfile": "sfincs_netampr.nc",
            }
        ),
        encoding="utf-8",
    )
    np.array([360.0, 360.0, 360.0, 10.0], dtype="<f4").tofile(run_root / "sfincs.ks")
    (run_root / "snapwave.bwd").write_text(
        "0 350 350\n3600 2 2\n",
        encoding="utf-8",
    )

    audit = audit_forcing_manifest(run_root)

    assert not audit.passed
    assert {
        "missing_timing_policy",
        "missing_run_window",
        "missing_rainfall_window_alignment",
        "ksat_cap_fraction_high",
        "wave_direction_wrap_for_plotting",
    } <= audit.issue_codes
    assert audit.issue("ksat_cap_fraction_high").severity == "error"
    assert audit.issue("wave_direction_wrap_for_plotting").severity == "warning"


def test_audit_forcing_manifest_passes_modern_manifest_with_conditioned_ksat(tmp_path):
    run_root = tmp_path / "evt_0004"
    run_root.mkdir()
    (run_root / "forcing_manifest.json").write_text(
        json.dumps(
            {
                "event_id": "evt_0004",
                "timing_policy": "descriptors",
                "event_reference_time": "2000-01-02 12:00:00",
                "run_start": "2000-01-01 00:00:00",
                "run_stop": "2000-01-04 00:00:00",
                "run_duration_hours": 72,
                "expected_has_precip": True,
                "rainfall_window_alignment": "run_window",
            }
        ),
        encoding="utf-8",
    )
    np.array([5.0, 15.0, 25.0, 75.0], dtype="<f4").tofile(run_root / "sfincs.ks")
    (run_root / "snapwave.bwd").write_text(
        "0 20 20\n3600 25 25\n",
        encoding="utf-8",
    )

    audit = audit_forcing_manifest(run_root)

    assert audit.passed
    assert audit.error_codes == set()
