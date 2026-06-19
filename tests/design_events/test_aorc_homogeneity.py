import pandas as pd

from design_events.collect_sources.aorc_sst import summarize_homogeneity


def test_summarize_homogeneity_flags_depth_and_season_spread(tmp_path):
    samples = pd.DataFrame(
        {
            "sample_id": ["target", "near", "far"],
            "sample_role": ["target", "candidate", "candidate"],
            "max_72h_mm": [100.0, 120.0, 260.0],
            "max_month": [1, 1, 8],
            "distance_km": [0.0, 50.0, 200.0],
        }
    )

    summary = summarize_homogeneity(samples)

    assert summary["target_max_72h_mm"] == 100.0
    assert summary["candidate_count"] == 2
    assert summary["max_ratio_to_target"] == 2.6
    assert summary["months_observed"] == [1, 8]
    assert summary["review_required"] is True
