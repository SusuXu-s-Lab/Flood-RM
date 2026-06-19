import json

import pandas as pd

from design_events.build_events.selection import (
    assign_severity_bands,
    summarize_event_distribution,
    write_event_distribution_artifacts,
)


def test_assign_severity_bands_uses_return_period_bins():
    bands = assign_severity_bands(pd.Series([1.2, 2.0, 25.0, 99.0, 100.0, 700.0]))

    assert bands.tolist() == [
        "mild",
        "common",
        "significant",
        "rare",
        "extreme",
        "beyond_design",
    ]


def test_summarize_event_distribution_reports_weighted_and_raw_counts():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003", "evt_0004"],
            "sample_rp_years": [1.5, 8.0, 75.0, 250.0],
            "sampling_region": ["body", "body", "tail", "tail"],
            "sampling_weight": [1.4, 1.4, 0.4, 0.4],
            "coastal_peak_m": [0.8, 1.0, 1.7, 2.2],
        }
    )

    summary = summarize_event_distribution(catalog)

    row = summary.set_index("severity_band").loc["extreme"]
    assert row["event_count"] == 1
    assert row["unweighted_fraction"] == 0.25
    assert row["weighted_count"] == 0.4
    assert round(row["weighted_fraction"], 6) == round(0.4 / 3.6, 6)
    assert row["tail_count"] == 1
    assert row["peak_max_m"] == 2.2


def test_summarize_event_distribution_separates_sampling_weight_from_probability_mass():
    catalog = pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003", "evt_0004"],
            "sample_rp_years": [1.5, 8.0, 75.0, 250.0],
            "sampling_region": ["body", "body", "tail", "tail"],
            "sampling_weight": [1.0, 1.0, 1.0, 1.0],
            "probability_weight": [0.70, 0.10, 0.05, 0.15],
            "coastal_peak_m": [0.8, 1.0, 1.7, 2.2],
        }
    )

    summary = summarize_event_distribution(catalog)

    row = summary.set_index("severity_band").loc["extreme"]
    assert row["event_count"] == 1
    assert row["weighted_count"] == 1.0
    assert row["probability_mass"] == 0.15
    assert row["probability_fraction"] == 0.15


def test_write_event_distribution_artifacts_reads_sampled_peaks_without_catalog(tmp_path):
    paths = {
        "sampled_peaks_csv": tmp_path / "catalog/sampled_peaks.csv",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_distribution_summary_csv": tmp_path / "catalog/event_distribution_summary.csv",
        "event_distribution_summary_json": tmp_path / "catalog/event_distribution_summary.json",
        "event_distribution_plot_png": tmp_path / "catalog/event_distribution.png",
    }
    paths["sampled_peaks_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001", "evt_0002", "evt_0003"],
            "sample_rp_years": [1.5, 25.0, 150.0],
            "sampling_region": ["body", "body", "tail"],
            "sampling_weight": [1.2, 1.2, 0.6],
            "peak_m": [0.8, 1.3, 2.0],
        }
    ).to_csv(paths["sampled_peaks_csv"], index=False)

    summary = write_event_distribution_artifacts({}, paths)

    assert paths["event_distribution_summary_csv"].exists()
    assert paths["event_distribution_summary_json"].exists()
    assert paths["event_distribution_plot_png"].exists()
    assert summary["source_file"] == str(paths["sampled_peaks_csv"])
    assert summary["event_count"] == 3
    payload = json.loads(paths["event_distribution_summary_json"].read_text())
    assert payload["bands"][0]["severity_band"] == "mild"


def test_write_event_distribution_artifacts_uses_sampled_peaks_when_catalog_is_stale(tmp_path):
    paths = {
        "sampled_peaks_csv": tmp_path / "catalog/sampled_peaks.csv",
        "event_catalog_csv": tmp_path / "catalog/event_catalog.csv",
        "event_distribution_summary_csv": tmp_path / "catalog/event_distribution_summary.csv",
        "event_distribution_summary_json": tmp_path / "catalog/event_distribution_summary.json",
        "event_distribution_plot_png": tmp_path / "catalog/event_distribution.png",
    }
    paths["sampled_peaks_csv"].parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "sample_rp_years": [150.0],
            "sampling_region": ["tail"],
            "sampling_weight": [0.25],
            "peak_m": [2.1],
        }
    ).to_csv(paths["sampled_peaks_csv"], index=False)
    pd.DataFrame(
        {
            "event_id": ["evt_0001"],
            "sample_rp_years": [150.0],
            "coastal_peak_m": [2.1],
        }
    ).to_csv(paths["event_catalog_csv"], index=False)

    summary = write_event_distribution_artifacts({}, paths)

    assert summary["source"] == "sampled_peaks"
    payload = json.loads(paths["event_distribution_summary_json"].read_text())
    extreme = [row for row in payload["bands"] if row["severity_band"] == "extreme"][0]
    assert extreme["tail_count"] == 1
