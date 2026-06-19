import numpy as np
import pandas as pd

from design_events.build_events.coastal import (
    build_sampled_peaks,
    hybrid_peak_sample,
    hybrid_peak_sample_frame,
)


class FakeMarginal:
    def return_period(self, h):
        return np.asarray(h, dtype=float) * 10.0

    def magnitude(self, rps):
        return np.full(np.asarray(rps, dtype=float).shape, 99.0)


class IdentityMarginal:
    def return_period(self, h):
        return np.asarray(h, dtype=float)

    def magnitude(self, rps):
        return np.asarray(rps, dtype=float)


class SubannualBodyMarginal:
    def return_period(self, h):
        return np.asarray(h, dtype=float) / 10.0

    def magnitude(self, rps):
        return np.full(np.asarray(rps, dtype=float).shape, 99.0)


def test_hybrid_peak_sample_can_oversample_tail():
    sample = hybrid_peak_sample(
        peaks=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        n_samples=10,
        settings={
            "hybrid_splice_quantile": 0.80,
            "tail_sample_fraction": 0.40,
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
        },
        marginal=FakeMarginal(),
        seed=42,
    )

    assert int(np.sum(sample == 99.0)) == 4


def test_tail_oversampling_records_sampling_weights():
    frame = hybrid_peak_sample_frame(
        peaks=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        n_samples=10,
        settings={
            "hybrid_splice_quantile": 0.80,
            "tail_sample_fraction": 0.40,
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
        },
        marginal=FakeMarginal(),
        seed=42,
    )

    assert frame["sampling_region"].value_counts().to_dict() == {"body": 6, "tail": 4}
    assert set(frame.loc[frame["sampling_region"] == "body", "sampling_weight"].round(6)) == {1.333333}
    assert set(frame.loc[frame["sampling_region"] == "tail", "sampling_weight"].round(6)) == {0.5}
    assert round(frame["sampling_weight"].mean(), 6) == 1.0
    assert set(frame["catalog_role"]) == {"probability"}
    assert set(frame["sampling_scheme"]) == {"tail_enriched_hybrid"}
    assert set(frame["event_origin"]) == {"historical_bootstrap_body", "synthetic_tail"}


def test_hybrid_peak_sample_defaults_to_probability_tail_mass():
    frame = hybrid_peak_sample_frame(
        peaks=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        n_samples=20,
        settings={
            "hybrid_splice_quantile": 0.80,
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
        },
        marginal=FakeMarginal(),
        seed=42,
    )

    assert frame["sampling_region"].value_counts().to_dict() == {"body": 16, "tail": 4}
    assert set(frame["sampling_scheme"]) == {"probability_proportional_hybrid"}


def test_hybrid_peak_sample_can_select_design_events_from_candidate_pool():
    frame = hybrid_peak_sample_frame(
        peaks=np.linspace(1.0, 10.0, 200),
        n_samples=20,
        settings={
            "hybrid_splice_quantile": 0.80,
            "candidate_pool_count": 1000,
            "catalog_band_fractions": {
                "mild": 0.10,
                "common": 0.20,
                "significant": 0.30,
                "rare": 0.20,
                "extreme": 0.20,
            },
            "severity_bands": [
                {"severity_band": "mild", "rp_min_years": 0.0, "rp_max_years": 2.0},
                {"severity_band": "common", "rp_min_years": 2.0, "rp_max_years": 10.0},
                {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
                {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
                {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
            ],
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
        },
        marginal=IdentityMarginal(),
        seed=42,
    )

    assert len(frame) == 20
    assert set(frame["catalog_role"]) == {"design"}
    assert set(frame["sampling_scheme"]) == {"band_stratified_importance_from_candidate_pool"}
    assert round(frame["probability_weight"].sum(), 6) == 1.0
    assert frame["candidate_pool_count"].eq(1000).all()


def test_tail_oversampling_records_normalized_probability_weights():
    frame = hybrid_peak_sample_frame(
        peaks=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        n_samples=10,
        settings={
            "hybrid_splice_quantile": 0.80,
            "tail_sample_fraction": 0.40,
            "return_period_min_years": 1.5,
            "return_period_max_years": 500.0,
            "spacing": "log",
        },
        marginal=IdentityMarginal(),
        seed=42,
    )

    body = frame[frame["sampling_region"] == "body"]
    tail = frame[frame["sampling_region"] == "tail"].sort_values("peak_m")

    assert round(frame["probability_weight"].sum(), 6) == 1.0
    assert round(body["probability_weight"].sum(), 6) == 0.8
    assert round(tail["probability_weight"].sum(), 6) == 0.2
    assert set(body["probability_weight"].round(6)) == {0.133333}
    assert tail["probability_weight"].is_monotonic_decreasing
    assert len(set(tail["probability_weight"].round(6))) > 1


def test_hybrid_peak_sample_keeps_empirical_body_when_body_rps_are_below_tail_domain():
    frame = hybrid_peak_sample_frame(
        peaks=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        n_samples=10,
        settings={
            "hybrid_splice_quantile": 0.80,
            "tail_sample_fraction": 0.40,
            "return_period_min_years": 10.0,
            "return_period_max_years": 20.0,
        },
        marginal=SubannualBodyMarginal(),
        seed=42,
    )

    assert frame["sampling_region"].value_counts().to_dict() == {"body": 6, "tail": 4}
    assert int((frame["peak_m"] == 99.0).sum()) == 4
    assert frame.loc[frame["sampling_region"] == "body", "peak_m"].max() <= 4.2
    assert set(frame.loc[frame["sampling_region"] == "body", "sampling_weight"].round(6)) == {1.333333}
    assert set(frame.loc[frame["sampling_region"] == "tail", "sampling_weight"].round(6)) == {0.5}


def test_build_sampled_peaks_preserves_probability_weight_precision(tmp_path, monkeypatch):
    peaks_csv = tmp_path / "historical_peaks.csv"
    sampled_csv = tmp_path / "sampled_peaks.csv"
    pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=100, freq="D"),
            "h": np.arange(1, 101, dtype=float),
        }
    ).to_csv(peaks_csv, index=False)
    paths = {
        "historical_peaks_csv": peaks_csv,
        "marginal_params_csv": tmp_path / "marginal_params.csv",
        "sampled_peaks_csv": sampled_csv,
    }
    config = {
        "events": {"target_event_count": 2500},
        "template_assignment": {"random_seed": 42},
        "sampling": {
            "hybrid_splice_quantile": 0.95,
            "tail_sample_fraction": 0.20,
            "return_period_min_years": 1.5,
            "return_period_max_years": 250.0,
            "spacing": "log",
        },
    }
    monkeypatch.setattr(
        "design_events.build_events.coastal.load_historical_peak_marginal",
        lambda path: IdentityMarginal(),
    )

    build_sampled_peaks(config, paths)
    written = np.genfromtxt(sampled_csv, delimiter=",", names=True, dtype=None, encoding=None)

    assert round(float(written["probability_weight"].sum()), 6) == 1.0
    assert "event_origin" in written.dtype.names
    assert "catalog_role" in written.dtype.names
    assert "sampling_scheme" in written.dtype.names
