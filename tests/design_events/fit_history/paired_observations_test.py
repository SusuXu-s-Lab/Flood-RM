import numpy as np
import pandas as pd
import pyvinecopulib as pv
from scipy.stats import kendalltau

from design_events.fit_history.paired_observations import (
    build_paired_observations,
    calibrate_threshold_for_rate,
    declustered_pot_peaks,
)
from design_events.build_events.probability.dependence import fit_driver_dependence
from design_events.fit_history.return_curve import HistoricalPeakMarginal


def _correlated_driver_frame(n_events=120, rho=0.8, years=6, seed=0):
    idx = pd.date_range("2000-01-01", periods=years * 365 * 24, freq="1h")
    rng = np.random.default_rng(seed)
    rainfall = pd.Series(rng.gamma(1.0, 2.0, len(idx)), index=idx)
    surge = pd.Series(rng.gamma(1.0, 0.05, len(idx)), index=idx)
    positions = rng.choice(len(idx), size=n_events, replace=False)
    z = rng.standard_normal((n_events, 2))
    z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
    rainfall.iloc[positions] += 40.0 + 20.0 * np.maximum(z[:, 0], -1.5)
    surge.iloc[positions] += 0.6 + 0.4 * np.maximum(z[:, 1], -1.5)
    return pd.DataFrame({"rainfall": rainfall, "surge": surge})


def test_declustered_pot_peaks_respect_threshold_and_separation():
    frame = _correlated_driver_frame(seed=1)
    peaks = declustered_pot_peaks(frame["rainfall"], threshold_quantile=0.98, min_separation_hours=120.0)
    assert len(peaks) > 10
    thr = float(frame["rainfall"].quantile(0.98))
    assert (peaks["value"] > thr).all()
    gaps = np.diff(np.sort(peaks["time"].to_numpy())).astype("timedelta64[h]").astype(float)
    assert np.all(gaps >= 120.0)


def test_build_paired_observations_two_sided_and_dependent():
    frame = _correlated_driver_frame(rho=0.8, seed=2)
    paired = build_paired_observations(
        frame, decluster_window_hours=120.0, pairing_window_hours=72.0, threshold_quantiles=0.98
    )

    assert list(paired.columns) == ["event_time", "conditioned_on", "rainfall", "surge", "rainfall_time", "surge_time"]
    assert paired[["rainfall_time", "surge_time"]].notna().all().all()
    # both conditioning directions are present (the "two-sided" sample)
    assert set(paired["conditioned_on"].unique()) == {"rainfall", "surge"}
    assert len(paired) > 20
    # the co-occurrence sample carries the injected positive dependence
    tau, _ = kendalltau(paired["rainfall"], paired["surge"])
    assert tau > 0.1


def test_paired_observations_feed_the_vine_fit():
    frame = _correlated_driver_frame(rho=0.8, seed=3)
    paired = build_paired_observations(frame)
    marginals = [HistoricalPeakMarginal("exp", (0.0, 40.0, 15.0), 5.0, "pot", 0.98, 200),
                 HistoricalPeakMarginal("exp", (0.0, 0.5, 0.3), 5.0, "pot", 0.98, 200)]
    model = fit_driver_dependence(
        paired[["rainfall", "surge"]].to_numpy(),
        marginals=marginals,
        driver_names=["rainfall", "surge"],
        event_rate=5.0,
        seed=1,
    )
    assert model.dim == 2
    # the fitted vine reproduces positive dependence
    sample = np.asarray(model.vine.simulate(20_000, seeds=[7]), dtype=float)
    tau, _ = kendalltau(sample[:, 0], sample[:, 1])
    assert tau > 0.1


def test_calibrate_threshold_for_rate_hits_target():
    # Fix 1: the threshold is chosen *to obtain* the target declustered rate, not fixed.
    frame = _correlated_driver_frame(years=6, seed=4)
    threshold, peaks, record_years = calibrate_threshold_for_rate(
        frame["rainfall"], target_rate_per_year=5.0, min_separation_hours=120.0
    )
    assert abs(record_years - 6.0) < 0.1
    assert len(peaks) == round(5.0 * record_years)  # exactly the top-N declustered peaks
    assert (peaks["value"] >= threshold).all()
    gaps = np.diff(np.sort(peaks["time"].to_numpy())).astype("timedelta64[h]").astype(float)
    assert np.all(gaps >= 120.0)


def test_build_paired_observations_records_realized_rate_and_condition_on():
    frame = _correlated_driver_frame(years=6, seed=5)
    paired = build_paired_observations(
        frame, condition_on=["surge"], target_rate_per_year=5.0, decluster_window_hours=120.0
    )
    # only the named extreme driver seeds conditioning peaks
    assert set(paired["conditioned_on"].unique()) == {"surge"}
    # the realized distinct-storm rate (used in T = 1/(rate*S)) matches the calibration target
    assert abs(paired.attrs["base_event_rate_per_year"] - 5.0) < 1.0


def test_build_paired_observations_handles_duplicate_partner_timestamps():
    event_time = pd.Timestamp("2000-01-10 00:00:00")
    surge = pd.Series(
        [0.1, 2.0, 0.2],
        index=pd.to_datetime(["2000-01-09 00:00:00", event_time, "2000-01-11 00:00:00"]),
    )
    rainfall = pd.Series(
        [8.0, 14.0, 6.0],
        index=pd.to_datetime(["2000-01-10 03:00:00", "2000-01-10 03:00:00", "2000-01-10 04:00:00"]),
    )

    paired = build_paired_observations(
        {"surge": surge, "rainfall": rainfall},
        condition_on=["surge"],
        threshold_quantiles=0.5,
        decluster_window_hours=24.0,
        pairing_window_hours=12.0,
    )

    row = paired.iloc[0]
    assert row["rainfall_time"] == pd.Timestamp("2000-01-10 03:00:00")
    assert float(row["rainfall"]) == 14.0


def test_empty_when_no_exceedances():
    flat = pd.Series(np.zeros(1000), index=pd.date_range("2000-01-01", periods=1000, freq="1h"))
    peaks = declustered_pot_peaks(flat, threshold=10.0)
    assert peaks.empty
