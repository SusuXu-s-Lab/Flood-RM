import numpy as np
import pandas as pd
import pytest

from design_events.build_events.probability.realization import (
    attach_field_preserving_realization,
    draw_relative_lags,
    select_analog_realization,
)


def _rainfall_members(n=200, seed=0):
    rng = np.random.default_rng(seed)
    depth = np.sort(rng.gamma(shape=2.0, scale=40.0, size=n))
    return pd.DataFrame(
        {
            "member_id": [f"rainfall_72h_rank{i:04d}" for i in range(n)],
            "member_file": [f"data/sources/aorc_sst/event_windows/storm_{i:04d}.nc" for i in range(n)],
            "storm_start": pd.date_range("1990-01-01", periods=n, freq="30D").astype(str),
            "mean_precip_mm": depth,
        }
    )


def test_select_analog_picks_near_target_with_scale_near_one():
    members = _rainfall_members()
    values = members["mean_precip_mm"].to_numpy()
    target = float(np.median(values))
    idx, scale = select_analog_realization([target], values, seed=1)
    assert abs(values[idx[0]] - target) <= 0.5 * values.std()
    assert scale[0] == pytest.approx(target / values[idx[0]])
    assert 0.5 < scale[0] < 2.0


def test_above_record_target_scales_up_a_real_member():
    members = _rainfall_members()
    values = members["mean_precip_mm"].to_numpy()
    target = float(values.max() * 1.4)  # above-record design target
    idx, scale = select_analog_realization([target], values, seed=2)
    assert scale[0] > 1.0  # the observed field is scaled up, not invented
    assert np.isfinite(values[idx[0]])


def test_reuse_penalty_diversifies_identical_targets():
    members = _rainfall_members()
    values = members["mean_precip_mm"].to_numpy()
    target = float(np.quantile(values, 0.9))
    idx, _ = select_analog_realization([target] * 25, values, reuse_penalty_lambda=0.5, seed=3)
    # many identical targets should not all collapse onto a single template
    assert len(set(idx.tolist())) > 1


def test_draw_relative_lags_uses_observed_pool_and_default():
    observed = [-24.0, -12.0, 0.0, 6.0, 12.0]
    lags = draw_relative_lags(500, observed_lags=observed, seed=4)
    assert set(np.unique(lags)).issubset(set(observed))
    flat = draw_relative_lags(10, observed_lags=None, default_lag_hours=24.0)
    assert np.all(flat == 24.0)


def test_attach_field_preserving_realization_points_to_fields_not_scalars():
    members = _rainfall_members()
    # a copula-style sampled catalog: scalar target rainfall depths spanning the tail
    catalog = pd.DataFrame(
        {
            "event_id": [f"design_{i:04d}" for i in range(1, 41)],
            "rainfall_mm": np.linspace(40.0, members["mean_precip_mm"].max() * 1.3, 40),
        }
    )
    out = attach_field_preserving_realization(
        catalog,
        members,
        driver="rainfall",
        target_column="rainfall_mm",
        index_column="mean_precip_mm",
        time_column="storm_start",
        design_method="scaled_sst_analog",
        observed_lags=[-12.0, 0.0, 6.0],
        seed=5,
    )

    for column in [
        "rainfall_template_member_id",
        "rainfall_template_value",
        "rainfall_scale_factor",
        "rainfall_design_method",
        "rainfall_member_id",
        "rainfall_member_file",
        "rainfall_member_time",
        "rainfall_realization_lag_hours",
    ]:
        assert column in out.columns

    # every event points to a real observed field (NetCDF) + a scalar scale factor:
    # spatio-temporal structure is preserved, not collapsed.
    assert out["rainfall_member_file"].str.endswith(".nc").all()
    assert out["rainfall_member_id"].notna().all()
    np.testing.assert_allclose(
        out["rainfall_scale_factor"].to_numpy(),
        out["rainfall_mm"].to_numpy() / out["rainfall_template_value"].to_numpy(),
        rtol=1e-9,
    )
    # the highest design target is realized by scaling a real member up
    assert out.sort_values("rainfall_mm")["rainfall_scale_factor"].iloc[-1] > 1.0
    assert out["rainfall_design_method"].eq("scaled_sst_analog").all()
