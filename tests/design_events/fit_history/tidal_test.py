import numpy as np
import pandas as pd

from design_events.fit_history.tidal import coastal_components, non_tidal_residual


def _synthetic_water_level(days=120, surge_peak=1.0, seed=0):
    # hourly total water level = slow MSL + M2/S2 tide + an injected surge bump + noise
    idx = pd.date_range("2015-01-01", periods=days * 24, freq="1h")
    h = (idx - idx[0]) / pd.Timedelta(hours=1)
    msl = 0.10 + 0.05 * np.sin(2 * np.pi * h / (30 * 24))          # slow seasonal MSL
    tide = 1.0 * np.cos(2 * np.pi * h / 12.42) + 0.3 * np.cos(2 * np.pi * h / 12.0)  # M2 + S2
    surge = surge_peak * np.exp(-0.5 * ((h - days * 12) / 18.0) ** 2)  # a storm-surge bump
    rng = np.random.default_rng(seed)
    return pd.Series(msl + tide + surge + 0.02 * rng.standard_normal(len(idx)), index=idx)


def test_coastal_components_reconstruct_total_water_level():
    wl = _synthetic_water_level()
    comp = coastal_components(wl, latitude=42.1)
    # identity: total water level is exactly mean sea level + tide + non-tidal residual
    recon = comp["msl"] + comp["tide"] + comp["ntr"]
    assert np.allclose(wl.reindex(comp.index).to_numpy(), recon.to_numpy(), atol=1e-9)


def test_ntr_captures_surge_not_tide():
    wl = _synthetic_water_level(surge_peak=1.2)
    comp = coastal_components(wl, latitude=42.1)
    # the fitted tide carries the ~1.3 m astronomical swing
    assert comp["tide"].max() - comp["tide"].min() > 2.0
    # the surge lives in the NTR: its peak is well above the quiet-period NTR scatter
    ntr = non_tidal_residual(wl, latitude=42.1)
    assert ntr.max() > 0.8
    assert ntr.std() < 0.3  # away from the bump the residual is small (tide removed)
