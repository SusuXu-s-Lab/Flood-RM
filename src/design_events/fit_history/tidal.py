"""Tide / non-tidal-residual split of the coastal water-level record.

At a macro-tidal site (Marshfield daily tidal range ~2.75 m) the *total* water-level POT
peak is dominated by the astronomical tide, so it barely resolves surge and dilutes the
surge-rainfall dependence the copula is meant to capture. We split CORA total water level
into a slow mean sea level, the astronomical tide (utide harmonic analysis), and the
non-tidal residual (NTR = storm surge). The copula axis and the
realization use NTR; the tide is added back unscaled downstream so it is never amplified by
the surge scale factor. The CORA record itself is unchanged — NTR is computed on the fly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import utide


def coastal_components(waterlevel, *, latitude, msl_window="30D", msl_min_periods=200):
    """Split a total-water-level Series into mean sea level + tide + NTR (surge).

    ``msl`` is a 30-day centered moving average: it removes the secular MSL trend and the
    seasonal sea-level cycle, which also resolves the Stage-2-vs-copula detrend
    inconsistency by construction. ``tide`` is the utide harmonic reconstruction of the
    MSL-removed anomaly; ``ntr = wl - msl - tide``. utide receives the datetime64 index
    directly (passing matplotlib datenum instead needs ``epoch='python'``, which silently
    mis-resolves the constituents). Returns a frame [wl, msl, tide, ntr] over the rows that
    have a full MSL window.
    """
    s = pd.Series(waterlevel).dropna().sort_index()
    if not isinstance(s.index, pd.DatetimeIndex):
        raise ValueError("waterlevel must have a DatetimeIndex")
    msl = s.rolling(msl_window, center=True, min_periods=msl_min_periods).mean()
    anomaly = (s - msl).dropna()
    coef = utide.solve(
        anomaly.index.values, anomaly.to_numpy(dtype=float),
        lat=float(latitude), conf_int="none", trend=False, verbose=False,
    )
    tide = pd.Series(utide.reconstruct(anomaly.index.values, coef, verbose=False).h, index=anomaly.index)
    ntr = anomaly - tide
    return pd.DataFrame(
        {"wl": s.reindex(anomaly.index), "msl": msl.reindex(anomaly.index), "tide": tide, "ntr": ntr}
    )


def non_tidal_residual(waterlevel, *, latitude, **kwargs):
    """Just the NTR (surge) Series — the coastal copula axis and realization index."""
    return coastal_components(waterlevel, latitude=latitude, **kwargs)["ntr"]
