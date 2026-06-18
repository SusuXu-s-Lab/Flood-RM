"""Coastal Field-Preserving Realization for copula-joint catalog rows (ADR-0011, Fix 2).

Builds the coastal SFINCS water-level boundary for a copula-joint event. Each event names an
observed surge (NTR) analog — its declustered POT peak time — and a scale factor ``K``. We cut
the observed window around that peak from the tide/NTR decomposition and rebuild total water
level as ``MSL + tide + K*NTR``: only the non-tidal residual (surge) is scaled, while the
astronomical tide and mean sea level are added back unscaled, so the observed tidal modulation
and tide-surge phasing are preserved and never amplified by ``K`` (the macro-tidal failure mode
of scaling total water level). SLR enters as a rigid MSL translation (``msl_offset_m``). This is
the coastal analogue of the AORC SST field-scaling workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_coastal_hydrograph_from_analog(components, peak_time, scale_factor, *, window_hours=72.0, msl_offset_m=0.0):
    """Total water level over a window centred on a surge peak, with only NTR scaled.

    ``components`` is the tide/NTR decomposition frame (columns ``msl``, ``tide``, ``ntr`` on a
    DatetimeIndex, from ``fit_history.tidal.coastal_components``). Returns a Series indexed by
    relative hour from the peak.
    """
    if not {"msl", "tide", "ntr"}.issubset(components.columns):
        raise ValueError("components must have msl/tide/ntr columns from coastal_components")
    scale = float(scale_factor)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"scale_factor must be finite and > 0, got {scale_factor!r}")
    peak_time = pd.Timestamp(peak_time)
    half = pd.Timedelta(hours=float(window_hours))
    window = components.loc[peak_time - half : peak_time + half]
    if window.empty:
        raise ValueError(f"no coastal components within +/-{window_hours}h of {peak_time}")

    # rebuild total water level: tide + MSL unscaled, surge (NTR) scaled by K, plus SLR offset.
    h = window["msl"].to_numpy() + window["tide"].to_numpy() + scale * window["ntr"].to_numpy() + float(msl_offset_m)
    rel_hours = np.round((window.index - peak_time) / pd.Timedelta(hours=1)).astype(int)
    out = pd.Series(h, index=pd.Index(rel_hours, name="relative_hour"))
    return out[~out.index.duplicated(keep="first")].sort_index()


def build_coastal_event_timeseries(
    row,
    components,
    *,
    member_time_column="coastal_water_level_member_time",
    scale_column="coastal_water_level_scale_factor",
    window_hours=72.0,
    msl_offset_m=0.0,
):
    """Coastal SFINCS boundary series for one copula-joint catalog row.

    Drop-in for ``scenarios.build_event_timeseries`` returning ``{"h": series, ...}``.
    """
    peak_time = row.get(member_time_column)
    if peak_time is None or (isinstance(peak_time, float) and pd.isna(peak_time)):
        raise ValueError(f"catalog row is missing {member_time_column!r}")
    scale = row.get(scale_column, 1.0)
    series = build_coastal_hydrograph_from_analog(
        components, peak_time, scale, window_hours=window_hours, msl_offset_m=msl_offset_m
    )
    return {
        "h": series,
        "forcing_variable": "coastal_water_level",
        "analog_peak_time": str(peak_time),
        "scale_factor": float(scale),
        "msl_offset_m": float(msl_offset_m),
    }
