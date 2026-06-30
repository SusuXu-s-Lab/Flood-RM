"""Coastal Field-Preserving Realization for copula-joint catalog rows.

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

import pandas as pd

from sfincs_v2.coastal import build_coastal_hydrograph_from_analog as _v2_build_coastal_hydrograph_from_analog


def build_coastal_hydrograph_from_analog(components, peak_time, scale_factor, *, window_hours=72.0, msl_offset_m=0.0):
    """Total water level over a window centred on a surge peak, with only NTR scaled.

    ``components`` is the tide/NTR decomposition frame (columns ``msl``, ``tide``, ``ntr`` on a
    DatetimeIndex, from ``fit_history.tidal.coastal_components``). Returns a Series indexed by
    relative hour from the peak.
    """
    return _v2_build_coastal_hydrograph_from_analog(
        components,
        peak_time,
        scale_factor,
        window_hours=window_hours,
        msl_offset_m=msl_offset_m,
        return_absolute_time=False,
    )


def build_timeseries(
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
