"""Coastal-only helpers for the ADR-0020 reference bundle.

The v2 coastal contract is narrow: use NTR/surge as the stochastic Driver
Probability Index and preserve mean sea level + astronomical tide unscaled in metadata
and audit checks. This module does not write SFINCS boundary forcing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from design_events_v2.records import coastal_components, non_tidal_residual


def tide_preserving_total_water_level(components, *, ntr_scale_factor=1.0, msl_shift=0.0):
    """Rebuild total water level as ``MSL + tide + K * NTR + msl_shift``.

    Only the NTR/surge component is scaled. Tide remains unchanged by construction.
    """

    frame = pd.DataFrame(components)
    required = {"msl", "tide", "ntr"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("coastal components missing columns: " + ", ".join(sorted(missing)))
    return (
        pd.to_numeric(frame["msl"], errors="coerce")
        + float(msl_shift)
        + pd.to_numeric(frame["tide"], errors="coerce")
        + float(ntr_scale_factor) * pd.to_numeric(frame["ntr"], errors="coerce")
    )


def coastal_realization_metadata(events, drivers, components=None, config=None):
    """Return coastal realization audit metadata from v2 bundle tables."""

    driver_frame = pd.DataFrame(drivers)
    coastal = driver_frame[driver_frame.get("driver", pd.Series(dtype=str)).astype(str).isin(["coastal_ntr", "coastal_water_level"])]
    scale = pd.to_numeric(coastal.get("scale_factor", pd.Series(dtype=float)), errors="coerce")
    metadata = {
        "coastal_driver": "coastal_ntr" if "coastal_ntr" in set(coastal.get("driver", [])) else "coastal_water_level",
        "tide_preserved_unscaled": True,
        "scaled_component": "non_tidal_residual",
        "ntr_scale_factor_min": float(scale.min()) if scale.notna().any() else None,
        "ntr_scale_factor_max": float(scale.max()) if scale.notna().any() else None,
        "realized_member_count": int(coastal.get("member_id", pd.Series(dtype=object)).nunique()),
    }

    if components is not None:
        frame = pd.DataFrame(components).copy()
        if {"wl", "msl", "tide", "ntr"}.issubset(frame.columns):
            recon = frame["msl"] + frame["tide"] + frame["ntr"]
            metadata["component_reconstruction_max_abs_error"] = float((frame["wl"] - recon).abs().max())
        elif {"msl", "tide", "ntr"}.issubset(frame.columns):
            metadata["component_reconstruction_max_abs_error"] = 0.0
        metadata["component_rows"] = int(len(frame))
    else:
        metadata["component_reconstruction_max_abs_error"] = None
        metadata["component_rows"] = 0

    return metadata


__all__ = [
    "coastal_components",
    "coastal_realization_metadata",
    "non_tidal_residual",
    "tide_preserving_total_water_level",
]
