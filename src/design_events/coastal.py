"""Coastal-only helpers for the reference bundle.

The v2 coastal contract is narrow: use NTR/surge as the stochastic Driver
Probability Index and preserve mean sea level + astronomical tide unscaled in metadata
and audit checks. This module does not write SFINCS boundary forcing.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


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


# --------------------------------------------------------------------------------------
# Production coastal hybrid sampler + surge hydrograph templates/members (moved from the
# legacy nested coastal builders). NTR/tide contract preserved: the copula
# and sampler use NTR; tide rides back unscaled in the realized total water level.
# --------------------------------------------------------------------------------------


def _round_to_total(fractions, total):
    raw = {name: float(frac) * total for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = int(total) - sum(counts.values())
    for name in sorted(raw, key=lambda value: raw[value] - counts[value], reverse=True):
        if remainder <= 0:
            break
        counts[name] += 1
        remainder -= 1
    return counts


# Surge hydrograph templates and event-member artifacts


template_columns = [
    "peak_time",
    "baseline_m",
    "threshold_m",
    "absolute_peak_m",
    "peak_m",
    "volume",
    "duration_above_50pct_peak",
    "rise_time_to_peak",
    "fall_time_from_peak",
    "asymmetry_ratio",
    "n_secondary_peaks",
    "valid_start_hour",
    "valid_end_hour",
]

member_columns = [
    "sample_rp_years",
    "sampling_region",
    "sampling_weight",
    "probability_weight",
    "template_id",
    "template_peak_m",
    "template_peak_time",
    "tail_morph_factor",
    "peak",
    "volume",
    "duration_above_50pct_peak",
    "rise_time_to_peak",
    "fall_time_from_peak",
    "asymmetry_ratio",
    "n_secondary_peaks",
    "valid_start_hour",
    "valid_end_hour",
]

def _write_netcdf_replace(dataset, path):
    # Avoid corrupting an existing NetCDF if a write fails halfway through.
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.unlink(missing_ok=True)
        dataset.to_netcdf(tmp_path)
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)

__all__ = [
    "coastal_components",
    "coastal_realization_metadata",
    "non_tidal_residual",
    "tide_preserving_total_water_level",
    "sample_return_periods",
    "bootstrap_body_sample",
    "hybrid_peak_sample",
    "hybrid_peak_sample_frame",
    "build_sampled_peaks",
    "extract_historical_templates",
    "template_bank_to_dataset",
    "build_surge_event_members",
    "build_acceptance_report",
    "write_overview_plot",
    "build_surge_event_artifacts",
    "write_event_artifacts",
    "template_columns",
    "member_columns",
]
