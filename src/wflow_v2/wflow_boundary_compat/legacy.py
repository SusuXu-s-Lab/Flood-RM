from __future__ import annotations

from pathlib import Path


def removed_legacy_repair(*_args, **_kwargs):
    """Compatibility marker for code that no longer belongs in the docs component."""
    raise RuntimeError(
        "Legacy Wflow staticmap repair code has been removed from wflow_boundary. "
        "Rebuild with current HydroMT-Wflow, or place project-specific migration repairs "
        "in an external migration package before calling the boundary component."
    )


def normalize_wflow_staticmaps_nodata(model_root: str | Path):
    return removed_legacy_repair(model_root)


def repair_wflow_river_width(model_root: str | Path):
    return removed_legacy_repair(model_root)


def repair_wflow_canopy_parameters(model_root: str | Path):
    return removed_legacy_repair(model_root)


def repair_wflow_gauge_map(model_root: str | Path):
    return removed_legacy_repair(model_root)
