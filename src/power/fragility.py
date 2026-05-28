"""Marshfield defaults for shared ERAD flood-depth fragility curves."""

from __future__ import annotations

from pathlib import Path

from fragility.flood_depth import DEFAULT_CURVES_CSV
from fragility.flood_depth import FloodDepthFragilityCurve
from fragility.flood_depth import erad_asset_type as _erad_asset_type
from fragility.flood_depth import failure_probability as _failure_probability
from fragility.flood_depth import line_local_asset_type
from fragility.flood_depth import load_asset_type_mapping as _load_asset_type_mapping
from fragility.flood_depth import load_flood_depth_curves
from power.paths import POWER_GRID


FRAGILITY_DIR = POWER_GRID / "fragility"
DEFAULT_MAPPING_CSV = FRAGILITY_DIR / "asset_type_mapping.csv"


def load_asset_type_mapping(path: str | Path = DEFAULT_MAPPING_CSV) -> dict[str, str]:
    """Load the Marshfield Asset Registry type to ERAD asset type mapping."""
    return _load_asset_type_mapping(path)


def erad_asset_type(
    local_asset_type: str, mapping: dict[str, str] | None = None
) -> str:
    """Map a Marshfield local asset type name to the ERAD curve asset type."""
    return _erad_asset_type(
        local_asset_type,
        mapping=mapping or load_asset_type_mapping(),
    )


def failure_probability(
    local_asset_type: str,
    depth_m: float | int | None,
    *,
    curves: dict[str, FloodDepthFragilityCurve] | None = None,
    mapping: dict[str, str] | None = None,
) -> float:
    """Return ERAD-derived flood failure probability for a Marshfield asset."""
    return _failure_probability(
        local_asset_type,
        depth_m,
        curves=curves,
        mapping=mapping or load_asset_type_mapping(),
    )
