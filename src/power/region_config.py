"""Static configuration for SMART-DS Regional Case Domains used as audit references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegionConfig:
    """Static configuration for one Regional Case Domain."""

    region_id: str
    smart_ds_code: str
    smart_ds_subregions: tuple[str, ...]
    artifact_root: Path

    @property
    def smart_ds_manifest_path(self) -> Path:
        return self.artifact_root / "smart_ds" / "manifest.json"


_sfo_subregions = tuple(
    [f"P{i}R" for i in range(1, 6)] + [f"P{i}U" for i in range(1, 36)]
)
_austin_subregions = ("P1R", "P1U", "P2U", "P3U", "P4U", "P5U")
_greensboro_subregions = ("industrial", "rural", "urban-suburban")


def _region(region_id: str, code: str, subregions: tuple[str, ...]) -> RegionConfig:
    return RegionConfig(
        region_id,
        code,
        subregions,
        Path("artifacts") / "regions" / region_id,
    )


regions: dict[str, RegionConfig] = {
    "sfo": _region("sfo", "SFO", _sfo_subregions),
    "austin": _region("austin", "AUS", _austin_subregions),
    "greensboro": _region("greensboro", "GSO", _greensboro_subregions),
}


def get_region_config(region_id: str) -> RegionConfig:
    """Return the configured Regional Case Domain."""
    if region_id not in regions:
        valid = ", ".join(sorted(regions))
        raise ValueError(f"unknown region_id {region_id!r}; expected one of: {valid}")
    return regions[region_id]
