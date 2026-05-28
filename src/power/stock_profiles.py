"""OEDI ResStock/ComStock profile access for REopt load inputs."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


OEDI_BUCKET_HTTPS_ROOT = "https://oedi-data-lake.s3.amazonaws.com"
EULP_DATASET_PREFIX = (
    "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021"
)
EULP_RELEASE_BY_SOURCE = {
    "comstock": "comstock_amy2018_release_1",
    "resstock": "resstock_amy2018_release_1",
}
PLYMOUTH_COUNTY_GEOID = "G2500230"
EULP_TOTAL_ELECTRICITY_COLUMN = "out.electricity.total.energy_consumption"


@dataclass(frozen=True)
class OediProfileSelection:
    """A selected public EULP building profile."""

    profile_source: str
    release: str
    county_id: str
    building_id: str
    source_building_type: str
    metadata_filter_column: str | None
    profile_url: str
    metadata_url: str
    selection_method: str

    def provenance(self) -> dict[str, str | None]:
        return {
            "profile_source": self.profile_source,
            "profile_source_version": self.release,
            "county_id": self.county_id,
            "building_id": self.building_id,
            "source_building_type": self.source_building_type,
            "metadata_filter_column": self.metadata_filter_column,
            "profile_url": self.profile_url,
            "metadata_url": self.metadata_url,
            "selection_method": self.selection_method,
        }


def metadata_url(profile_source: str) -> str:
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    return f"{OEDI_BUCKET_HTTPS_ROOT}/{EULP_DATASET_PREFIX}/{release}/metadata/metadata.parquet"


def profile_url(profile_source: str, *, county_id: str, building_id: str) -> str:
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    return (
        f"{OEDI_BUCKET_HTTPS_ROOT}/{EULP_DATASET_PREFIX}/{release}/"
        "timeseries_individual_buildings/by_county/upgrade=0/"
        f"county={county_id}/{building_id}-0.parquet"
    )


def download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
            path.write_bytes(response.read())
    return path


def load_oedi_metadata(profile_source: str, *, cache_dir: Path) -> pd.DataFrame:
    url = metadata_url(profile_source)
    path = cache_dir / "metadata" / profile_source / "metadata.parquet"
    download_if_missing(url, path)
    return pd.read_parquet(path)


def select_oedi_profile(
    archetype: dict[str, Any],
    *,
    cache_dir: Path,
    county_id: str = PLYMOUTH_COUNTY_GEOID,
    selection_token: str,
) -> OediProfileSelection:
    """Select a deterministic EULP individual-building profile."""

    profile_source = str(archetype["profile_source"])
    release = EULP_RELEASE_BY_SOURCE[profile_source]
    source_building_type = str(archetype["source_building_type"])
    metadata = load_oedi_metadata(profile_source, cache_dir=cache_dir)

    county_filtered = _filter_county(metadata, county_id)
    type_column = _building_type_column(county_filtered)
    if type_column is not None:
        type_filtered = county_filtered[county_filtered[type_column] == source_building_type]
    else:
        type_filtered = county_filtered
    candidates = type_filtered if not type_filtered.empty else county_filtered
    if candidates.empty:
        raise ValueError(f"No {profile_source} EULP metadata rows found for county {county_id}")

    building_id = _stable_building_id(candidates.index, selection_token)
    return OediProfileSelection(
        profile_source=profile_source,
        release=release,
        county_id=county_id,
        building_id=building_id,
        source_building_type=source_building_type,
        metadata_filter_column=type_column,
        profile_url=profile_url(profile_source, county_id=county_id, building_id=building_id),
        metadata_url=metadata_url(profile_source),
        selection_method="county_building_type_stable_hash",
    )


def load_oedi_8760_profile_kw(
    selection: OediProfileSelection,
    *,
    cache_dir: Path,
    target_peak_kw: float | None = None,
) -> list[float]:
    """Load a selected 15-minute EULP profile and convert it to hourly kW."""

    path = cache_dir / "profiles" / selection.profile_source / selection.county_id / (
        f"{selection.building_id}-0.parquet"
    )
    download_if_missing(selection.profile_url, path)
    frame = pd.read_parquet(path, columns=[EULP_TOTAL_ELECTRICITY_COLUMN])
    values = frame[EULP_TOTAL_ELECTRICITY_COLUMN].astype(float).to_list()
    if len(values) % 4 != 0:
        raise ValueError(f"Expected 15-minute EULP profile length divisible by 4; got {len(values)}")
    hourly_kw = [sum(values[idx : idx + 4]) for idx in range(0, len(values), 4)]
    if len(hourly_kw) != 8760:
        raise ValueError(f"Expected 8760 hourly values after aggregation; got {len(hourly_kw)}")
    if target_peak_kw is not None and max(hourly_kw) > 0:
        scale = target_peak_kw / max(hourly_kw)
        hourly_kw = [value * scale for value in hourly_kw]
    return [round(value, 6) for value in hourly_kw]


def _filter_county(metadata: pd.DataFrame, county_id: str) -> pd.DataFrame:
    if "in.county" in metadata.columns:
        filtered = metadata[metadata["in.county"] == county_id]
        if not filtered.empty:
            return filtered
    if "in.resstock_county_id" in metadata.columns:
        # SHIFT's example uses the human-readable county field for first pass.
        filtered = metadata[metadata["in.resstock_county_id"] == "MA, Plymouth County"]
        if not filtered.empty:
            return filtered
    return metadata


def _building_type_column(metadata: pd.DataFrame) -> str | None:
    for column in ("in.building_type", "in.geometry_building_type_recs"):
        if column in metadata.columns:
            return column
    return None


def _stable_building_id(index: pd.Index, token: str) -> str:
    labels = [str(value) for value in index]
    labels.sort()
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return labels[int(digest[:12], 16) % len(labels)]
