"""Public SMART-DS v1.0 OpenDSS model locations on the OEDI data lake.

SMART-DS publishes synthetic distribution feeders per region/subregion; we use
them only as audit references. ``SmartDsModelRef`` is the single source of truth
for the OEDI key layout — the URL list and the download plan both derive from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from power.region_config import get_region_config

oedi_s3_base_url = "https://oedi-data-lake.s3.amazonaws.com"
oedi_bucket = "oedi-data-lake"
valid_years = (2016, 2017, 2018)  # snapshots published in the public OEDI catalog


@dataclass(frozen=True)
class SmartDsModelRef:
    """OEDI location of one subregion's OpenDSS Master model."""

    region_id: str
    dataset_code: str
    year: int
    subregion: str
    scenario: str
    model_format: str

    @property
    def model_prefix(self) -> str:
        # OEDI key layout: SMART-DS/<version>/<year>/<region>/<subregion>/scenarios/<scenario>/<format>/
        return (
            f"SMART-DS/v1.0/{self.year}/{self.dataset_code}/{self.subregion}/"
            f"scenarios/{self.scenario}/{self.model_format}/"
        )

    @property
    def master_dss_key(self) -> str:
        return f"{self.model_prefix}Master.dss"

    @property
    def master_dss_url(self) -> str:
        return f"{oedi_s3_base_url}/{quote(self.master_dss_key)}"

    @property
    def source_s3_prefix(self) -> str:
        return f"s3://{oedi_bucket}/{self.model_prefix}"


def smart_ds_models(
    region_id,
    year=2016,
    scenario="base_timeseries",
    model_format="opendss",
) -> tuple[SmartDsModelRef, ...]:
    """One model reference per subregion of the configured region."""
    if year not in valid_years:
        raise ValueError(f"Year {year} is not in the public OEDI catalog.")
    cfg = get_region_config(region_id)
    return tuple(
        SmartDsModelRef(
            region_id=cfg.region_id,
            dataset_code=cfg.smart_ds_code,
            year=year,
            subregion=sub,
            scenario=scenario,
            model_format=model_format,
        )
        for sub in cfg.smart_ds_subregions
    )


def get_smart_ds_urls(
    region_id,
    year=2016,
    scenario="base_timeseries",
    model_format="opendss",
) -> list[str]:
    """Public OEDI download URLs for every subregion's Master.dss."""
    return [model.master_dss_url for model in smart_ds_models(region_id, year, scenario, model_format)]


@dataclass(frozen=True)
class SmartDsDownloadPlanItem:
    """One planned SMART-DS model download into the canonical artifact tree."""

    model: SmartDsModelRef
    output_dir: Path

    @property
    def source_s3_prefix(self) -> str:
        return self.model.source_s3_prefix

    @property
    def master_dss_path(self) -> Path:
        return self.output_dir / "Master.dss"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "download_manifest.json"


def smart_ds_model_dir(model: SmartDsModelRef, output_root: Path | None = None) -> Path:
    cfg = get_region_config(model.region_id)
    root = Path(output_root) if output_root else cfg.artifact_root / "smart_ds"
    return root / str(model.year) / model.subregion / model.scenario / model.model_format


def smart_ds_download_plan(
    region_id: str,
    *,
    year: int = 2016,
    scenario: str = "base_timeseries",
    model_format: str = "opendss",
    subregions: tuple[str, ...] | list[str] = (),
    all_subregions: bool = False,
    output_root: Path | None = None,
) -> tuple[SmartDsDownloadPlanItem, ...]:
    """Plan SMART-DS downloads for chosen subregions without fetching them."""
    models = smart_ds_models(region_id, year=year, scenario=scenario, model_format=model_format)
    if not all_subregions:
        wanted = set(subregions)
        if not wanted:
            raise ValueError("choose at least one subregion, or set all_subregions=True")
        unknown = sorted(wanted - {model.subregion for model in models})
        if unknown:
            available = ", ".join(sorted(model.subregion for model in models))
            raise ValueError(f"unknown subregion(s) {unknown}; expected one of: {available}")
        models = tuple(model for model in models if model.subregion in wanted)
    return tuple(
        SmartDsDownloadPlanItem(model=model, output_dir=smart_ds_model_dir(model, output_root=output_root))
        for model in models
    )
