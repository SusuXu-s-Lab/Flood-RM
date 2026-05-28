"""Build public SMART-DS v1.0 OEDI links for configured regions."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from power.region_config import get_region_config

oedi_s3_base_url = "https://oedi-data-lake.s3.amazonaws.com"
oedi_bucket = "oedi-data-lake"
valid_years = (2016, 2017, 2018)

def get_smart_ds_urls(
    region_id,
    year=2016,
    scenario="base_timeseries",
    model_format="opendss",
):
    """
    Builds OEDI S3 links for all subregions in a study area.
    Returns a simple list of strings.
    """
    base_url = oedi_s3_base_url
    if year not in valid_years:
        raise ValueError(f"Year {year} is not in the public OEDI catalog.")

    cfg = get_region_config(region_id)
    urls = []

    for sub in cfg.smart_ds_subregions:
        path = (
            f"SMART-DS/v1.0/{year}/{cfg.smart_ds_code}/{sub}/"
            f"scenarios/{scenario}/{model_format}/Master.dss"
        )
        urls.append(f"{base_url}/{quote(path)}")

    return urls

@dataclass(frozen=True)
class SmartDsModelRef:
    """Small compatibility wrapper for callers that need OEDI keys and metadata."""

    region_id: str
    dataset_code: str
    year: int
    subregion: str
    scenario: str
    model_format: str

    @property
    def model_prefix(self):
        return (
            f"SMART-DS/v1.0/{self.year}/{self.dataset_code}/{self.subregion}/"
            f"scenarios/{self.scenario}/{self.model_format}/"
        )

    @property
    def master_dss_key(self):
        return f"{self.model_prefix}Master.dss"

    @property
    def master_dss_url(self):
        return f"{oedi_s3_base_url}/{quote(self.master_dss_key)}"

    @property
    def source_s3_prefix(self):
        return f"s3://{oedi_bucket}/{self.model_prefix}"


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


def smart_ds_models(
    region_id,
    year=2016,
    scenario="base_timeseries",
    model_format="opendss",
):
    """Return SMART-DS model references for callers that need subregion metadata."""
    get_smart_ds_urls(region_id, year, scenario, model_format)

    cfg = get_region_config(region_id)
    models = []

    for sub in cfg.smart_ds_subregions:
        models.append(
            SmartDsModelRef(
                region_id=cfg.region_id,
                dataset_code=cfg.smart_ds_code,
                year=year,
                subregion=sub,
                scenario=scenario,
                model_format=model_format,
            )
        )
    return tuple(models)


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
    """Return validated SMART-DS model downloads without executing them."""
    models = smart_ds_models(
        region_id,
        year=year,
        scenario=scenario,
        model_format=model_format,
    )
    if all_subregions:
        selected = models
    else:
        if not subregions:
            raise ValueError("choose at least one subregion, or set all_subregions=True")
        wanted = set(subregions)
        available = {model.subregion for model in models}
        missing = sorted(wanted - available)
        if missing:
            valid = ", ".join(sorted(available))
            raise ValueError(f"unknown subregion(s) {missing}; expected one of: {valid}")
        selected = tuple(model for model in models if model.subregion in wanted)

    return tuple(
        SmartDsDownloadPlanItem(
            model=model,
            output_dir=smart_ds_model_dir(model, output_root=output_root),
        )
        for model in selected
    )
