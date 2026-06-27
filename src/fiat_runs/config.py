"""Runtime paths for the Delft-FIAT risk stage.

Reuses the SFINCS runtime (:func:`sfincs_runs.config.load_runtime`) so the FIAT
stage sees the same location, data root, and SFINCS artifacts, then layers on the
FIAT model / hazard / risk roots. Path resolution follows the same
derive-from-location-data-root convention as ``sfincs_runs.config.build_paths`` so a
relocated location stays internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sfincs_runs.config import build_paths, load_config, load_runtime

__all__ = [
    "FiatNotebookRuntime",
    "build_paths",
    "fiat_paths",
    "load_config",
    "load_notebook_runtime",
    "load_runtime",
]


@dataclass(frozen=True)
class FiatNotebookRuntime:
    location_root: Path
    location_name: str
    repo_root: Path
    config: dict
    paths: dict
    catalog_csv: Path
    metadata_json: Path
    model_root: Path
    per_event_damage_csv: Path
    tide_gauge_root: Path
    tide_gauge_fig_root: Path


def fiat_paths(paths: dict) -> dict:
    """Extend SFINCS runtime ``paths`` with Delft-FIAT artifact roots.

    Added keys:
      - ``fiat_root``           : ``data/fiat`` (stage root)
      - ``fiat_model_root``     : built FIAT model (settings.toml, exposure, vulnerability)
      - ``fiat_exposure_root``  : cached NSI exposure pull (reproducibility/provenance)
      - ``fiat_hazard_root``    : per-event water-level hazard GeoTIFFs, by SLR scenario
      - ``fiat_risk_root``      : EAD tables, audit receipts, risk maps
    """
    data_root = Path(paths["location_data_root"])
    fiat_root = data_root / "fiat"
    extended = dict(paths)
    extended.update(
        {
            "fiat_root": fiat_root,
            "fiat_model_root": fiat_root / "model",
            "fiat_exposure_root": data_root / "static" / "exposure",
            "fiat_hazard_root": fiat_root / "hazard",
            "fiat_risk_root": fiat_root / "risk",
        }
    )
    return extended


def load_notebook_runtime(location_root, *, create_tide_gauge_dirs: bool = False) -> FiatNotebookRuntime:
    """Load derived FIAT paths for the coastal risk notebook."""
    location_root = Path(location_root).resolve()
    config, paths = load_runtime(location_root / "config.yaml")
    paths = fiat_paths(paths)
    tide_gauge_root = paths["location_data_root"] / "sfincs" / "tide_gauges"
    tide_gauge_fig_root = tide_gauge_root / "figures"
    if create_tide_gauge_dirs:
        tide_gauge_root.mkdir(parents=True, exist_ok=True)
        tide_gauge_fig_root.mkdir(parents=True, exist_ok=True)
    return FiatNotebookRuntime(
        location_root=location_root,
        location_name=location_root.name,
        repo_root=location_root.parents[1],
        config=config,
        paths=paths,
        catalog_csv=paths["design_outputs_root"] / "catalog" / "event_catalog.csv",
        metadata_json=paths["design_outputs_root"] / "catalog" / "catalog_risk_metadata.json",
        model_root=paths["fiat_model_root"],
        per_event_damage_csv=paths["fiat_risk_root"] / "per_event_damage.csv",
        tide_gauge_root=tide_gauge_root,
        tide_gauge_fig_root=tide_gauge_fig_root,
    )
