"""Runtime paths for the Delft-FIAT risk stage.

Reuses the SFINCS runtime (:func:`sfincs_runs.config.load_runtime`) so the FIAT
stage sees the same location, data root, and SFINCS artifacts, then layers on the
FIAT model / hazard / risk roots. Path resolution follows the same
derive-from-location-data-root convention as ``sfincs_runs.config.build_paths`` so a
relocated location stays internally consistent.
"""

from __future__ import annotations

from pathlib import Path

from sfincs_runs.config import build_paths, load_config, load_runtime

__all__ = ["fiat_paths", "load_runtime", "load_config", "build_paths"]


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
