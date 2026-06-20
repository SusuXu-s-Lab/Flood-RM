"""Run the Delft-FIAT engine per event via the isolated conda env.

The FIAT model built by :mod:`fiat_runs.build_model` provides the exposure
(``exposure/exposure.csv`` + ``exposure/buildings.gpkg``) and vulnerability
(``vulnerability/vulnerability_curves.csv``) files. For each event we write a fresh
``settings.toml`` in the delft_fiat 0.4.0 schema (the hydromt_fiat-written settings use
an incompatible ``[global] crs`` block), point ``hazard.file`` at the event water-level
raster, and invoke ``fiat run``. The hazard is a datum-referenced water-level map in feet,
so ``hazard.elevation_reference = "datum"``.

Single-event (``risk = false``) gives per-asset and total damage; the probability
weighting into Expected Annual Damage is done in :mod:`fiat_runs.risk` (main env).
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ._env import run_in_fiat_env


def _toml_path(p) -> str:
    # forward slashes are safe on all platforms in TOML and avoid escaping
    return str(Path(p).resolve()).replace("\\", "/")


def write_event_settings(
    settings_path,
    *,
    model_root,
    hazard_files,
    out_dir,
    risk=False,
    return_periods=None,
    srs="EPSG:4326",
) -> Path:
    """Write a delft_fiat settings.toml for one event (or an RP stack when risk=True)."""
    model_root = Path(model_root)
    exposure_csv = model_root / "exposure" / "exposure.csv"
    buildings = model_root / "exposure" / "buildings.gpkg"
    vulnerability = model_root / "vulnerability" / "vulnerability_curves.csv"

    # delft_fiat risk mode reads ONE multi-band raster (band per return period), not a list.
    hazard_files = [hazard_files] if isinstance(hazard_files, (str, Path)) else list(hazard_files)
    haz_block = f'file = "{_toml_path(hazard_files[0])}"\n'
    rp_block = ""
    if risk and return_periods:
        rp_block = "return_periods = [" + ", ".join(str(r) for r in return_periods) + "]\n"

    lines = f"""[model]
risk = {str(bool(risk)).lower()}

[model.srs]
value = "{srs}"

[output]
path = "{_toml_path(out_dir)}"

[output.csv]
name = "output.csv"

[output.geom]
name1 = "spatial.gpkg"

[hazard]
{haz_block}elevation_reference = "datum"
{rp_block}
[vulnerability]
file = "{_toml_path(vulnerability)}"

[exposure.csv]
file = "{_toml_path(exposure_csv)}"

[exposure.geom]
file1 = "{_toml_path(buildings)}"
crs = "{srs}"
"""
    settings_path = Path(settings_path)
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(lines, encoding="utf-8")
    return settings_path


def read_fiat_damages(out_dir) -> gpd.GeoDataFrame:
    """Read the per-asset FIAT output (geometry + damage columns) for one run."""
    gpkg = Path(out_dir) / "spatial.gpkg"
    if not gpkg.exists():
        raise RuntimeError(f"FIAT produced no spatial.gpkg in {out_dir}")
    return gpd.read_file(gpkg)


def run_fiat_event(model_root, hazard_tif, out_dir, *, event_id=None, srs="EPSG:4326") -> dict:
    """Run FIAT for a single event; return total + per-asset damage summary."""
    out_dir = Path(out_dir)
    settings_path = out_dir / "settings.toml"
    write_event_settings(
        settings_path, model_root=model_root, hazard_files=hazard_tif, out_dir=out_dir, risk=False, srs=srs
    )
    run_in_fiat_env(["fiat", "run", _toml_path(settings_path)], capture_output=True)

    gdf = read_fiat_damages(out_dir)
    total = pd.to_numeric(gdf.get("total_damage"), errors="coerce").fillna(0.0)
    return {
        "event_id": event_id,
        "total_damage": float(total.sum()),
        "n_assets": int(len(gdf)),
        "n_assets_damaged": int((total > 0).sum()),
        "output_gpkg": str(Path(out_dir) / "spatial.gpkg"),
    }
