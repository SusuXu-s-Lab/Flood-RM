"""Build a Delft-FIAT model inside the isolated conda ``fiat`` env.

This script is executed by ``conda run -n fiat python _build_in_fiat_env.py <params.json>``
from :mod:`fiat_runs.build_model`. It must stay self-contained: import only
``hydromt`` / ``hydromt_fiat`` + stdlib, never the main-env ``fiat_runs`` package
(the two environments do not share dependencies).

It reads the declarative ``fiat_config.yml`` (the auditable build recipe) plus a small
runtime params JSON (paths that the recipe leaves to be injected: region geometry and
the SFINCS ground-elevation DEM), then builds and writes the FIAT model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def _usa_catalog_path() -> str:
    import hydromt_fiat

    path = Path(hydromt_fiat.__file__).parent / "data" / "hydromt_fiat_catalog_USA.yml"
    if not path.exists():
        raise FileNotFoundError(f"bundled hydromt_fiat USA data catalog not found at {path}")
    return str(path)


def main(params_path: str) -> int:
    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    region = params["region"]  # MUST already be EPSG:4326 (the NSI API requires lon/lat)
    model_root = params["model_root"]
    config_yml = params["config_yml"]
    data_libs = [_usa_catalog_path(), *params.get("extra_data_libs", [])]

    steps = yaml.safe_load(Path(config_yml).read_text(encoding="utf-8"))
    # The SFINCS DEM is not injected as exposure ground_elevation: it is baked into the
    # per-event hazard as inundation depth (downscale_floodmap), so FIAT reads a
    # DEM-referenced depth map and the model stays purely NSI/HAZUS here.

    from hydromt_fiat.fiat import FiatModel

    model = FiatModel(root=model_root, mode="w+", data_libs=data_libs)
    model.build(region={"geom": region}, opt=steps)
    model.write()

    # Provenance receipt: counts + the NSI endpoint actually used.
    try:
        n_assets = int(len(model.exposure.exposure_db)) if getattr(model, "exposure", None) else None
    except Exception:
        n_assets = None
    receipt = {
        "model_root": str(model_root),
        "region": str(region),
        "data_libs": data_libs,
        "n_exposure_assets": n_assets,
        "crs": str(getattr(model, "crs", None)),
        "build_steps": list(steps.keys()),
    }
    Path(model_root, "fiat_build_receipt.json").write_text(json.dumps(receipt, indent=2, default=str), encoding="utf-8")
    print(json.dumps(receipt, default=str))
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python _build_in_fiat_env.py <params.json>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
