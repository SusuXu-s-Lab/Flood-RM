"""Fetch ERA5 ocean wave reanalysis from Copernicus CDS.

Produces the local NetCDF referenced by the `era5_waves` entry in
`locations/marshfield/data/static/data_catalogue.yaml`. Used as SnapWave Boundary Forcing
in the Wave-Coupled build path.

Requires the `cdsapi` package and a valid ~/.cdsapirc; both are user-local
(network + credentials) so we import cdsapi lazily and keep the request
builder as a pure function for testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tqdm.auto import tqdm as progress_bar

from collect_sources.era5 import CDS_WAVE_VARIABLES, cds_payload as _v2_cds_payload


# canonical ERA5 single-levels variable long names for ocean waves
era5_wave_variables = CDS_WAVE_VARIABLES


def build_cds_request_payload(bbox_wgs84, time_window, variables=None) -> dict:
    return _v2_cds_payload(bbox_wgs84, time_window[0], time_window[1], variables)


def fetch_era5_waves(bbox_wgs84, time_window, output_path, variables=None, force=False) -> Path:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"skip: {output_path} already exists (use --force to overwrite)")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi not installed. Run: uv add cdsapi (or pip install cdsapi)."
        ) from exc
    with progress_bar(total=3, desc="CDS ERA5 waves", unit="stage", dynamic_ncols=True) as progress:
        progress.set_postfix_str("build request", refresh=False)
        payload = build_cds_request_payload(bbox_wgs84, time_window, variables)
        progress.update()
        progress.set_postfix_str("submit request", refresh=False)
        client = cdsapi.Client()
        progress.update()
        progress.set_postfix_str(f"write {output_path.name}", refresh=False)
        client.retrieve("reanalysis-era5-single-levels", payload, str(output_path))
        progress.update()
    return output_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fetch ERA5 ocean wave reanalysis from CDS.")
    p.add_argument("--bbox", required=True, nargs=4, type=float,
                   metavar=("W", "S", "E", "N"), help="Bounding box in WGS84 (W S E N)")
    p.add_argument("--start", required=True, type=pd.Timestamp, help="event window start (ISO)")
    p.add_argument("--stop", required=True, type=pd.Timestamp, help="event window stop (ISO)")
    p.add_argument("--out", required=True, type=Path, help="output NetCDF path")
    p.add_argument("--force", action="store_true", help="overwrite if output exists")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fetch_era5_waves(
        bbox_wgs84=tuple(args.bbox),
        time_window=(args.start, args.stop),
        output_path=args.out,
        force=args.force,
    )


if __name__ == "__main__":
    main()
