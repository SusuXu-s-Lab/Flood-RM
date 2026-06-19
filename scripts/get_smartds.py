#!/usr/bin/env python3
"""Download SMART-DS data from the public OEDI bucket."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import fsspec


BUCKET = "oedi-data-lake"
VERSION = "v1.0"
DEFAULT_FORMATS = ("opendss_no_loadshapes", "geojson")
ALIASES = {"sfo": "SFO", "austin": "AUS", "greensboro": "GSO"}
SLUGS = {"SFO": "sfo", "AUS": "austin", "GSO": "greensboro"}


def code(name: str) -> str:
    return ALIASES.get(name.lower(), name.upper())


def prefix(year: int, dataset: str) -> str:
    return f"{BUCKET}/SMART-DS/{VERSION}/{year}/{dataset}"


def count_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def subregions(fs, year: int, dataset: str, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(Path(path).name for path in fs.ls(prefix(year, dataset), detail=False))


def download_prefix(fs, remote: str, local: Path, execute: bool) -> int:
    files = [path for path in fs.find(remote) if not path.endswith("/")]
    verb = "download" if execute else "dry-run"
    print(f"{verb} s3://{remote} -> {local} ({len(files)} files)")
    if not execute:
        return len(files)
    for src in files:
        dst = local / Path(src).relative_to(remote)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fs.get(src, str(dst))
    return len(files)


def write_manifest(root: Path, dataset: str, year: int, scenario: str, formats: list[str], subs: list[str]) -> Path:
    year_root = root / str(year)
    year_root.mkdir(parents=True, exist_ok=True)
    path = year_root / "download_manifest.json"
    path.write_text(json.dumps({
        "dataset_code": dataset,
        "year": year,
        "scenario": scenario,
        "formats": formats,
        "subregions": subs,
        "source_s3_prefix": f"s3://{prefix(year, dataset)}/",
        "local_dir": str(year_root),
        "local_file_count": count_files(year_root),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("region", nargs="?", default="sfo")
    parser.add_argument("--year", type=int, default=2016)
    parser.add_argument("--scenario", default="base_timeseries")
    parser.add_argument("--format", action="append", dest="formats")
    parser.add_argument("--subregion", action="append", dest="subregions")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    dataset = code(args.region)
    formats = args.formats or list(DEFAULT_FORMATS)
    root = args.output_root or Path("locations") / SLUGS.get(dataset, args.region.lower()) / "data" / "smart_ds"
    fs = fsspec.filesystem("s3", anon=True)
    subs = subregions(fs, args.year, dataset, args.subregions)
    for sub in subs:
        for fmt in formats:
            remote = f"{prefix(args.year, dataset)}/{sub}/scenarios/{args.scenario}/{fmt}"
            local = root / str(args.year) / sub / args.scenario / fmt
            download_prefix(fs, remote, local, args.execute)
    if args.execute:
        print(f"wrote {write_manifest(root, dataset, args.year, args.scenario, formats, subs)}")


if __name__ == "__main__":
    main()
