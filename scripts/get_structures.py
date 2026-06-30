#!/usr/bin/env python3
"""Search a Study Location's region for coastal-structure GIS data"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sfincs_runs.structures import derive_massgis_sfincs_structure_layers  # noqa: E402
from study_location import define_location, study_area_bbox  # noqa: E402

# Rough degrees-per-km at mid latitudes; only used to pad the AOI envelope.
KM_PER_DEGREE = 111.0

@dataclass(frozen=True)
class StructureProvider:
    """One public GIS source of coastal structures, plus how to read and derive it."""

    name: str               # registry key / CLI --provider value
    source_label: str       # provenance string written into every feature + manifest
    source_url: str         # human landing page (mirrors artifacts/data_links.txt)
    file_stem: str          # names the derived source layers, e.g. weirs_<loc>_<stem>.geojson
    derive: str             # "massgis" -> reviewed derivation; "none" -> raw passthrough
    service_url: str | None = None  # ArcGIS REST feature-layer base for live queries
    where: str = "1=1"
    out_fields: str = "*"

# Provider registry. MassGIS is the reviewed Marshfield source; add coastal providers here
# rather than hard-coding endpoints in callers. A location may override service_url on the CLI.
PROVIDERS: dict[str, StructureProvider] = {
    "massgis_shoreline_structures": StructureProvider(
        name="massgis_shoreline_structures",
        source_label="MassGIS/CZM Shoreline Stabilization Structures, Public Structures 2015 Update",
        source_url="https://www.mass.gov/info-details/inventories-of-seawalls-and-other-coastal-structures",
        file_stem="massgis_public_2015",
        derive="massgis",
        # MassCZM "Shoreline Stabilization Structures" hosted feature service (ArcGIS Online item
        # 14938ac47b43427f87a96231fc1eaec5). Layer 3 is the "Public Structures, 2015 Update" set
        # the reviewed Marshfield derivation expects (carries PrimaryTyp / PositionZ / STR_ID).
        service_url="https://services1.arcgis.com/7iJyYTjCtKsZS1LR/arcgis/rest/services/Shoreline_Stabilization_Structures/FeatureServer/3",
    ),
}

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("location", help="Location folder under locations/, e.g. marshfield.")
    parser.add_argument(
        "--provider",
        default="massgis_shoreline_structures",
        choices=sorted(PROVIDERS),
        help="Registered GIS structure provider to query.",
    )
    parser.add_argument("--service-url", help="Override the provider ArcGIS REST feature-layer URL (live mode).")
    parser.add_argument("--source-file", type=Path, help="Ingest an already-downloaded GeoJSON instead of querying.")
    parser.add_argument("--buffer-km", type=float, default=1.0, help="Padding added around the Study Area bbox.")
    parser.add_argument("--no-derive", action="store_true", help="Write the raw pull only; skip SFINCS derivation.")
    parser.add_argument("--execute", action="store_true", help="Write artifacts; without it the run is a dry plan.")
    return parser.parse_args(argv)


def resolve_region(location: str) -> tuple[dict, Path, tuple[float, float, float, float]]:
    """Return (config, location_root, bbox) for a Study Location via the public interface."""
    config_path = REPO_ROOT / "locations" / location / "config.yaml"
    if not config_path.exists():
        raise SystemExit(f"no config.yaml for location {location!r} at {config_path}")
    definition = define_location(config_path)
    bbox = study_area_bbox(definition.config, REPO_ROOT)
    return definition.config, definition.root, bbox


def pad_bbox(bbox: tuple[float, float, float, float], buffer_km: float) -> tuple[float, float, float, float]:
    pad = float(buffer_km) / KM_PER_DEGREE
    west, south, east, north = bbox
    return (west - pad, south - pad, east + pad, north + pad)


def fetch_arcgis_geojson(service_url: str, bbox, *, where: str, out_fields: str, page_size: int = 1000) -> dict:
    """Query an ArcGIS REST feature layer for an envelope, paging past the transfer limit."""
    import requests  # local import: only needed for live queries

    west, south, east, north = bbox
    query_url = service_url.rstrip("/") + "/query"
    base = {
        "where": where,
        "outFields": out_fields,
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "outSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": "true",
        "f": "geojson",
    }
    features: list[dict] = []
    offset = 0
    while True:
        params = {**base, "resultOffset": offset, "resultRecordCount": page_size}
        response = requests.get(query_url, params=params, timeout=120)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("features") or []
        features.extend(page)
        if len(page) < page_size and not payload.get("exceededTransferLimit"):
            break
        if not page:
            break
        offset += len(page)
    return {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:4326"}}, "features": features}


def clip_to_bbox(geojson: dict, bbox) -> dict:
    """Keep features intersecting the AOI envelope (file pulls may be statewide)."""
    from shapely.geometry import box, shape

    envelope = box(*bbox)
    kept = []
    for feature in geojson.get("features") or []:
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        try:
            if shape(geometry).intersects(envelope):
                kept.append(feature)
        except Exception:  # malformed geometry: drop rather than fail the whole pull
            continue
    return {**geojson, "features": kept}


def primary_type_counts(geojson: dict) -> dict[str, int]:
    counter = Counter((feature.get("properties") or {}).get("PrimaryTyp", "unknown") for feature in geojson.get("features") or [])
    return dict(sorted(counter.items()))


def write_manifest(evidence_root: Path, payload: dict) -> Path:
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / "structure_acquisition_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    provider = PROVIDERS[args.provider]
    if args.service_url:
        provider = replace(provider, service_url=args.service_url)

    config, location_root, raw_bbox = resolve_region(args.location)
    bbox = pad_bbox(raw_bbox, args.buffer_km)
    structures_cfg = config.get("sfincs_structures") or {}
    evidence_root = location_root / structures_cfg.get("evidence_root", "data/static/structures/evidence")
    sources_root = location_root / structures_cfg.get("source_root", "data/static/structures/sources")

    mode = "file" if args.source_file else "live"
    if mode == "live" and not provider.service_url:
        raise SystemExit(
            f"provider {provider.name!r} has no service_url; pass --service-url <verified ArcGIS REST layer> "
            f"or --source-file <downloaded GeoJSON>. See {provider.source_url}"
        )

    print(f"location      : {args.location}")
    print(f"provider      : {provider.name} ({provider.source_label})")
    print(f"aoi bbox (pad): {tuple(round(v, 5) for v in bbox)}  [+{args.buffer_km} km]")
    print(f"mode          : {mode}")

    if mode == "file":
        if not args.source_file.exists():
            raise SystemExit(f"--source-file not found: {args.source_file}")
        raw = json.loads(args.source_file.read_text(encoding="utf-8"))
        raw = clip_to_bbox(raw, bbox)
        source_ref = str(args.source_file)
    else:
        print(f"service_url   : {provider.service_url}")
        raw = fetch_arcgis_geojson(provider.service_url, bbox, where=provider.where, out_fields=provider.out_fields)
        source_ref = provider.service_url

    type_counts = primary_type_counts(raw)
    n_raw = len(raw.get("features") or [])
    print(f"raw features  : {n_raw}  {type_counts}")

    raw_name = f"{args.location}_{provider.name}_raw.geojson"
    derived_summary: dict = {}
    if not args.execute:
        print("\n[dry run] would write:")
        print(f"  raw pull   -> {evidence_root / raw_name}")
        if not args.no_derive and provider.derive == "massgis":
            print(f"  weirs/thin -> {sources_root}/")
        print(f"  manifest   -> {evidence_root / 'structure_acquisition_manifest.json'}")
        print("re-run with --execute to write.")
        return

    evidence_root.mkdir(parents=True, exist_ok=True)
    raw_path = evidence_root / raw_name
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    if not args.no_derive and provider.derive == "massgis":
        derived = derive_massgis_sfincs_structure_layers(
            raw_path,
            sources_root,
            weirs_name=f"weirs_{args.location}_{provider.file_stem}.geojson",
            thin_dams_name=f"thin_dams_{args.location}_{provider.file_stem}.geojson",
        )
        summary = json.loads(Path(derived["summary"]).read_text(encoding="utf-8"))
        derived_summary = {
            "weirs": {"path": derived["weirs"].as_posix(), "feature_count": summary.get("weir_features", 0)},
            "thin_dams": {"path": derived["thin_dams"].as_posix(), "feature_count": summary.get("thin_dam_features", 0)},
            "omitted_features": summary.get("omitted_features", 0),
        }
        print(f"derived       : weirs={derived_summary['weirs']['feature_count']} "
              f"thin_dams={derived_summary['thin_dams']['feature_count']} "
              f"omitted={derived_summary['omitted_features']}")

    manifest = write_manifest(
        evidence_root,
        {
            "location": args.location,
            "provider": provider.name,
            "source_label": provider.source_label,
            "source_url": provider.source_url,
            "acquisition_mode": mode,
            "source_ref": source_ref,
            "where": provider.where,
            "aoi_bbox_padded": list(bbox),
            "aoi_buffer_km": args.buffer_km,
            "crs": "EPSG:4326",
            "raw_feature_count": n_raw,
            "primary_type_counts": type_counts,
            "raw_pull_path": raw_path.as_posix(),
            "derived": derived_summary,
            "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"wrote manifest: {manifest}")


if __name__ == "__main__":
    main()
