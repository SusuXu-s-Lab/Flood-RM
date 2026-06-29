from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from collect_sources.national_hydrography import (
    STREAM_GEO_FIGSHARE_ARTICLE_ID,
    fetch_nldi_comid,
)
from collect_sources.source_artifacts import write_source_artifact

FIGSHARE_ARTICLE_API = "https://api.figshare.com/v2/articles/{article_id}"


def collect_stream_geo_nldi(settings, *, skip_existing=True, smoke=False):
    """Cache STREAM-geo river geometry estimates for Wflow river enrichment.

    NLDI is kept as a companion lookup API for COMID review/audit. Bulk NLDI calls
    are intentionally not made during normal source collection because NHDPlus river
    geometry should already carry a COMID-like identifier.
    """
    config = settings["config"]
    paths = settings["paths"]
    location_root = Path(paths["location_root"])
    collection = config.get("collection", {})
    national = collection.get("national_hydrography", {})
    spec = collection.get("stream_geo_nldi", {})

    table_path = _location_path(
        location_root,
        spec.get("stream_geo_table", national.get("stream_geo_table", "data/sources/national_hydrography/stream_geo.parquet")),
    )
    raw_dir = _location_path(location_root, spec.get("raw_dir", "data/sources/national_hydrography/stream_geo_raw"))
    manifest = Path(paths.get("source_artifacts_root", location_root / "data/sources/source_artifacts")) / "stream_geo_nldi_sources.json"

    if skip_existing and table_path.exists():
        rows = _table_rows(table_path)
        _write_manifest(paths, table_path, raw_dir, manifest, "reused", rows, spec, raw_file=None, smoke=smoke)
        return {
            "status": "reused",
            "reused": True,
            "stream_geo_table": table_path,
            "source_artifact_json": manifest,
            "rows": rows,
            "nldi_status": "available_for_point_lookup",
        }

    if smoke:
        _write_manifest(paths, table_path, raw_dir, manifest, "smoke_skipped", 0, spec, raw_file=None, smoke=True)
        return {
            "status": "smoke_skipped",
            "reused": False,
            "stream_geo_table": table_path,
            "source_artifact_json": manifest,
            "rows": 0,
            "nldi_status": "available_for_point_lookup",
        }

    article_id = int(spec.get("figshare_article_id", STREAM_GEO_FIGSHARE_ARTICLE_ID))
    metadata = fetch_figshare_article_metadata(article_id, timeout_seconds=float(spec.get("request_timeout_seconds", 120)))
    file_info = select_stream_geo_file(metadata, preferred_name=spec.get("figshare_file_name"))
    raw_file = download_figshare_file(file_info, raw_dir, timeout_seconds=float(spec.get("request_timeout_seconds", 120)))
    frame = read_stream_geo_download(raw_file)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(table_path, index=False)
    rows = int(len(frame))
    _write_manifest(paths, table_path, raw_dir, manifest, "collected", rows, spec, raw_file=raw_file, smoke=False)
    return {
        "status": "collected",
        "reused": False,
        "stream_geo_table": table_path,
        "source_artifact_json": manifest,
        "rows": rows,
        "raw_file": raw_file,
        "nldi_status": "available_for_point_lookup",
    }


def fetch_figshare_article_metadata(article_id: int, *, timeout_seconds=120, session_get=None) -> dict:
    get = session_get or requests.get
    response = get(FIGSHARE_ARTICLE_API.format(article_id=int(article_id)), timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def select_stream_geo_file(metadata: dict, *, preferred_name: str | None = None) -> dict:
    files = list(metadata.get("files") or [])
    if not files:
        raise ValueError("Figshare article metadata did not include downloadable files")
    if preferred_name:
        for info in files:
            if str(info.get("name", "")).lower() == preferred_name.lower():
                return info
        raise ValueError(f"STREAM-geo Figshare file not found: {preferred_name}")
    preferred_suffixes = (".parquet", ".csv", ".zip")
    candidates = [
        info
        for info in files
        if str(info.get("name", "")).lower().endswith(preferred_suffixes)
        and "stream" in str(info.get("name", "")).lower()
    ]
    return candidates[0] if candidates else files[0]


def download_figshare_file(file_info: dict, raw_dir: Path, *, timeout_seconds=120, session_get=None) -> Path:
    url = file_info.get("download_url") or file_info.get("downloadUrl") or file_info.get("url")
    if not url:
        raise ValueError(f"Figshare file metadata is missing a download URL: {file_info}")
    name = file_info.get("name") or Path(urlparse(str(url)).path).name or "stream_geo_download"
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / Path(name).name
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    get = session_get or requests.get
    with get(url, stream=True, timeout=timeout_seconds) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    stream.write(chunk)
    tmp_path.replace(output_path)
    return output_path


def read_stream_geo_download(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            for member in members:
                member_suffix = Path(member).suffix.lower()
                with archive.open(member) as stream:
                    if member_suffix == ".parquet":
                        extracted = path.parent / Path(member).name
                        with extracted.open("wb") as output:
                            shutil.copyfileobj(stream, output)
                        return pd.read_parquet(extracted)
                    if member_suffix in {".csv", ".txt"}:
                        return pd.read_csv(stream)
    raise ValueError(f"Unsupported STREAM-geo download format: {path}")


def _write_manifest(paths, table_path, raw_dir, manifest, status, rows, spec, *, raw_file, smoke):
    write_source_artifact(
        paths,
        source="stream_geo_nldi",
        kind="river_geometry_lookup",
        start=pd.Timestamp("1970-01-01"),
        end=pd.Timestamp("2100-12-31"),
        artifacts={
            "stream_geo_table": table_path,
            "stream_geo_raw_dir": raw_dir,
            "stream_geo_raw_file": raw_file or "",
            "manifest": manifest,
        },
        metadata={
            "status": status,
            "rows": int(rows),
            "stream_geo_article_id": int(spec.get("figshare_article_id", STREAM_GEO_FIGSHARE_ARTICLE_ID)),
            "nldi_base_url": "https://api.water.usgs.gov/nldi/linked-data",
            "nldi_role": "COMID point lookup companion; no bulk calls during normal collection",
            "smoke": bool(smoke),
        },
    )
    payload = {
        "source": "stream_geo_nldi",
        "kind": "river_geometry_lookup",
        "status": status,
        "metadata": {
            "rows": int(rows),
            "stream_geo_article_id": int(spec.get("figshare_article_id", STREAM_GEO_FIGSHARE_ARTICLE_ID)),
            "nldi_base_url": "https://api.water.usgs.gov/nldi/linked-data",
            "nldi_role": "COMID point lookup companion; no bulk calls during normal collection",
            "smoke": bool(smoke),
        },
        "artifacts": {
            "stream_geo_table": str(table_path),
            "stream_geo_raw_dir": str(raw_dir),
            "stream_geo_raw_file": str(raw_file) if raw_file else "",
        },
    }
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _table_rows(path: Path) -> int:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    return int(sum(1 for _ in path.open(encoding="utf-8", errors="ignore")) - 1)


def _location_path(location_root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(location_root) / path
