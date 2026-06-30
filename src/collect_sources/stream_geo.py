from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from collect_sources.audit import Artifact, resolve, write_artifact

FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
ARTICLE_ID = 24463240


def collect(settings: dict, *, skip_existing=False) -> Artifact:
    paths, spec = settings["paths"], settings["spec"]
    table = resolve(paths, spec.get("stream_geo_table", "data/sources/stream_geo/stream_geo.parquet"))
    raw_dir = resolve(paths, spec.get("raw_dir", "data/sources/stream_geo/raw"))
    if skip_existing and table.exists():
        return Artifact("stream_geo", "river_geometry_lookup", None, None, {"stream_geo_table": table, "raw_dir": raw_dir}, {"reused": True, "rows": _rows(table)})
    info = select_file(requests.get(FIGSHARE_API.format(article_id=int(spec.get("figshare_article_id", ARTICLE_ID))), timeout=int(spec.get("request_timeout_seconds", 120))).json(), spec.get("figshare_file_name"))
    raw = download(info, raw_dir, timeout=int(spec.get("request_timeout_seconds", 120)))
    frame = read_table(raw)
    table.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(table, index=False) if table.suffix.lower() == ".parquet" else frame.to_csv(table, index=False)
    artifact = Artifact("stream_geo", "river_geometry_lookup", None, None, {"stream_geo_table": table, "raw_file": raw}, {"rows": len(frame), "figshare_article_id": int(spec.get("figshare_article_id", ARTICLE_ID))})
    write_artifact(paths, artifact)
    return artifact


def select_file(metadata: dict, preferred_name=None) -> dict:
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


def download(info: dict, raw_dir: Path, *, timeout=120, session_get=None) -> Path:
    url = info.get("download_url") or info.get("downloadUrl") or info.get("url")
    if not url:
        raise ValueError(f"Figshare file metadata is missing a download URL: {info}")
    name = Path(info.get("name") or Path(urlparse(str(url)).path).name or "stream_geo_download").name
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / name
    tmp = out.with_suffix(out.suffix + ".tmp")
    get = session_get or requests.get
    with get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(out)
    return out


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            members = [name for name in z.namelist() if not name.endswith("/")]
            for member in members:
                member_suffix = Path(member).suffix.lower()
                with z.open(member) as stream:
                    if member_suffix == ".parquet":
                        tmp = path.parent / Path(member).name
                        with tmp.open("wb") as f:
                            shutil.copyfileobj(stream, f)
                        return pd.read_parquet(tmp)
                    if member_suffix in {".csv", ".txt"}:
                        return pd.read_csv(stream)
    raise ValueError(f"Unsupported STREAM-geo download format: {path}")


def _rows(path: Path) -> int:
    return len(pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path))
