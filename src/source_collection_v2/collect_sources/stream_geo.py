from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from design_events.stochastic_boundary.audit import Artifact, resolve, write_artifact

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
    if preferred_name:
        return next(f for f in files if str(f.get("name", "")).lower() == preferred_name.lower())
    return next((f for f in files if "stream" in str(f.get("name", "")).lower() and str(f.get("name", "")).lower().endswith((".parquet", ".csv", ".zip"))), files[0])


def download(info: dict, raw_dir: Path, *, timeout=120) -> Path:
    url = info.get("download_url") or info.get("downloadUrl") or info.get("url")
    name = Path(info.get("name") or Path(urlparse(url).path).name or "stream_geo_download").name
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / name
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with out.open("wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
    return out


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            member = next(n for n in z.namelist() if Path(n).suffix.lower() in {".parquet", ".csv", ".txt"})
            with z.open(member) as stream:
                if Path(member).suffix.lower() == ".parquet":
                    tmp = path.parent / Path(member).name
                    with tmp.open("wb") as f:
                        shutil.copyfileobj(stream, f)
                    return pd.read_parquet(tmp)
                return pd.read_csv(stream)
    raise ValueError(f"unsupported STREAM-geo file: {path}")


def _rows(path: Path) -> int:
    return len(pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path))
