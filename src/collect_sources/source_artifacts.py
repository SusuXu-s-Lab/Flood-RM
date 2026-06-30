"""Source Artifact manifest helpers for source-collection adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def source_artifact_path(paths, source, kind):
    return _source_artifacts_root(paths) / f"{source}_{kind}.json"


def read_source_artifact(paths, source, kind):
    path = source_artifact_path(paths, source, kind)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_artifact_covers(paths, source, kind, start, end):
    manifest = read_source_artifact(paths, source, kind)
    if manifest is None or manifest.get("status") != "complete":
        return False
    if manifest.get("metadata", {}).get("smoke") is True:
        return False
    artifact_start = manifest.get("start")
    artifact_end = manifest.get("end")
    if not artifact_start or not artifact_end:
        return False
    return pd.Timestamp(artifact_start) <= pd.Timestamp(start) and pd.Timestamp(artifact_end) >= pd.Timestamp(end)


def write_source_artifact(paths, source, kind, start=None, end=None, artifacts=None, metadata=None, status="complete"):
    manifest = {
        "study_location": paths["location_name"],
        "source": source,
        "kind": kind,
        "status": status,
        "start": None if start is None else pd.Timestamp(start).isoformat(),
        "end": None if end is None else pd.Timestamp(end).isoformat(),
        "artifacts": {
            key: _relative_path(value, paths["repo_root"])
            for key, value in (artifacts or {}).items()
        },
        "metadata": metadata or {},
    }
    path = source_artifact_path(paths, source, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _source_artifacts_root(paths):
    if paths.get("source_artifacts_root") is not None:
        return Path(paths["source_artifacts_root"])
    root = Path(paths.get("location_root") or paths.get("repo_root") or Path.cwd())
    return root / "data/sources/source_artifacts"
