import json
import shutil
from pathlib import Path
import pandas as pd

def ensure_clean_dir(root, *, force):
    root = Path(root)
    if root.exists() and not force:
        raise FileExistsError(f"{root} exists. Use --force to replace it.")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    return root

def events_dir(root, scen):
    return Path(root) / ("events" if scen == "base" else f"events_{scen}")

def assert_event_catalog_audit(root):
    audit_path = Path(root) / "catalog" / "event_catalog_audit.json"
    if not audit_path.exists():
        return None
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("passed", False):
        return audit
    issue_count = int(audit.get("issue_count", len(audit.get("issues", []))))
    raise RuntimeError(f"Event Catalog audit failed for {root}: {issue_count} issues")

def build_event_timeseries(row, *, surge_event_members, forcing_variable="auto"):
    event_id = str(row.get("event_id")).strip()
    if forcing_variable == "auto" and "water_level_total" in surge_event_members:
        var = "water_level_total"
    else:
        var = "surge_absolute" if forcing_variable == "auto" and "surge_absolute" in surge_event_members else forcing_variable
    var = "surge" if var == "auto" else var

    # Select data, convert to a Series, and drop padded NaNs.
    series = surge_event_members[var].sel(event_id=event_id).to_series().dropna().astype(float)
    if series.empty:
        raise RuntimeError(f"{event_id} has no finite surge values.")
    series.index = pd.Index(series.index.astype(int), name="relative_hour")
    return {"h": series, "forcing_variable": var}

def select_zsini_from_series(series, *, mode="dry"):
    return float(series.iloc[0]) if mode == "boundary_t0" and len(series) else 0.0
