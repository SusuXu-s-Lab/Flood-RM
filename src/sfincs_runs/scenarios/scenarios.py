import json
import shutil
from pathlib import Path
import pandas as pd
import xarray as xr

scenario_static_files = [
    "sfincs.bnd",
    "sfincs.dep",
    "sfincs.ind",
    "sfincs.manning",
    "sfincs.msk",
    "sfincs.nc",
    "sfincs.rug",
    "sfincs_subgrid.nc",
    "snapwave.bds",
    "snapwave.bhs",
    "snapwave.bnd",
    "snapwave.btp",
    "snapwave.bwd",
]

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

def read_design_inputs(root, *, scenario="base"):
    root = Path(root)
    assert_event_catalog_audit(root)
    ev_dir = events_dir(root, scenario)

    # 1. Read sampled design-event peaks.
    df = pd.read_csv(root / "catalog" / "sampled_peaks.csv")
    if "event_id" not in df:
        df.insert(0, "event_id", [f"evt_{i + 1:04d}" for i in range(len(df))])
    df["event_id"] = df["event_id"].astype(str)

    # 2. Add template metadata if present.
    summary_path = ev_dir / "surge_event_members_summary.csv"
    if summary_path.exists():
        df = df.merge(pd.read_csv(summary_path, parse_dates=["template_peak_time"]), on="event_id", how="left")

    # 3. Open hydrographs.
    ds = xr.open_dataset(ev_dir / "surge_event_members.nc").load()
    df["design_scenario"] = ds.attrs.get("scenario_name", scenario)
    df["design_slr_offset_m"] = ds.attrs.get("slr_offset_m", 0.0)
    return df, ds

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

def ensure_static_files(event_dir, template_dir):
    event_dir, template_dir = Path(event_dir), Path(template_dir)
    event_dir.mkdir(parents=True, exist_ok=True)

    # Static base-model files are shared by every event folder.
    for name in scenario_static_files:
        src, dst = template_dir / name, event_dir / name
        if not src.exists():
            continue
        if dst.exists():
            continue
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
