from __future__ import annotations
from pathlib import Path
import pandas as pd
from .run import run_event

def historical_events(catalog_csv) -> pd.DataFrame:
    cat = pd.read_csv(catalog_csv)
    cols = ["event_id", "sample_rp_years", "historical_event_time", "coastal_absolute_peak_m"]
    return cat[cat["event_origin"] == "historical_tail"][[c for c in cols if c in cat.columns]].reset_index(drop=True)

def validate_history(model_root, rasterizer, storage_root, catalog_csv, out_root, hazard_root) -> dict:
    """Run FIAT for each historical event that has a completed SFINCS run."""
    storage_root, out_root, hazard_root = Path(storage_root), Path(out_root), Path(hazard_root)
    hist = historical_events(catalog_csv)
    rows, missing = [], []
    for ev in hist.itertuples(index=False):
        eid = ev.event_id
        map_path = storage_root / eid / "sfincs_map.nc"
        if not map_path.exists():
            missing.append(eid)
            continue
        haz = hazard_root / f"{eid}.tif"
        rasterizer.export(map_path, haz)
        res = run_event(model_root, haz, out_root / eid, event_id=eid)
        res.update({
            "historical_event_time": getattr(ev, "historical_event_time", None),
            "coastal_absolute_peak_m": getattr(ev, "coastal_absolute_peak_m", None),
            "sample_rp_years": getattr(ev, "sample_rp_years", None),
        })
        rows.append(res)
    return {
        "n_historical": int(len(hist)),
        "n_run": len(rows),
        "missing_sfincs_runs": missing,
        "damages": pd.DataFrame(rows),
    }