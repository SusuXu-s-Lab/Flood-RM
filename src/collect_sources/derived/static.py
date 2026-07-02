from __future__ import annotations

import numpy as np
import pandas as pd


def stream_geometry_parameters(rivers: pd.DataFrame, stream_geo: pd.DataFrame | None = None) -> pd.DataFrame:
    """Attach width/depth/discharge using source columns first, simple formula second."""
    out = rivers.copy()
    if stream_geo is not None and not stream_geo.empty:
        rid = _first(out, "comid", "COMID", "nhdplusid", "NHDPlusID", "featureid")
        sid = _first(stream_geo, "comid", "COMID", "nhdplusid", "NHDPlusID", "featureid")
        if rid and sid:
            keep = [sid, *[c for c in [_first(stream_geo, "rivwth", "width_m", "XGB_Width_m"), _first(stream_geo, "rivdph", "depth_m", "XGB_Depth_m"), _first(stream_geo, "qbankfull", "Qbf")] if c]]
            out = out.merge(stream_geo[keep].drop_duplicates(sid), left_on=rid, right_on=sid, how="left")
    area = _numeric(out, _first(out, "TotDASqKm", "areasqkm", "uparea", "uparea_km2"), default=100).clip(lower=1)
    out["rivwth"] = _numeric(out, _first(out, "rivwth", "width_m", "XGB_Width_m")).fillna((5 + 2.5 * np.sqrt(area)).clip(lower=10))
    out["rivdph"] = _numeric(out, _first(out, "rivdph", "depth_m", "XGB_Depth_m"))
    out["qbankfull"] = _numeric(out, _first(out, "qbankfull", "Qbf")).fillna((2 * area ** 0.8).clip(lower=1))
    out["review_status"] = "review_required_stream_geometry_parameters"
    return out


def reservoir_parameters(waterbodies: pd.DataFrame, *, min_area_km2=1.0, default_depth_m=5.0, default_discharge_m3s=0.01) -> pd.DataFrame:
    out = waterbodies.copy()
    area_col = _first(out, "areasqkm", "area_km2")
    area = _numeric(out, area_col)
    out = out.loc[area >= float(min_area_km2)].copy()
    out["waterbody_id"] = np.arange(1, len(out) + 1)
    out["Area_avg"] = _numeric(out, area_col) * 1_000_000.0
    out["Depth_avg"] = float(default_depth_m)
    out["Vol_avg"] = out["Area_avg"] * out["Depth_avg"]
    out["Dis_avg"] = float(default_discharge_m3s)
    out["reservoir_operation"] = "no_control"
    out["review_status"] = "review_required_public_waterbody_estimates"
    return out


def ssurgo_layer_table(attrs: pd.DataFrame) -> pd.DataFrame:
    """SSURGO horizons -> SoilGrids-shaped layer table; rasterization is kept outside."""
    depths = [0, 5, 15, 30, 60, 100, 200]
    cols = ["hzdept_r", "hzdepb_r", "sandtotal_r", "silttotal_r", "claytotal_r", "dbthirdbar_r", "om_r", "ph1to1h2o_r"]
    frame = attrs.copy()
    frame[cols] = frame[cols].apply(pd.to_numeric, errors="coerce")
    rows = []
    for mukey, g in frame.dropna(subset=["mukey", "hzdept_r", "hzdepb_r"]).groupby("mukey"):
        g = g[g.hzdepb_r > g.hzdept_r].sort_values("hzdept_r")
        if g.empty:
            continue
        row = {"mukey": str(mukey), "soilthickness": min(200.0, float(g.hzdepb_r.max()))}
        for i, depth in enumerate(depths, 1):
            h = g[(g.hzdept_r <= depth) & (g.hzdepb_r > depth)]
            h = h.iloc[0] if len(h) else g.iloc[(g.hzdept_r - depth).abs().argmin()]
            row.update({f"bd_sl{i}": h.dbthirdbar_r, f"oc_sl{i}": h.om_r / 1.724, f"ph_sl{i}": h.ph1to1h2o_r, f"clyppt_sl{i}": h.claytotal_r, f"sltppt_sl{i}": h.silttotal_r, f"sndppt_sl{i}": h.sandtotal_r})
        rows.append(row)
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def _numeric(frame, column, *, default=np.nan):
    if column is None or column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _first(frame, *names):
    lower = {str(c).lower(): c for c in getattr(frame, "columns", [])}
    return next((lower[n.lower()] for n in names if n.lower() in lower), None)
