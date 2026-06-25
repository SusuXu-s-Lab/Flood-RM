"""Damage-informed coastal water-level sensor placement helpers.

These utilities rank deployable water-level sensor candidate points against
catalog-wide SFINCS flood outcomes and FIAT event damages. Existing SFINCS
runup-gauge transects are treated as computational output features and candidate
seeds, not as already-installed physical instruments.
"""

from __future__ import annotations

import math
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import LineString, Point

from sfincs_runs import diagnostics

MODEL_CRS = "EPSG:26919"
WEB_MERCATOR = "EPSG:3857"


def load_runup_transects(path, *, crs: str = MODEL_CRS) -> gpd.GeoDataFrame:
    """Load configured SFINCS runup-gauge transects from YAML or ``sfincs.rug``."""
    path = Path(path)
    if not path.exists():
        return gpd.GeoDataFrame(columns=["gauge", "candidate_source", "geometry"], geometry="geometry", crs=crs)
    if path.suffix.lower() in {".yml", ".yaml"}:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        runup_cfg = (
            cfg.get("coastal_wave_coupling", {}).get("runup_gauges")
            or cfg.get("setup_runup_gauges")
            or cfg.get("runup_gauges")
            or {}
        )
        rows = []
        for rec in runup_cfg.get("transects", []) or []:
            rows.append(
                {
                    "gauge": str(rec.get("name", f"runup_{len(rows) + 1:02d}")),
                    "candidate_source": "computational_runup_transect",
                    "geometry": LineString([(float(rec["x0"]), float(rec["y0"])), (float(rec["x1"]), float(rec["y1"]))]),
                }
            )
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=runup_cfg.get("crs", crs) or crs).to_crs(crs)
    return _load_runup_transects_from_rug(path, crs=crs)


def candidate_points_from_runup_transects(transects: gpd.GeoDataFrame, *, crs: str = MODEL_CRS) -> gpd.GeoDataFrame:
    """Use runup-transect midpoints as candidate water-level sensor seeds."""
    if transects is None or transects.empty:
        return _empty_candidates(crs)
    gdf = transects.to_crs(crs).copy()
    rows = []
    for i, row in enumerate(gdf.itertuples(index=False), start=1):
        geom = row.geometry
        point = geom.interpolate(0.5, normalized=True) if geom.geom_type != "Point" else geom
        gauge = str(getattr(row, "gauge", f"runup_{i:02d}"))
        rows.append(
            {
                "candidate_id": f"runup_{i:03d}",
                "candidate_source": "runup_transect_midpoint",
                "candidate_label": gauge,
                "x": float(point.x),
                "y": float(point.y),
                "nearby_annual_damage": 0.0,
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def candidate_points_from_building_risk(
    building_risk: gpd.GeoDataFrame,
    *,
    top_n: int = 30,
    metric: str = "annual_damage",
    min_distance_m: float = 120.0,
    crs: str = MODEL_CRS,
) -> gpd.GeoDataFrame:
    """Create candidate points from the highest-risk FIAT building clusters."""
    if building_risk is None or building_risk.empty or metric not in building_risk:
        return _empty_candidates(crs)
    gdf = building_risk.to_crs(crs).copy()
    gdf[metric] = pd.to_numeric(gdf[metric], errors="coerce").fillna(0.0)
    positive = gdf[gdf[metric] > 0].sort_values(metric, ascending=False)
    selected = []
    for row in positive.itertuples(index=False):
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        point = geom if geom.geom_type == "Point" else geom.centroid
        if any(point.distance(existing) < float(min_distance_m) for existing in selected):
            continue
        selected.append(point)
        if len(selected) >= int(top_n):
            break
    rows = []
    for i, point in enumerate(selected, start=1):
        nearby = gdf[gdf.geometry.distance(point) <= float(min_distance_m)]
        nearby_damage = float(nearby[metric].sum())
        rows.append(
            {
                "candidate_id": f"risk_{i:03d}",
                "candidate_source": "fiat_damage_cluster",
                "candidate_label": f"FIAT risk cluster {i}",
                "x": float(point.x),
                "y": float(point.y),
                "nearby_annual_damage": nearby_damage,
                "geometry": point,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def sample_sfincs_at_candidates(
    runs: pd.DataFrame,
    candidates: gpd.GeoDataFrame,
    *,
    sample_radius_m: float = 120.0,
    depth_column: str = "nearby_max_depth_ft",
) -> pd.DataFrame:
    """Sample local peak flood depth around each candidate for every completed event."""
    if runs.empty or candidates.empty:
        return pd.DataFrame(
            columns=["event_id", "design_scenario", "candidate_id", "sampled_depth_ft", depth_column, "sampled_cell_count"]
        )
    cand = candidates.to_crs(MODEL_CRS).copy()
    rows = []
    for run in runs.itertuples(index=False):
        data = diagnostics.masked_sfincs_depth(run.map_path)
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        depth = np.asarray(data["depth_ft"], dtype=float)
        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        flat_depth = depth.reshape(-1)
        for candidate in cand.itertuples(index=False):
            point = candidate.geometry
            dist2 = (flat_x - point.x) ** 2 + (flat_y - point.y) ** 2
            within = dist2 <= float(sample_radius_m) ** 2
            if np.any(within):
                vals = flat_depth[within]
                sampled_cell_count = int(np.count_nonzero(np.isfinite(vals)))
                local_max = float(np.nanmax(vals)) if np.isfinite(vals).any() else 0.0
            else:
                idx = int(np.nanargmin(dist2))
                val = flat_depth[idx]
                sampled_cell_count = 1 if np.isfinite(val) else 0
                local_max = float(val) if np.isfinite(val) else 0.0
            rows.append(
                {
                    "event_id": run.event_id,
                    "design_scenario": run.design_scenario,
                    "candidate_id": candidate.candidate_id,
                    "sampled_depth_ft": local_max,
                    depth_column: local_max,
                    "sampled_cell_count": sampled_cell_count,
                }
            )
    return pd.DataFrame(rows)


def candidate_event_response_table(
    samples: pd.DataFrame,
    event_outcomes: pd.DataFrame,
    *,
    damage_column: str = "total_damage",
) -> pd.DataFrame:
    """Join candidate flood response samples to event drivers, rates, and damages."""
    if samples.empty:
        return samples.copy()
    join_cols = ["event_id", "design_scenario"]
    event_cols = [c for c in event_outcomes.columns if c not in samples.columns or c in join_cols]
    out = samples.merge(event_outcomes[event_cols].drop_duplicates(join_cols), on=join_cols, how="left")
    if damage_column in out:
        out[damage_column] = pd.to_numeric(out[damage_column], errors="coerce")
    return out


def score_sensor_candidates(
    response: pd.DataFrame,
    candidates: gpd.GeoDataFrame,
    *,
    signal_column: str = "nearby_max_depth_ft",
    damage_column: str = "total_damage",
    weight_column: str = "probability_weight",
) -> gpd.GeoDataFrame:
    """Score candidate sensors by weighted association with event damages."""
    if candidates.empty:
        return _empty_candidates(candidates.crs or MODEL_CRS)
    rows = []
    total_nearby_damage = float(pd.to_numeric(candidates.get("nearby_annual_damage", 0.0), errors="coerce").fillna(0.0).sum())
    for candidate_id, sub in response.groupby("candidate_id"):
        signal = pd.to_numeric(sub.get(signal_column), errors="coerce")
        damage = pd.to_numeric(sub.get(damage_column), errors="coerce")
        weights = pd.to_numeric(sub.get(weight_column), errors="coerce").fillna(0.0)
        valid = signal.notna() & damage.notna() & weights.notna() & (weights >= 0)
        corr = _weighted_corr(signal[valid].to_numpy(float), damage[valid].to_numpy(float), weights[valid].to_numpy(float))
        auc_like = _weighted_high_damage_auc(signal[valid].to_numpy(float), damage[valid].to_numpy(float), weights[valid].to_numpy(float))
        rows.append(
            {
                "candidate_id": candidate_id,
                "weighted_damage_correlation": corr,
                "high_damage_auc_like_score": auc_like,
                "events_reviewed": int(valid.sum()),
            }
        )
    score = pd.DataFrame(rows)
    out = candidates.copy().merge(score, on="candidate_id", how="left")
    out["nearby_annual_damage"] = pd.to_numeric(out.get("nearby_annual_damage"), errors="coerce").fillna(0.0)
    out["represented_damage_share"] = (
        out["nearby_annual_damage"] / total_nearby_damage if total_nearby_damage > 0 else 0.0
    )
    corr_component = pd.to_numeric(out["weighted_damage_correlation"], errors="coerce").clip(lower=0.0).fillna(0.0)
    auc_component = pd.to_numeric(out["high_damage_auc_like_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    share_component = pd.to_numeric(out["represented_damage_share"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out["score"] = 0.55 * corr_component + 0.35 * auc_component + 0.10 * share_component
    out["selected_rank"] = pd.NA
    out["x"] = out.geometry.x
    out["y"] = out.geometry.y
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def greedy_sensor_selection(
    scores: gpd.GeoDataFrame,
    *,
    sensor_count: int = 3,
    min_distance_m: float = 350.0,
) -> gpd.GeoDataFrame:
    """Greedily select high-score candidates while enforcing minimum spacing."""
    if scores.empty:
        return scores.copy()
    gdf = scores.copy()
    metric_crs = gdf.crs or MODEL_CRS
    ranked = gdf.sort_values("score", ascending=False).to_crs(metric_crs)
    selected_indices = []
    selected_points = []
    for idx, row in ranked.iterrows():
        point = row.geometry
        if any(point.distance(existing) < float(min_distance_m) for existing in selected_points):
            continue
        selected_indices.append(idx)
        selected_points.append(point)
        if len(selected_indices) >= int(sensor_count):
            break
    selected = gdf.loc[selected_indices].copy()
    selected["selected_rank"] = range(1, len(selected) + 1)
    return selected.sort_values("selected_rank").reset_index(drop=True)


def mark_selected_candidates(scores: gpd.GeoDataFrame, selected: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Copy selected ranks back onto the full candidate score table."""
    out = scores.copy()
    out["selected_rank"] = pd.NA
    if not selected.empty:
        ranks = selected.set_index("candidate_id")["selected_rank"]
        out["selected_rank"] = out["candidate_id"].map(ranks)
    return out


def plot_candidate_scores(
    scores: gpd.GeoDataFrame,
    *,
    building_risk: gpd.GeoDataFrame | None = None,
    ax=None,
    title: str = "Candidate water-level sensor scores",
    basemap_style: str = "osm",
):
    """Map ranked candidate scores over optional FIAT annualized building risk."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    crs = scores.crs or MODEL_CRS
    building_layer = building_risk.to_crs(crs) if building_risk is not None and not building_risk.empty else None
    _set_bounds(ax, [scores, building_layer])
    _try_basemap(ax, crs=crs, style=basemap_style)
    if building_risk is not None and not building_risk.empty:
        risk = building_layer.copy()
        risk["annual_damage"] = pd.to_numeric(risk.get("annual_damage"), errors="coerce").fillna(0.0)
        positive = risk[risk["annual_damage"] > 0]
        if not positive.empty:
            sizes = _scaled_sizes(positive["annual_damage"], min_size=4, max_size=42)
            positive.plot(ax=ax, markersize=sizes, color="#d95f0e", alpha=0.34, linewidth=0, zorder=2)
    if not scores.empty:
        values = pd.to_numeric(scores["score"], errors="coerce").fillna(0.0)
        plotted = ax.scatter(
            scores.geometry.x,
            scores.geometry.y,
            c=values,
            s=_scaled_sizes(values, min_size=60, max_size=220),
            cmap="viridis",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        plt.colorbar(plotted, ax=ax, shrink=0.78, label="candidate score")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_selected_sensor_network(
    scores: gpd.GeoDataFrame,
    selected: gpd.GeoDataFrame,
    *,
    building_risk: gpd.GeoDataFrame | None = None,
    transects: gpd.GeoDataFrame | None = None,
    ax=None,
    title: str = "Selected deployable water-level sensor network",
    basemap_style: str = "osm",
):
    """Map selected sensor network against candidates, risk, and runup transects."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 8))
    crs = scores.crs or selected.crs or MODEL_CRS
    building_layer = building_risk.to_crs(crs) if building_risk is not None and not building_risk.empty else None
    transect_layer = transects.to_crs(crs) if transects is not None and not transects.empty else None
    score_layer = scores.to_crs(crs) if scores is not None and not scores.empty else None
    selected_layer = selected.to_crs(crs) if selected is not None and not selected.empty else None
    _set_bounds(ax, [score_layer, selected_layer, building_layer, transect_layer])
    _try_basemap(ax, crs=crs, style=basemap_style)
    if building_risk is not None and not building_risk.empty:
        risk = building_layer.copy()
        risk["annual_damage"] = pd.to_numeric(risk.get("annual_damage"), errors="coerce").fillna(0.0)
        risk[risk["annual_damage"] > 0].plot(ax=ax, markersize=5, color="#fd8d3c", alpha=0.25, linewidth=0, zorder=2)
    if transects is not None and not transects.empty:
        transect_layer.plot(ax=ax, color="#3182bd", linewidth=2.0, alpha=0.65, zorder=3, label="SFINCS runup transect")
    if scores is not None and not scores.empty:
        score_layer.plot(ax=ax, markersize=24, color="0.2", alpha=0.35, linewidth=0, zorder=4, label="candidate")
    if selected is not None and not selected.empty:
        sel = selected_layer
        sel.plot(ax=ax, markersize=170, color="#31a354", edgecolor="white", linewidth=1.4, zorder=5, label="selected")
        for row in sel.itertuples(index=False):
            ax.annotate(
                str(int(row.selected_rank)),
                xy=(row.geometry.x, row.geometry.y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color="white",
                zorder=6,
            )
    ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax


def plot_candidate_damage_response(
    response: pd.DataFrame,
    candidate_id: str,
    *,
    signal_column: str = "nearby_max_depth_ft",
    damage_column: str = "total_damage",
    storm_type_column: str = "storm_type",
    ax=None,
    title: str | None = None,
):
    """Scatter candidate signal against event damage, colored by storm type."""
    sub = response[response["candidate_id"].astype(str) == str(candidate_id)].copy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    if sub.empty:
        ax.text(0.5, 0.5, f"No response rows for {candidate_id}", ha="center", va="center", transform=ax.transAxes)
        return ax
    sub[signal_column] = pd.to_numeric(sub[signal_column], errors="coerce")
    sub[damage_column] = pd.to_numeric(sub[damage_column], errors="coerce")
    weights = pd.to_numeric(sub.get("probability_weight"), errors="coerce").fillna(0.0)
    sub["_plot_size"] = 24 + 220 * np.sqrt(weights / weights.max()) if weights.max() > 0 else 36
    if storm_type_column in sub:
        for storm_type, group in sub.groupby(storm_type_column, dropna=False):
            ax.scatter(
                group[signal_column],
                group[damage_column],
                s=group["_plot_size"],
                alpha=0.7,
                edgecolor="white",
                linewidth=0.4,
                label=str(storm_type),
            )
        ax.legend(title="storm type", fontsize=8)
    else:
        ax.scatter(sub[signal_column], sub[damage_column], s=sub["_plot_size"], alpha=0.7, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(signal_column.replace("_", " "))
    ax.set_ylabel(damage_column.replace("_", " "))
    ax.set_title(title or f"{candidate_id}: flood signal vs damage")
    return ax


def _load_runup_transects_from_rug(path: Path, *, crs: str) -> gpd.GeoDataFrame:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    i = 0
    while i < len(lines):
        name = lines[i].strip("'\"")
        if i + 3 >= len(lines):
            break
        coords = []
        for coord_line in (lines[i + 2], lines[i + 3]):
            parts = coord_line.replace(",", " ").split()
            coords.append((float(parts[0]), float(parts[1])))
        rows.append({"gauge": name, "candidate_source": "computational_runup_transect", "geometry": LineString(coords)})
        i += 4
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)


def _empty_candidates(crs: str):
    return gpd.GeoDataFrame(
        columns=[
            "candidate_id",
            "candidate_source",
            "candidate_label",
            "x",
            "y",
            "nearby_annual_damage",
            "geometry",
        ],
        geometry="geometry",
        crs=crs,
    )


def _weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if valid.sum() < 2:
        return math.nan
    x = x[valid]
    y = y[valid]
    w = w[valid]
    w_sum = float(w.sum())
    if w_sum <= 0:
        return math.nan
    mx = float(np.sum(w * x) / w_sum)
    my = float(np.sum(w * y) / w_sum)
    vx = float(np.sum(w * (x - mx) ** 2) / w_sum)
    vy = float(np.sum(w * (y - my) ** 2) / w_sum)
    if vx <= 0 or vy <= 0:
        return math.nan
    cov = float(np.sum(w * (x - mx) * (y - my)) / w_sum)
    return cov / math.sqrt(vx * vy)


def _weighted_high_damage_auc(signal: np.ndarray, damage: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(signal) & np.isfinite(damage) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return math.nan
    signal = signal[valid]
    damage = damage[valid]
    weights = weights[valid]
    positive_damage = damage[damage > 0]
    if len(positive_damage) == 0:
        return math.nan
    threshold = float(np.nanquantile(positive_damage, 0.75))
    high = damage >= threshold
    low = ~high
    if not high.any() or not low.any():
        return math.nan
    high_signal = signal[high]
    low_signal = signal[low]
    high_weights = weights[high]
    low_weights = weights[low]
    pair_weight = high_weights[:, None] * low_weights[None, :]
    comparison = (high_signal[:, None] > low_signal[None, :]).astype(float)
    comparison += 0.5 * (high_signal[:, None] == low_signal[None, :])
    denom = float(pair_weight.sum())
    return float((pair_weight * comparison).sum() / denom) if denom > 0 else math.nan


def _scaled_sizes(values, *, min_size: float, max_size: float) -> np.ndarray:
    vals = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0).to_numpy(float)
    if len(vals) == 0 or np.nanmax(vals) <= 0:
        return np.full(len(vals), min_size)
    scaled = np.sqrt(np.clip(vals, 0.0, None) / np.nanmax(vals))
    return min_size + (max_size - min_size) * scaled


def _set_bounds(ax, layers) -> None:
    bounds = []
    for layer in layers:
        if layer is None or layer.empty:
            continue
        minx, miny, maxx, maxy = layer.total_bounds
        if np.all(np.isfinite([minx, miny, maxx, maxy])):
            bounds.append((minx, miny, maxx, maxy))
    if not bounds:
        return
    minx = min(b[0] for b in bounds)
    miny = min(b[1] for b in bounds)
    maxx = max(b[2] for b in bounds)
    maxy = max(b[3] for b in bounds)
    pad = max(maxx - minx, maxy - miny, 500.0) * 0.12
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)


def _try_basemap(ax, *, crs: str, style: str) -> None:
    providers = {
        "dark": ctx.providers.CartoDB.DarkMatter,
        "satellite": ctx.providers.Esri.WorldImagery,
        "osm": ctx.providers.OpenStreetMap.HOT,
    }
    try:
        ctx.add_basemap(ax, crs=crs, source=providers.get(style, providers["osm"]), attribution_size=7)
    except Exception as exc:
        ax.text(0.01, 0.01, f"Basemap unavailable: {exc}", transform=ax.transAxes, fontsize=8)


load_transects = load_runup_transects
runup_candidates = candidate_points_from_runup_transects
risk_candidates = candidate_points_from_building_risk
sample_candidates = sample_sfincs_at_candidates
response_table = candidate_event_response_table
score_candidates = score_sensor_candidates
select_sensors = greedy_sensor_selection
mark_selected = mark_selected_candidates
plot_network = plot_selected_sensor_network
plot_response = plot_candidate_damage_response
