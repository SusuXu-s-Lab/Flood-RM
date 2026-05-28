from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def default_severity_bands():
    return [
        {"severity_band": "mild", "rp_min_years": 0.0, "rp_max_years": 2.0},
        {"severity_band": "common", "rp_min_years": 2.0, "rp_max_years": 10.0},
        {"severity_band": "significant", "rp_min_years": 10.0, "rp_max_years": 50.0},
        {"severity_band": "rare", "rp_min_years": 50.0, "rp_max_years": 100.0},
        {"severity_band": "extreme", "rp_min_years": 100.0, "rp_max_years": 500.0},
        {"severity_band": "beyond_design", "rp_min_years": 500.0, "rp_max_years": None},
    ]


def _configured_bands(config):
    bands = config.get("sampling", {}).get("severity_bands")
    return bands or default_severity_bands()


def assign_severity_bands(return_periods, bands=None):
    bands = bands or default_severity_bands()
    rp = pd.to_numeric(pd.Series(return_periods), errors="coerce")
    out = pd.Series(["unclassified"] * len(rp), index=rp.index, dtype="object")
    for band in bands:
        name = str(band["severity_band"])
        lower = float(band.get("rp_min_years", 0.0))
        upper = band.get("rp_max_years")
        mask = rp >= lower
        if upper is not None:
            mask &= rp < float(upper)
        out.loc[mask] = name
    out.loc[rp.isna() | (rp < 0)] = "unclassified"
    return out


def _peak_column(df):
    for column in ["coastal_peak_m", "peak_m", "peak", "absolute_peak_m"]:
        if column in df:
            return column
    return None


def _as_distribution_frame(events, bands):
    df = events.copy()
    if "severity_band" not in df:
        df["severity_band"] = assign_severity_bands(df["sample_rp_years"], bands=bands)
    if "sampling_weight" not in df:
        df["sampling_weight"] = 1.0
    if "probability_weight" not in df:
        df["probability_weight"] = np.nan
    if "sampling_region" not in df:
        df["sampling_region"] = "unknown"
    df["sampling_weight"] = pd.to_numeric(df["sampling_weight"], errors="coerce")
    df["probability_weight"] = pd.to_numeric(df["probability_weight"], errors="coerce")
    if df["probability_weight"].notna().sum() == 0:
        total = float(df["sampling_weight"].sum())
        if np.isfinite(total) and total > 0:
            df["probability_weight"] = df["sampling_weight"] / total
    df["sample_rp_years"] = pd.to_numeric(df["sample_rp_years"], errors="coerce")
    peak = _peak_column(df)
    if peak is not None:
        df["distribution_peak_m"] = pd.to_numeric(df[peak], errors="coerce")
    else:
        df["distribution_peak_m"] = np.nan
    return df


def summarize_event_distribution(events, config=None):
    config = config or {}
    bands = _configured_bands(config)
    df = _as_distribution_frame(events, bands)
    total = max(len(df), 1)
    weighted_total = float(df["sampling_weight"].sum())
    if not np.isfinite(weighted_total) or weighted_total <= 0:
        weighted_total = np.nan
    probability_total = float(df["probability_weight"].sum())
    if not np.isfinite(probability_total) or probability_total <= 0:
        probability_total = np.nan

    rows = []
    for band in bands:
        name = str(band["severity_band"])
        subset = df[df["severity_band"] == name]
        weighted_count = float(subset["sampling_weight"].sum()) if len(subset) else 0.0
        probability_mass = float(subset["probability_weight"].sum()) if len(subset) else 0.0
        rows.append(
            {
                "severity_band": name,
                "rp_min_years": float(band.get("rp_min_years", 0.0)),
                "rp_max_years": band.get("rp_max_years"),
                "event_count": int(len(subset)),
                "unweighted_fraction": float(len(subset) / total),
                "weighted_count": weighted_count,
                "weighted_fraction": float(weighted_count / weighted_total) if np.isfinite(weighted_total) else np.nan,
                "probability_mass": probability_mass,
                "probability_fraction": float(probability_mass / probability_total) if np.isfinite(probability_total) else np.nan,
                "body_count": int((subset["sampling_region"] == "body").sum()),
                "tail_count": int((subset["sampling_region"] == "tail").sum()),
                "rp_min_observed": float(subset["sample_rp_years"].min()) if len(subset) else np.nan,
                "rp_median_observed": float(subset["sample_rp_years"].median()) if len(subset) else np.nan,
                "rp_max_observed": float(subset["sample_rp_years"].max()) if len(subset) else np.nan,
                "peak_min_m": float(subset["distribution_peak_m"].min()) if len(subset) else np.nan,
                "peak_median_m": float(subset["distribution_peak_m"].median()) if len(subset) else np.nan,
                "peak_max_m": float(subset["distribution_peak_m"].max()) if len(subset) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _read_distribution_source(paths):
    catalog_path = paths.get("event_catalog_csv")
    if catalog_path is not None and Path(catalog_path).exists():
        catalog = pd.read_csv(catalog_path)
        if {"sampling_region", "sampling_weight"}.issubset(catalog.columns):
            return "event_catalog", Path(catalog_path), catalog
    sampled_path = paths.get("sampled_peaks_csv")
    if sampled_path is not None and Path(sampled_path).exists():
        return "sampled_peaks", Path(sampled_path), pd.read_csv(sampled_path)
    if catalog_path is not None and Path(catalog_path).exists():
        return "event_catalog", Path(catalog_path), pd.read_csv(catalog_path)
    raise FileNotFoundError("no event catalog or sampled peaks file found for event distribution")


def _write_distribution_plot(events, summary, path):
    cache = Path(path).resolve().parents[1] / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = events.copy()
    df["sample_rp_years"] = pd.to_numeric(df["sample_rp_years"], errors="coerce")
    df["sampling_weight"] = pd.to_numeric(df.get("sampling_weight", 1.0), errors="coerce")
    if "probability_weight" in df:
        df["probability_weight"] = pd.to_numeric(df["probability_weight"], errors="coerce")
    else:
        total = float(df["sampling_weight"].sum())
        df["probability_weight"] = df["sampling_weight"] / total if np.isfinite(total) and total > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    valid_rp = df["sample_rp_years"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid_rp):
        rp_low = max(1.0, float(valid_rp.min()))
        rp_high = max(float(valid_rp.max()), rp_low * 1.01, 2.0)
        bins = np.geomspace(rp_low, rp_high, 24)
        axes[0].hist(valid_rp, bins=bins, color="#4c78a8", alpha=0.70, label="sampled count")
        axes[0].hist(
            valid_rp,
            bins=bins,
            weights=df.loc[valid_rp.index, "probability_weight"],
            histtype="step",
            linewidth=2,
            color="#f58518",
            label="probability mass",
        )
        axes[0].set_xscale("log")
    axes[0].set_xlabel("Coastal driver return period (years)")
    axes[0].set_ylabel("Events")
    axes[0].legend()

    x = np.arange(len(summary))
    axes[1].bar(x - 0.18, summary["unweighted_fraction"], width=0.36, color="#4c78a8", label="sampled")
    axes[1].bar(x + 0.18, summary["probability_fraction"], width=0.36, color="#f58518", label="probability")
    axes[1].set_xticks(x, summary["severity_band"], rotation=30, ha="right")
    axes[1].set_ylabel("Fraction")
    axes[1].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_event_distribution_artifacts(config, paths):
    source_name, source_path, events = _read_distribution_source(paths)
    bands = _configured_bands(config)
    events = _as_distribution_frame(events, bands)
    summary = summarize_event_distribution(events, config=config)

    summary_csv = Path(paths["event_distribution_summary_csv"])
    summary_json = Path(paths["event_distribution_summary_json"])
    plot_png = Path(paths["event_distribution_plot_png"])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    _write_distribution_plot(events, summary, plot_png)

    payload = {
        "source": source_name,
        "source_file": str(source_path),
        "event_count": int(len(events)),
        "weighted_event_count": float(events["sampling_weight"].sum()),
        "probability_weight_total": float(events["probability_weight"].sum()),
        "bands": json.loads(summary.to_json(orient="records")),
        "artifacts": {
            "summary_csv": str(summary_csv),
            "plot_png": str(plot_png),
        },
    }
    summary_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
