"""Notebook-facing synthetic validation audit helpers in the shallow power namespace.

Most validation logic lives in :mod:`power._audit_core`.  This module keeps the
Marshfield notebook import path and report filenames stable while avoiding the
old high-LOC compatibility implementation.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from power._audit_core import *  # noqa: F403
from power import _audit_core as _core


def write_synthetic_validation_report(
    report: dict[str, Any],
    output_dir: Path = _core.default_report_dir,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "marshfield_synthetic_validation_audit.json"
    markdown_path = output_dir / "marshfield_synthetic_validation_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(_core.render_markdown_report(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}


def _metric_status_table(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for metric in report.get("statistical_metrics", []):
        row = {
            key: metric.get(key)
            for key in (
                "grade",
                "feeder_count",
                "typical_fraction",
                "uncommon_fraction",
                "rare_fraction",
                "min",
                "median",
                "max",
                "value",
            )
            if metric.get(key) is not None
        }
        table[str(metric["metric_id"])] = row
    return table


def _infer_smart_ds_compat_dir(registry_dir: Path) -> Path:
    candidates = [
        registry_dir.parent / "augmented",
        registry_dir.parent / "smart_ds_compat",
        registry_dir.parent.parent / "static" / "power_grid" / "smart_ds_compat",
        _core.default_smart_ds_compat_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def run_stats(
    *,
    registry_dir: Path = _core.default_registry_dir,
    smart_ds_reference_dir: Path | None = None,
    smart_ds_compat_dir: Path | None = None,
    grid_network_dir: Path = _core.default_grid_network_dir,
    output_dir: Path = _core.default_report_dir,
) -> dict[str, dict[str, Any]]:
    """Build statistical audit artifacts and return a display table."""

    del smart_ds_reference_dir
    registry_dir = Path(registry_dir)
    compat_dir = Path(smart_ds_compat_dir) if smart_ds_compat_dir is not None else _infer_smart_ds_compat_dir(registry_dir)
    output_dir = Path(output_dir)
    report = _core.build_synthetic_validation_report(
        registry_dir=registry_dir,
        smart_ds_compat_dir=compat_dir,
        grid_network_dir=Path(grid_network_dir),
    )
    write_synthetic_validation_report(report, output_dir)
    _core.write_validation_compliance_gate(_core.build_validation_compliance_gate(report), output_dir)
    plot_audit(report, output_dir / "validation_region_report_card.png")
    return _metric_status_table(report)


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _operational_report_status(report: Mapping[str, Any] | None) -> str:
    if not report:
        return "missing"
    criteria = report.get("validation_exit_criteria") or {}
    if not criteria:
        return "partial" if report.get("compiled") else "gap"
    return "pass" if all(bool(item.get("passed")) for item in criteria.values()) else "check"


def run_ops(
    *,
    opendss_root: Path,
    registry_dir: Path = _core.default_registry_dir,
    output_dir: Path = _core.default_report_dir,
) -> dict[str, Any]:
    """Summarize available operational-validation evidence without a solver run."""

    root = Path(opendss_root)
    output = Path(output_dir)
    masters = sorted(root.glob("*/Master.dss")) if root.exists() else []
    power_flow = _load_optional_json(output / "power_flow_validation.json") or _load_optional_json(_core.default_power_flow_report)
    short_circuit = _load_optional_json(output / "short_circuit_validation.json") or _load_optional_json(
        _core.default_short_circuit_report
    )
    power_flow_status = _operational_report_status(power_flow)
    short_circuit_status = _operational_report_status(short_circuit)
    status = "pass" if power_flow_status == "pass" and short_circuit_status == "pass" else "partial"
    if not masters:
        status = "gap"
    summary = {
        "status": status,
        "mode": "evidence_inventory",
        "opendss_root": str(root),
        "opendss_feeder_cases": len(masters),
        "registry_dir": str(registry_dir),
        "power_flow_evidence": power_flow_status,
        "short_circuit_evidence": short_circuit_status,
        "note": (
            "This cell inventories operational evidence. Run build_power_flow_validation_report "
            "and build_short_circuit_validation_report explicitly to generate solver evidence."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "operational_validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _stage_a1_status(stage_a1: Mapping[str, Any]) -> str:
    if stage_a1.get("passed"):
        return "pass"
    if stage_a1.get("errors"):
        return "check"
    return "gap"


def _statistical_status(stat_results: Mapping[str, Mapping[str, Any]]) -> str:
    grades = [row.get("grade") for row in stat_results.values() if isinstance(row, Mapping)]
    if not grades:
        return "gap"
    if any(grade == "check" for grade in grades):
        return "check"
    if any(grade in {"marginal", "unavailable"} for grade in grades):
        return "partial"
    return "pass"


def audit_summary(
    *,
    block_report: dict[str, Any],
    stage_a1: dict[str, Any],
    stat_results: dict[str, dict[str, Any]],
    op_results: dict[str, Any],
) -> dict[str, Any]:
    """Collect the notebook audit gates into a concise report-card summary."""

    block_status = "check" if block_report.get("violations") else "pass"
    stage_status = _stage_a1_status(stage_a1)
    stat_status = _statistical_status(stat_results)
    op_status = str(op_results.get("status") or "gap")
    gate_results = {
        "block_invariant": block_status,
        "smart_ds_interface": stage_status,
        "statistical_validation": stat_status,
        "operational_validation": op_status,
    }

    gaps: list[str] = []
    if block_status != "pass":
        gaps.append("Switch-bounded load blocks still have invariant violations.")
    if stage_status != "pass":
        gaps.append("SMART-DS-compatible Stage A1 interface validation has not passed.")
    if stat_status in {"check", "gap"}:
        checked = [metric for metric, row in stat_results.items() if row.get("grade") == "check"]
        gaps.append("Statistical validation has metrics outside validation regions: " + ", ".join(checked or ["none recorded"]))
    if op_results.get("opendss_feeder_cases", 0) == 0:
        gaps.append("OpenDSS feeder cases are missing from the operational validation inventory.")
    elif op_status != "pass":
        missing = [
            name
            for name in ("power_flow_evidence", "short_circuit_evidence")
            if op_results.get(name) in {"missing", "gap", "partial"}
        ]
        gaps.append("Operational validation evidence is incomplete: " + ", ".join(missing or [op_status]))

    return {
        "gate_results": gate_results,
        "gaps": gaps,
        "block_summary": block_report.get("summary", {}),
        "statistical_metric_count": len(stat_results),
        "operational_summary": op_results,
    }


def plot_audit(report: dict[str, Any], output_path: Path) -> Path:
    """Plot validation target ranges against synthetic-grid metric ranges."""

    mpl_config_dir = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    labels = {
        "distribution_transformer_mva_per_feeder": "Transformer MVA / feeder",
        "real_load_kw_per_feeder": "Real load kW / feeder",
        "customer_count_per_feeder": "Customers / feeder",
        "switches_per_feeder": "Switches / feeder",
        "graph_diameter_miles_per_feeder": "Graph diameter (mi) / feeder",
        "characteristic_path_length_miles_per_feeder": "Path length (mi) / feeder",
    }
    metrics = [
        metric
        for metric in report.get("statistical_metrics", [])
        if metric.get("metric_id") in labels and metric.get("validation_target")
    ]
    if not metrics:
        metrics = [metric for metric in report.get("statistical_metrics", []) if metric.get("validation_target")][:6]
    if not metrics:
        raise ValueError("report contains no statistical metrics with validation_target regions")

    cols = 2
    rows = math.ceil(len(metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.55 * rows), squeeze=False)
    typical_color = "#cfe8cf"
    uncommon_color = "#fee7b5"
    range_color = "#3d5a6c"
    median_color = "#111111"

    for ax, metric in zip(axes.ravel(), metrics):
        target = metric["validation_target"]
        ranges = list(target.get("typical", ())) + list(target.get("uncommon", ()))
        high = max([float(hi) for _, hi in ranges] + [float(metric.get("max") or 0.0), 1.0])
        low = min([float(lo) for lo, _ in ranges] + [float(metric.get("min") or 0.0), 0.0])
        pad = (high - low) * 0.06 if high > low else 1.0
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.grid(axis="x", color="#d8dee4", linewidth=0.7, alpha=0.8)
        for lo, hi in target.get("uncommon", ()):
            ax.axvspan(float(lo), float(hi), color=uncommon_color, zorder=0)
        for lo, hi in target.get("typical", ()):
            ax.axvspan(float(lo), float(hi), color=typical_color, zorder=1)
        metric_min = float(metric.get("min") or 0.0)
        metric_median = float(metric.get("median") or metric_min)
        metric_max = float(metric.get("max") or metric_median)
        ax.hlines(0.52, metric_min, metric_max, color=range_color, linewidth=2.5, zorder=3)
        ax.scatter([metric_median], [0.52], color=median_color, s=34, zorder=4)
        ax.set_title(labels.get(metric["metric_id"], metric["metric_id"].replace("_", " ")), fontsize=10)
        ax.set_xlabel(f"Marshfield min-median-max, grade={metric.get('grade')}", fontsize=9)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)

    for ax in axes.ravel()[len(metrics) :]:
        ax.axis("off")

    legend = [
        Patch(facecolor=typical_color, edgecolor="none", label="Typical validation region"),
        Patch(facecolor=uncommon_color, edgecolor="none", label="Uncommon validation region"),
        Line2D([0], [0], color=range_color, lw=2.5, label="Synthetic-grid feeder range"),
        Line2D([0], [0], marker="o", color=median_color, lw=0, label="Synthetic-grid median"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Synthetic-distribution validation region report card", fontsize=14, y=0.985)
    fig.text(
        0.5,
        0.935,
        "Shaded bands are validation target regions, not utility certification.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
