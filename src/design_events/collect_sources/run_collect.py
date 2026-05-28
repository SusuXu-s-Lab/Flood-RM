from __future__ import annotations

import time

import pandas as pd

from design_events.progress import iter_progress


def _default_funcs():
    from design_events.collect_sources.aorc_sst import collect_aorc_sst
    from design_events.collect_sources.cora import collect_cora
    from design_events.collect_sources.era5_waves import collect_era5_waves
    from design_events.collect_sources.nwm import collect_nwm

    return {
        "collect_aorc_sst": collect_aorc_sst,
        "collect_cora": collect_cora,
        "collect_era5_waves": collect_era5_waves,
        "collect_nwm": collect_nwm,
    }


def _record(records, source, status, started, **details):
    records.append(
        {
            "source": source,
            "status": status,
            "duration_seconds": round(time.monotonic() - started, 2),
            **details,
        }
    )


def _steps(plan, progress):
    if not progress:
        return plan.steps
    return iter_progress(plan.steps, desc="Collecting sources", unit="source")


def run_collect(
    config,
    paths,
    plan,
    *,
    run_collection=True,
    skip_existing=True,
    stop_on_error=True,
    progress=True,
    funcs=None,
):
    """Run the configured source collection plan and return notebook-friendly rows."""
    funcs = {**_default_funcs(), **(funcs or {})}
    rows = []
    if not run_collection:
        return pd.DataFrame(
            [
                {
                    "source": "collection",
                    "status": "dry_run",
                    "duration_seconds": 0.0,
                    "rows": pd.NA,
                    "artifact": pd.NA,
                }
            ]
        )

    for step in _steps(plan, progress):
        started = time.monotonic()
        settings = plan.settings_for(step.name)
        try:
            if step.name == "cora":
                frame = funcs["collect_cora"](settings, skip_existing=skip_existing, smoke=False)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=len(frame),
                    artifact=str(paths["waterlevel_csv"]),
                )
            elif step.name == "nwm":
                result = funcs["collect_nwm"](settings, skip_existing=skip_existing, smoke=False)
                status = "reused" if result.get("reused") else "collected"
                _record(
                    rows,
                    step.name,
                    status,
                    started,
                    rows=result.get("soil_moisture_rows", 0),
                    artifact=str(result.get("soil_moisture_csv")),
                )
            elif step.name == "aorc_sst":
                result = funcs["collect_aorc_sst"](settings, skip_existing=skip_existing)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=result.get("ranked_rows", 0),
                    artifact=str(result.get("ranked_storms_csv")),
                )
            elif step.name == "era5_waves":
                result = funcs["collect_era5_waves"](settings, skip_existing=skip_existing, smoke=False)
                _record(
                    rows,
                    step.name,
                    "collected",
                    started,
                    rows=result.get("time_count", 0),
                    artifact=str(result.get("wave_netcdf")),
                )
        except Exception as exc:
            _record(
                rows,
                step.name,
                "failed",
                started,
                rows=pd.NA,
                artifact=pd.NA,
                error=f"{type(exc).__name__}: {exc}",
            )
            if stop_on_error:
                raise

    started = time.monotonic()
    try:
        if plan.has("aorc_sst"):
            rainfall_members = pd.read_csv(paths["aorc_sst_rainfall_members_csv"])
        else:
            rainfall_members = pd.DataFrame()
        _record(
            rows,
            "rainfall_members",
            "collected" if plan.has("aorc_sst") else "not_configured",
            started,
            rows=len(rainfall_members),
            artifact=str(paths["aorc_sst_rainfall_members_csv"]),
        )
    except Exception as exc:
        _record(
            rows,
            "rainfall_members",
            "failed",
            started,
            rows=pd.NA,
            artifact=str(paths["aorc_sst_rainfall_members_csv"]),
            error=f"{type(exc).__name__}: {exc}",
        )
        if stop_on_error:
            raise

    return pd.DataFrame(rows)
