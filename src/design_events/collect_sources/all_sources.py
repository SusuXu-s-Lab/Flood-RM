from __future__ import annotations

import pandas as pd

from design_events.collect_sources.plan import build_source_collection_plan


def _default_funcs():
    from design_events.collect_sources.cora import collect_cora
    from design_events.collect_sources.usgs_streamgages import collect_usgs_streamgages
    from design_events.collect_sources.nwm import collect_nwm
    from design_events.collect_sources.national_hydrography import collect_national_hydrography
    from design_events.collect_sources.era5_waves import collect_era5_waves
    from design_events.collect_sources.aorc_sst import collect_aorc_sst
    from design_events.collect_sources.hurdat2 import collect_hurdat2

    return {
        "collect_cora": collect_cora,
        "collect_usgs_streamgages": collect_usgs_streamgages,
        "collect_nwm": collect_nwm,
        "collect_national_hydrography": collect_national_hydrography,
        "collect_aorc_sst": collect_aorc_sst,
        "collect_era5_waves": collect_era5_waves,
        "collect_hurdat2": collect_hurdat2,
    }


def collect_all_sources(
    config,
    paths,
    *,
    start=None,
    end=None,
    skip_existing=False,
    smoke=False,
    funcs=None,
):
    funcs = {**_default_funcs(), **(funcs or {})}
    plan = build_source_collection_plan(config, paths, start=start, end=end)
    cora_frame = pd.DataFrame()
    if plan.has("cora"):
        cora_frame = funcs["collect_cora"](
            plan.settings_for("cora"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    usgs_streamgages_result = None
    if plan.has("usgs_streamgages"):
        usgs_streamgages_result = funcs["collect_usgs_streamgages"](
            plan.settings_for("usgs_streamgages"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    nwm_result = None
    national_hydrography_result = None
    if plan.has("national_hydrography"):
        national_hydrography_result = funcs["collect_national_hydrography"](
            plan.settings_for("national_hydrography"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    if plan.has("nwm"):
        nwm_result = funcs["collect_nwm"](
            plan.settings_for("nwm"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    aorc_sst_result = None
    if plan.has("aorc_sst"):
        aorc_sst_result = funcs["collect_aorc_sst"](
            plan.settings_for("aorc_sst"),
            skip_existing=skip_existing,
        )
    era5_result = None
    if plan.has("era5_waves"):
        era5_result = funcs["collect_era5_waves"](
            plan.settings_for("era5_waves"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    hurdat2_result = None
    if plan.has("hurdat2"):
        hurdat2_result = funcs["collect_hurdat2"](
            plan.settings_for("hurdat2"),
            skip_existing=skip_existing,
            smoke=smoke,
        )
    return {
        "cora_rows": int(len(cora_frame)),
        "waterlevel_csv": paths.get("waterlevel_csv"),
        "usgs_streamgages": usgs_streamgages_result,
        "nwm": nwm_result,
        "national_hydrography": national_hydrography_result,
        "aorc_sst": aorc_sst_result,
        "era5_waves": era5_result,
        "hurdat2": hurdat2_result,
    }
