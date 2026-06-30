from collect_sources.mgr import bind_module

bind_module(__name__, "collect_sources.mgr.workflow", globals())

from importlib import import_module


COLLECTORS = {
    "aorc": "collect_sources.aorc:collect",
    "aorc_sst": "collect_sources.rainfall:collect",
    "cora": "collect_sources.cora:collect_cora",
    "era5": "collect_sources.era5:collect",
    "era5_waves": "collect_sources.era5:collect",
    "nwm": "collect_sources.nwm:collect_nwm",
    "usgs": "collect_sources.usgs:collect",
    "usgs_streamgages": "collect_sources.usgs_streamgages:collect_usgs_streamgages",
    "hurdat2": "collect_sources.hurdat2:collect_hurdat2",
    "lcra_hydromet": "collect_sources.lcra_hydromet:collect_lcra_hydromet",
    "stream_geo": "collect_sources.stream_geo:collect",
    "stream_geo_nldi": "collect_sources.stream_geo_nldi:collect_stream_geo_nldi",
    "ssurgo": "collect_sources.ssurgo:collect_ssurgo",
    "national_hydrography": "collect_sources.national_hydrography:collect_national_hydrography",
}


def _collector(name: str):
    module_name, function_name = COLLECTORS[name].split(":")
    return getattr(import_module(module_name), function_name)
