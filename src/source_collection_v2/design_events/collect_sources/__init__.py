"""Source-name collection adapters only; science lives in design_events.stochastic_boundary."""

from .aorc import collect as collect_aorc
from .cora import collect as collect_cora
from .era5 import collect as collect_era5
from .hurdat2 import collect as collect_hurdat2
from .lcra_hydromet import collect as collect_lcra_hydromet
from .national_hydrography import collect as collect_national_hydrography
from .nwm import collect as collect_nwm
from .ssurgo import collect as collect_ssurgo
from .stream_geo import collect as collect_stream_geo
from .usgs import collect as collect_usgs

__all__ = [name for name in globals() if name.startswith("collect_")]
