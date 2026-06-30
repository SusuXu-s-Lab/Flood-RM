from __future__ import annotations

from pathlib import Path

import pandas as pd

from collect_sources.meteo import AORC_METEO as DEFAULT_AORC_METEO_VARIABLES
from collect_sources.meteo import aorc_variable_candidates
from collect_sources.meteo import write_wflow_temp_pet


def aorc_wflow_temp_pet_variables(config: dict | None = None) -> dict[str, tuple[str, ...]]:
    """Return configured AORC source-variable candidates for Wflow temp/PET forcing."""
    return aorc_variable_candidates(config)


def prepare_aorc_temp_pet_for_wflow(
    source_nc: str | Path,
    output_nc: str | Path,
    *,
    t_start: str | pd.Timestamp,
    t_stop: str | pd.Timestamp,
    precip_template: str | Path | None = None,
    variable_candidates: dict[str, tuple[str, ...]] | None = None,
    source_time_start: str | pd.Timestamp | None = None,
    freq: str = "1h",
    provenance_path: str | Path | None = None,
) -> dict:
    """Write a HydroMT-Wflow ``event_temp_pet`` dataset from AORC event fields."""
    return write_wflow_temp_pet(
        source_nc,
        output_nc,
        start=t_start,
        end=t_stop,
        freq=freq,
        precip_template=precip_template,
        variable_candidates=variable_candidates,
        source_time_start=source_time_start,
        provenance_path=provenance_path,
    )
