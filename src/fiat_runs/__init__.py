"""Delft-FIAT risk stage for the Flood-RM coastal pipeline.

Turns the wave-coupled SFINCS hazard ensemble (``06`` outputs) into per-asset damage
and Expected Annual Damage (EAD) using Delft-FIAT as the damage engine and the
event catalog's importance-sampling probability weights for the risk integral.

Environment split (see :mod:`fiat_runs._env`): hydromt_fiat (build) and the delft_fiat
engine run in an isolated conda env; hazard export and EAD integration run in the main
project env.
"""

from __future__ import annotations

from ._env import FIAT_CONDA_ENV, fiat_env_available, run_in_fiat_env
from .build_model import (
    apply_dem_ground_elevation,
    build_fiat_model,
    fiat_model_inputs,
    fiat_model_is_built,
)
from . import diagnostics, risk, risk_native, validate
from .config import fiat_paths, load_runtime
from .hazard import WaterLevelRasterizer
from .run import read_fiat_damages, run_fiat_event

env_ready = fiat_env_available
build_model = build_fiat_model
apply_ground = apply_dem_ground_elevation
model_ready = fiat_model_is_built
run_event = run_fiat_event

__all__ = [
    "load_runtime",
    "fiat_paths",
    "fiat_env_available",
    "run_in_fiat_env",
    "FIAT_CONDA_ENV",
    "build_fiat_model",
    "apply_dem_ground_elevation",
    "fiat_model_inputs",
    "fiat_model_is_built",
    "WaterLevelRasterizer",
    "run_fiat_event",
    "read_fiat_damages",
    "risk",
    "risk_native",
    "diagnostics",
    "validate",
    "env_ready",
    "build_model",
    "apply_ground",
    "model_ready",
    "run_event",
]
