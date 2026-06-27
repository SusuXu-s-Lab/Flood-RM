"""Delft-FIAT risk stage for the Flood-RM coastal pipeline.

Turns the wave-coupled SFINCS hazard ensemble (``06`` outputs) into per-asset damage
and Expected Annual Damage (EAD) using Delft-FIAT as the damage engine and the
event catalog's importance-sampling probability weights for the risk integral.

Environment split (see :mod:`fiat_runs._env`): hydromt_fiat (build) and the delft_fiat
engine run in an isolated conda env; hazard export and EAD integration run in the main
project env.
"""

from __future__ import annotations

from ._env import FIAT_CONDA_ENV, env_ready, run_in_fiat_env
from .build_model import (
    apply_ground,
    build_model,
    fiat_model_inputs,
    model_ready,
)
from . import diagnostics, risk, risk_native, validate
from .config import FiatNotebookRuntime, fiat_paths, load_notebook_runtime, load_runtime
from .hazard import WaterLevelRasterizer
from .run import read_fiat_damages, run_event


__all__ = [
    "load_runtime",
    "load_notebook_runtime",
    "FiatNotebookRuntime",
    "fiat_paths",
    "env_ready",
    "run_in_fiat_env",
    "FIAT_CONDA_ENV",
    "build_model",
    "apply_ground",
    "fiat_model_inputs",
    "model_ready",
    "WaterLevelRasterizer",
    "run_event",
    "read_fiat_damages",
    "risk",
    "risk_native",
    "diagnostics",
    "validate",
]
