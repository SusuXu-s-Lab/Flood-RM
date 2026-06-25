"""Bridge to the isolated conda ``fiat`` environment.

The main project ``.venv`` runs hydromt 1.x / hydromt_sfincs 2.x. It cannot host
``hydromt_fiat`` (which pins hydromt 0.10 and would break the SFINCS pipeline) nor
``delft_fiat`` (whose ``gdal`` binding needs system GDAL that pip cannot supply here).
FIAT model building and the FIAT engine therefore live in a separate conda env, and
this module shells into it. See ``.claude/plans/tingly-forging-toast.md``.

Everything else in :mod:`fiat_runs` (hazard export, EAD integration, plotting) runs
in the main env on xarray/pandas and never touches this bridge.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: Name of the conda environment that holds hydromt_fiat + delft_fiat.
FIAT_CONDA_ENV = os.environ.get("FIAT_CONDA_ENV", "fiat")


def conda_executable() -> str:
    """Locate a conda launcher for ``conda run`` (prefer conda over mamba for run)."""
    exe = shutil.which("conda") or shutil.which("mamba")
    if exe:
        return exe
    candidate = Path.home() / "miniconda3" / "condabin" / "conda"
    if candidate.exists():
        return str(candidate)
    raise RuntimeError(
        "conda/mamba not found. The FIAT bridge needs the isolated conda env "
        f"'{FIAT_CONDA_ENV}'. Create it per the 07_risk_fiat plan."
    )


def fiat_env_available() -> bool:
    """Return True if the conda FIAT env imports hydromt_fiat and fiat cleanly."""
    try:
        run_in_fiat_env(
            ["python", "-c", "import hydromt_fiat, fiat"],
            capture_output=True,
        )
    except (RuntimeError, subprocess.CalledProcessError):
        return False
    return True


def run_in_fiat_env(args, *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run ``args`` inside the conda FIAT env, raising on a non-zero exit by default.

    ``args`` is the command *after* the interpreter selection, e.g.
    ``["python", script, ...]`` or ``["fiat", "run", "settings.toml"]``.
    """
    cmd = [conda_executable(), "run", "-n", FIAT_CONDA_ENV, *args]
    return subprocess.run(cmd, check=check, text=True, **kwargs)


env_ready = fiat_env_available
