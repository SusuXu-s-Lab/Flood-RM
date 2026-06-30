from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shlex
import shutil
import subprocess
import tomllib


def wflow_run_command(config: dict[str, Any]) -> str:
    run_cfg = ((config.get("wflow", {}) or {}).get("run", {}) or {})
    env = str(run_cfg.get("bin_env") or "WFLOW_BIN")
    cmd = os.environ.get(env) or run_cfg.get("command") or "wflow_cli {run_config}"
    return cmd if "{run_config}" in cmd else f"{cmd} {{run_config}}"


def run_solver(config: dict[str, Any], run_config: str | Path, *, cwd: str | Path) -> None:
    command = shlex.split(wflow_run_command(config).format(run_config=Path(run_config)))
    subprocess.run(command, cwd=Path(cwd), check=True)


def clean_output_dir(run_config: str | Path) -> None:
    run_config = Path(run_config).resolve()
    if not run_config.exists():
        return
    cfg = tomllib.loads(run_config.read_text(encoding="utf-8"))
    out = cfg.get("dir_output")
    if not out:
        return
    model_root = run_config.parent
    out_dir = Path(out)
    if not out_dir.is_absolute():
        out_dir = model_root / out_dir
    out_dir = out_dir.resolve()
    if out_dir == model_root or model_root not in out_dir.parents:
        raise ValueError(f"refusing to clean Wflow output outside model dir: {out_dir}")
    if out_dir.exists():
        shutil.rmtree(out_dir) if out_dir.is_dir() else out_dir.unlink()


def zero_event_forcing(path: str | Path) -> Path:
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as opened:
        ds = opened.load()
    if "precip" not in ds:
        raise ValueError(f"{path} lacks precip for zero-rain control")
    ds["precip"] = ds["precip"] * 0
    if "river_inflow" in ds:
        ds["river_inflow"] = ds["river_inflow"] * 0
    tmp = path.with_suffix(".zero.tmp.nc")
    ds.to_netcdf(tmp)
    tmp.replace(path)
    return path
