from __future__ import annotations

from pathlib import Path


def open_model(
    root: str | Path,
    *,
    mode: str = "r+",
    read: bool = True,
    write_gis: bool = True,
    data_libs=None,
):
    """Open a HydroMT-SFINCS model using the native model class."""
    from hydromt_sfincs import SfincsModel

    sf = SfincsModel(root=str(root), mode=mode, write_gis=write_gis, data_libs=data_libs)
    if read:
        sf.read()
    return sf
