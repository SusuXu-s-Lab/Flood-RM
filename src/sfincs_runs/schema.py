from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

ForcingMode = Literal[
    "inland_wflow_discharge",
    "coastal_water_level",
    "compound_coastal_inland",
]


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved project paths for a SFINCS event runtime.

    Paths are deliberately location-relative and explicit.  The runtime does not
    mutate ``sys.path`` or import notebooks; callers pass a config dictionary and a
    location root.
    """

    location_root: Path
    base_model_root: Path
    scenarios_root: Path
    storage_root: Path
    run_root: Path
    data_catalog: Path | None = None
    event_catalog: Path | None = None
    wflow_events_root: Path | None = None
    source_contract: Path | None = None

    def asdict(self) -> dict[str, str | None]:
        return {key: None if value is None else str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class NativeSourceConfig:
    """HydroMT-SFINCS native river-inflow source-point settings."""

    hydrography: str | Path = "merit_hydro"
    river_upa_km2: float = 10.0
    river_len_m: float = 1000.0
    river_width_m: float = 0.0
    buffer_m: float = 200.0
    first_index: int = 1
    src_type: str = "inflow"
    keep_rivers_geom: bool = True
    reverse_river_geom: bool = False
    max_source_points: int | None = None


@dataclass(frozen=True)
class EventManifest:
    """Stakeholder-readable receipt for one SFINCS event folder."""

    event_id: str
    run_root: str
    forcing_mode: ForcingMode | str
    run_start: str
    run_stop: str
    sfincs_domain_id: str = ""
    probability_weight: float | None = None
    total_rate_per_year: float | None = None
    annual_rate: float | None = None
    wflow_discharge_nc: str = ""
    sfincs_src_names: tuple[str, ...] = ()
    coastal_water_level: bool = False
    precipitation_nc: str = ""
    netamprfile: str = ""
    initial_condition: dict[str, Any] = field(default_factory=dict)
    snapwave: bool = False
    snapwave_files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_hours(self) -> float:
        start = pd.Timestamp(self.run_start)
        stop = pd.Timestamp(self.run_stop)
        return float((stop - start) / pd.Timedelta(hours=1))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["run_duration_hours"] = self.duration_hours
        return out


@dataclass(frozen=True)
class SfincsRunResult:
    """Solver result for one SFINCS event folder."""

    event_id: str
    source_dir: Path
    stage_dir: Path
    storage_dir: Path
    log_path: Path
    map_path: Path
    returncode: int
    duration_sec: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_dir": str(self.source_dir),
            "stage_dir": str(self.stage_dir),
            "storage_dir": str(self.storage_dir),
            "log_path": str(self.log_path),
            "map_path": str(self.map_path),
            "returncode": int(self.returncode),
            "duration_sec": float(self.duration_sec),
            "status": self.status,
        }


@dataclass(frozen=True)
class SnapWaveForcing:
    """Boundary tables for one SnapWave event.

    Each frame is indexed by absolute time and has one column per SnapWave
    boundary point.  The file writer turns these into ``snapwave.bhs``,
    ``snapwave.btp``, ``snapwave.bwd``, and ``snapwave.bds``.
    """

    bhs: pd.DataFrame
    btp: pd.DataFrame
    bwd: pd.DataFrame
    bds: pd.DataFrame

    def frames(self) -> dict[str, pd.DataFrame]:
        return {"bhs": self.bhs, "btp": self.btp, "bwd": self.bwd, "bds": self.bds}
