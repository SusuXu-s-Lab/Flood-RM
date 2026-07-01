from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DesignEvent:
    r"""One stochastic Wflow boundary event.

    The hydrologic contract is

    .. math::

       Q^W_\omega(h,t) = \mathcal{W}_\theta(P_\omega,T_\omega,PET_\omega,S_{0,\omega})_h.
    """

    event_id: str
    reference_time: pd.Timestamp
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    precip_path: Path
    temp_pet_path: Path
    rainfall_member_id: str | None = None
    rainfall_member_file: Path | None = None
    rainfall_scale_factor: float = 1.0
    probability: dict[str, Any] = field(default_factory=dict)
    q_target_cms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ["reference_time", "window_start", "window_end"]:
            payload[key] = pd.Timestamp(payload[key]).isoformat()
        for key in ["precip_path", "temp_pet_path", "rainfall_member_file"]:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        payload["probability"] = dict(self.probability)
        return payload


@dataclass(frozen=True)
class HandoffPoint:
    """Reviewed SFINCS/Wflow handoff point."""

    id: str
    x: float
    y: float
    weight: float = 1.0
    sfincs_domain_id: str | None = None
    wflow_submodel_id: str | None = None


@dataclass(frozen=True)
class BoundaryRun:
    """Artifacts from one event boundary replay."""

    event: DesignEvent
    discharge_nc: Path
    acceptance_json: Path
    qa_csv: Path
    status: str
    amplification: dict[str, Any]
    report: pd.DataFrame

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "event_id": self.event.event_id,
                "status": self.status,
                "sfincs_discharge": str(self.discharge_nc),
                "acceptance_json": str(self.acceptance_json),
                "qa_csv": str(self.qa_csv),
                "K": self.amplification.get("K", 1.0),
            },
            name="wflow_event_boundary",
        )


@dataclass(frozen=True)
class WflowBuildPlan:
    study_location: str
    plugin: str
    base_model_root: Path
    events_root: Path
    data_catalog: Path
    build_config: Path
    update_forcing_config: Path
    build_steps: tuple[str, ...]
    update_steps: tuple[str, ...]
    region_kind: str
    review_required: bool
    domain_status: str
    build_command: str
    update_command: str


@dataclass(frozen=True)
class WflowDomainSetPlan:
    reviewed_network: Path
    status: str
    gage_count: int
    submodel_count: int
    handoff_count: int
    submodels: tuple[dict, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class WflowSourceStrategy:
    status: str
    hydrography_policy: str
    hydromt_basemap_source: str
    river_geometry_source: str | None
    catchment_source: str | None
    hydrography_api: str
    soil_policy: str
    wflow_soil_parameter_source: str
    ssurgo_inputs: tuple[str, ...]
    global_fallbacks: tuple[str, ...]
    issues: tuple[str, ...]
