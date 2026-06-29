from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class Probability:
    r"""Probability metadata for one stochastic event :math:`\omega`."""

    p_event: float | None = None
    aep: float | None = None
    return_period_years: float | None = None
    weight: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


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
    probability: Probability = field(default_factory=Probability)
    q_target_cms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ["reference_time", "window_start", "window_end"]:
            payload[key] = pd.Timestamp(payload[key]).isoformat()
        for key in ["precip_path", "temp_pet_path", "rainfall_member_file"]:
            if payload.get(key) is not None:
                payload[key] = str(payload[key])
        payload["probability"] = self.probability.to_dict()
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
