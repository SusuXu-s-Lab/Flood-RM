"""Compatibility skin for ONM/DNMG readiness gates.

NOTEBOOK_API, VALIDATION_QA: notebooks still import ``power.audit.onm_readiness``
while the clean implementation lives in ``power_v2.readiness``.
"""

from power_v2.readiness import build_onm_readiness_report
from power_v2.readiness import summarize_der_export_readiness
from power_v2.readiness import summarize_dynagrid_smoke_readiness
from power_v2.readiness import summarize_event_bundle_readiness
from power_v2.readiness import summarize_opendss_solve_readiness
from power_v2.readiness import summarize_powermodels_onm_smoke_readiness

__all__ = [
    "build_onm_readiness_report",
    "summarize_der_export_readiness",
    "summarize_dynagrid_smoke_readiness",
    "summarize_event_bundle_readiness",
    "summarize_opendss_solve_readiness",
    "summarize_powermodels_onm_smoke_readiness",
]
