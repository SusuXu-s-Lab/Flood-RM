from __future__ import annotations

from pathlib import Path

import pandas as pd

from paths import resolve_location_path
from sfincs_runs.forcing import select_inland_scenario_rows
from coupling.dynamic_handoff import dynamic_handoff_paths, require_handoff
from wflow_runs.streamflow_realization import wflow_streamflow_gage_overlap


def handoff_readiness(
    config,
    location_root,
    *,
    catalog_path=None,
    event_ids=None,
    limit=None,
) -> pd.DataFrame:
    """Return event-level readiness for dynamic Wflow-to-SFINCS handoff staging."""
    location_root = Path(location_root)
    if event_ids is None:
        catalog_path = resolve_location_path(
            location_root,
            catalog_path or "data/event_catalog/catalog/scenario_catalog.csv",
        )
        catalog = pd.read_csv(catalog_path)
        if "event_id" not in catalog:
            raise ValueError(f"Event Catalog is missing event_id: {catalog_path}")
        catalog["event_id"] = catalog["event_id"].astype(str)
        event_ids = select_inland_scenario_rows(catalog, limit=limit)["event_id"].tolist()
    elif limit is not None:
        event_ids = list(event_ids)[: int(limit)]

    rows = []
    for event_id in event_ids:
        event_id = str(event_id)
        paths = dynamic_handoff_paths(config, location_root, event_id)
        try:
            accepted = require_handoff(config, location_root, event_id, catalog_path=catalog_path)
            rows.append(
                {
                    "event_id": event_id,
                    "status": "accepted",
                    "sfincs_discharge_forcing": accepted["sfincs_discharge_forcing"],
                    "acceptance": accepted["dynamic_handoff_acceptance"],
                    "issue": "",
                }
            )
        except Exception as exc:
            status = "blocked"
            issue = str(exc)
            compatibility = {}
            try:
                compatibility = wflow_streamflow_gage_overlap(
                    config,
                    location_root,
                    event_id,
                    catalog_path=catalog_path,
                )
                if not compatibility.get("compatible", False):
                    status = "incompatible"
                    issue = str(compatibility.get("message", issue))
            except Exception:
                compatibility = {}
            rows.append(
                {
                    "event_id": event_id,
                    "status": status,
                    "sfincs_discharge_forcing": str(paths["discharge"]),
                    "acceptance": str(paths["acceptance"]),
                    "issue": issue,
                    "streamflow_member_id": compatibility.get("member_id", ""),
                    "streamflow_member_sites": ",".join(compatibility.get("member_sites", [])),
                    "reviewed_gage_overlap": ",".join(compatibility.get("overlap_site_nos", [])),
                    "reviewed_gage_count": compatibility.get("reviewed_site_count", ""),
                }
            )
    return pd.DataFrame(rows)


def accepted_dynamic_handoff_event_ids(readiness: pd.DataFrame) -> list[str]:
    """Return accepted dynamic handoff event IDs from a readiness table."""
    if readiness.empty:
        return []
    return readiness.loc[readiness["status"].eq("accepted"), "event_id"].astype(str).tolist()


def missing_dynamic_wflow_acceptance(config: dict, location_root: Path, event_ids) -> list[str]:
    source = str(((config.get("inland_coupling", {}) or {}).get("discharge_forcing", {}) or {}).get("source", "")).lower()
    if source != "wflow_dynamic":
        return []
    missing = []
    for event_id in event_ids:
        try:
            require_handoff(config, location_root, str(event_id))
        except Exception as exc:
            paths = dynamic_handoff_paths(config, location_root, str(event_id))
            missing.append(f"{event_id}: {paths['acceptance']} ({exc})")
    return missing
