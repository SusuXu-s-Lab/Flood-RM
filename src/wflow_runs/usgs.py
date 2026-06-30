from __future__ import annotations


def usgs_streamflow_records_config(value) -> dict:
    """Return streamflow_records settings from compact or expanded config."""
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    return {"output": value}


def usgs_instantaneous_streamflow_spec(config: dict) -> dict:
    """Build a USGS streamgage spec for instantaneous discharge records."""
    spec = dict(((config.get("collection", {}) or {}).get("usgs_streamgages", {}) or {}))
    records_cfg = usgs_streamflow_records_config(spec.get("streamflow_records"))
    records_cfg["service"] = "iv"
    records_cfg.pop("stat_cd", None)
    spec["streamflow_records"] = records_cfg
    return spec
