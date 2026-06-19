from __future__ import annotations

from io import StringIO
import json
from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
import requests

from design_events.utils import read_source_artifact, source_artifact_covers, write_source_artifact


USGS_SITE_SERVICE_URL = "https://waterservices.usgs.gov/nwis/site/"
USGS_DAILY_VALUES_URL = "https://waterservices.usgs.gov/nwis/dv/"
USGS_INSTANTANEOUS_VALUES_URL = "https://waterservices.usgs.gov/nwis/iv/"
REVIEW_SCHEMA = [
    "site_no",
    "site_name",
    "status",
    "drainage_area_sqmi",
    "period_start",
    "period_end",
    "record_years",
    "completeness_score",
    "roles",
    "frequency_basis",
    "wflow_submodel_id",
    "sfincs_domain_id",
    "sfincs_handoff_id",
    "review_status",
    "review_notes",
]
DEFAULT_HUC_REGION_REVIEW = {
    "method": "huc_region",
    "source_geometry": "data/static/aoi/wflow_nhdplus_watersheds.geojson",
    "geometry_predicate": "covers",
    "review_status": "accepted_with_warning",
    "roles": ["frequency", "calibration", "validation"],
    "frequency_basis_from": "wflow_submodel_id",
    "wflow_submodel_id_from": "wflow_submodel_id",
    "sfincs_domain_id_from": "sfincs_domain_id",
}


def active_streamgage_candidate_artifact_ready(config, paths) -> bool:
    settings = _settings(config, paths)
    spec = settings["usgs_streamgages"]
    output_path = _candidate_output_path(spec, paths)
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    return (
        output_path.exists()
        and source_artifact_covers(paths, "usgs_streamgages", "active_candidates", start, end)
        and _active_candidate_artifact_matches_discovery(paths, spec)
    )


def discover_active_streamgage_candidates(
    config,
    paths,
    *,
    site_records=None,
    skip_existing=False,
):
    """Discover active USGS discharge streamgage candidates and write GeoJSON."""
    settings = _settings(config, paths)
    output_path = _candidate_output_path(settings["usgs_streamgages"], paths)
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    if (
        skip_existing
        and active_streamgage_candidate_artifact_ready(config, paths)
    ):
        candidate_count = len(json.loads(output_path.read_text(encoding="utf-8")).get("features", []))
        return {
            "reused": True,
            "candidate_count": candidate_count,
            "candidate_geojson": output_path,
        }

    spec = settings["usgs_streamgages"]
    discovery_signature = _safe_discovery_signature(spec, paths)
    records = site_records if site_records is not None else fetch_nwis_streamgage_site_records(spec, paths)
    frame = _candidate_frame(records, spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_to_geojson(frame), encoding="utf-8")
    artifact_json = write_source_artifact(
        paths,
        source="usgs_streamgages",
        kind="active_candidates",
        start=start,
        end=end,
        artifacts={"candidate_geojson": output_path},
        metadata={
            "candidate_count": int(len(frame)),
            "parameter_cd": _parameter_cd(spec),
            "site_status": _site_status(spec),
            "active_records_only": bool(spec.get("active_records_only", True)),
            "discovery_signature": discovery_signature,
        },
    )
    return {
        "reused": False,
        "candidate_count": int(len(frame)),
        "candidate_geojson": output_path,
        "source_artifact_json": artifact_json,
    }


def collect_usgs_streamgages(settings, skip_existing=False, smoke=False):
    result = discover_active_streamgage_candidates(
        settings["config"],
        settings["paths"],
        skip_existing=skip_existing,
    )
    spec = settings["config"].get("collection", {}).get("usgs_streamgages", {})
    if spec.get("streamflow_records", {}).get("collect", False):
        result["streamflow_records"] = collect_usgs_streamflow_records(
            settings["config"],
            settings["paths"],
            skip_existing=skip_existing,
        )
    result["smoke"] = bool(smoke)
    return result


def collect_usgs_streamflow_records(
    config,
    paths,
    *,
    response_text_by_site=None,
    reviewed_network_records=None,
    skip_existing=False,
):
    settings = _settings(config, paths)
    spec = settings["usgs_streamgages"]
    output_path = _streamflow_records_output_path(spec, paths)
    start = pd.Timestamp(settings["start"])
    end = pd.Timestamp(settings["end"])
    gages = (
        _reviewed_network_frame(reviewed_network_records, spec, paths)
        if reviewed_network_records is not None
        else _read_reviewed_network(spec, paths)
    )
    if gages.empty:
        raise ValueError("reviewed streamgage network contains no accepted active gages")
    if (
        skip_existing
        and output_path.exists()
        and source_artifact_covers(paths, "usgs_streamgages", "streamflow_records", start, end)
        and _streamflow_record_artifact_matches_gages(paths, output_path, gages)
    ):
        record_count = len(pd.read_csv(output_path))
        return {
            "reused": True,
            "record_count": int(record_count),
            "site_count": int(gages["site_no"].astype(str).nunique()),
            "streamflow_records_csv": output_path,
        }

    records = []
    cached_sites = set()
    if (
        skip_existing
        and output_path.exists()
        and source_artifact_covers(paths, "usgs_streamgages", "streamflow_records", start, end)
    ):
        try:
            cached = pd.read_csv(output_path, dtype={"site_no": str})
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
            cached = pd.DataFrame()
        cached_columns = ["site_no", "time", "discharge_cfs", "source"]
        if set(cached_columns).issubset(cached.columns):
            expected_sites = set(gages["site_no"].astype(str))
            cached = cached[cached["site_no"].astype(str).isin(expected_sites)].copy()
            cached["site_no"] = cached["site_no"].astype(str)
            cached_sites = set(cached["site_no"])
            records.extend(cached[cached_columns].to_dict("records"))

    for site_no in gages["site_no"].astype(str):
        if site_no in cached_sites:
            continue
        if response_text_by_site is not None:
            site_records = _parse_discharge_records(response_text_by_site.get(site_no, ""), _streamflow_service(spec))
        else:
            site_records = fetch_nwis_discharge_records(spec, site_no, start, end)
        records.extend(site_records)
    frame = pd.DataFrame(records, columns=["site_no", "time", "discharge_cfs", "source"])
    frame["site_no"] = frame["site_no"].astype(str)
    frame = frame.sort_values(["site_no", "time"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    artifact_json = write_source_artifact(
        paths,
        source="usgs_streamgages",
        kind="streamflow_records",
        start=start,
        end=end,
        artifacts={"streamflow_records_csv": output_path},
        metadata={
            "site_count": int(len(gages)),
            "record_count": int(len(frame)),
            "service": _streamflow_service(spec),
            "parameter_cd": _parameter_cd(spec),
            "reviewed_network": str(_reviewed_network_path(spec, paths)),
        },
    )
    return {
        "reused": False,
        "site_count": int(len(gages)),
        "record_count": int(len(frame)),
        "streamflow_records_csv": output_path,
        "source_artifact_json": artifact_json,
    }


def _streamflow_record_artifact_matches_gages(paths, output_path, gages):
    expected_sites = set(gages["site_no"].astype(str))
    manifest = read_source_artifact(paths, "usgs_streamgages", "streamflow_records") or {}
    manifest_site_count = manifest.get("metadata", {}).get("site_count")
    if manifest_site_count is not None and int(manifest_site_count) != len(expected_sites):
        return False
    try:
        records = pd.read_csv(output_path, dtype={"site_no": str}, usecols=["site_no"])
    except (FileNotFoundError, ValueError):
        return False
    return expected_sites.issubset(set(records["site_no"].astype(str)))


def _active_candidate_artifact_matches_discovery(paths, spec):
    manifest = read_source_artifact(paths, "usgs_streamgages", "active_candidates") or {}
    expected = _safe_discovery_signature(spec, paths)
    if expected is None:
        return False
    return manifest.get("metadata", {}).get("discovery_signature") == expected


def _discovery_signature(spec, paths):
    params = _nwis_site_query_params(spec, paths)
    return {key: params[key] for key in sorted(params)}


def _safe_discovery_signature(spec, paths):
    try:
        return _discovery_signature(spec, paths)
    except ValueError:
        return None


def write_reviewed_streamgage_network(
    config,
    paths,
    review_records,
    *,
    candidate_records=None,
    output_path=None,
):
    """Write the reviewed USGS streamgage-network artifact from candidate gages."""
    settings = _settings(config, paths)
    spec = settings["usgs_streamgages"]
    network_path = Path(output_path) if output_path is not None else _reviewed_network_path(spec, paths)
    candidate_features = (
        _features_from_records(candidate_records)
        if candidate_records is not None
        else _read_candidate_features(_candidate_output_path(spec, paths))
    )
    reviews = _review_decisions(review_records)
    selected_features = []
    for feature in candidate_features:
        properties = dict(feature.get("properties", {}))
        site_no = str(properties.get("site_no", ""))
        if site_no not in reviews:
            continue
        decision = reviews[site_no]
        reviewed = {**properties, **decision}
        _validate_reviewed_streamgage(site_no, reviewed)
        selected_features.append(
            {
                "type": "Feature",
                "properties": reviewed,
                "geometry": feature.get("geometry"),
            }
        )
    missing = sorted(set(reviews) - {str(feature.get("properties", {}).get("site_no", "")) for feature in candidate_features})
    if missing:
        raise ValueError("review records reference unknown candidate site_no values: " + ", ".join(missing))
    network_path.parent.mkdir(parents=True, exist_ok=True)
    network_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": selected_features}, indent=2) + "\n",
        encoding="utf-8",
    )
    accepted = [
        feature
        for feature in selected_features
        if str(feature["properties"].get("review_status", "")).lower() in {"accepted", "accepted_with_warning"}
    ]
    return {
        "reviewed_network_geojson": network_path,
        "reviewed_count": len(selected_features),
        "accepted_count": len(accepted),
    }


def build_reviewed_streamgage_decisions(config, paths, *, candidate_records=None):
    """Build reviewed streamgage decisions from candidates and configured review method."""
    settings = _settings(config, paths)
    spec = settings["usgs_streamgages"]
    candidate_features = (
        _features_from_records(candidate_records)
        if candidate_records is not None
        else _read_candidate_features(_candidate_output_path(spec, paths))
    )
    if _review_method(spec) == "huc_region":
        decisions = _huc_region_review_decisions(candidate_features, spec, paths)
    else:
        policy = spec.get("review_policy") or {}
        decisions = []
        for feature in candidate_features:
            properties = dict(feature.get("properties", {}))
            if str(properties.get("status", "")).lower() != "active":
                continue
            decisions.append(_streamgage_review_decision(properties, policy))
    if not decisions:
        raise ValueError("no active streamgage candidates available for review decisions")
    return decisions


def _review_method(spec):
    if spec.get("review"):
        return str(spec["review"].get("method", "huc_region")).strip().lower()
    if spec.get("review_policy"):
        return "review_policy"
    return str(DEFAULT_HUC_REGION_REVIEW["method"])


def streamgage_review_config(spec):
    return dict(_review_config(spec))


def _review_config(spec):
    review = dict(DEFAULT_HUC_REGION_REVIEW)
    review.update(spec.get("review") or {})
    return review


def _huc_region_review_decisions(candidate_features, spec, paths):
    review = _review_config(spec)
    hucs = _review_huc_regions(review, spec, paths)
    candidates = _candidate_features_frame(candidate_features)
    if candidates.empty:
        return []
    if "status" in candidates:
        candidates = candidates[candidates["status"].fillna("").astype(str).str.lower() == "active"].copy()
    if candidates.empty:
        return []

    decisions = []
    predicate = str(review.get("geometry_predicate", "covers")).strip().lower()
    roles = list(review.get("roles", ["frequency", "calibration", "validation"]))
    for _, candidate in candidates.iterrows():
        huc = _matching_huc_region(hucs, candidate.geometry, predicate)
        if huc is None:
            continue
        wflow_submodel_id = _huc_review_value(
            huc,
            review.get("wflow_submodel_id_from", "wflow_submodel_id"),
            fallback=_huc_review_value(huc, "huc_id", fallback="huc_region"),
        )
        frequency_basis = _huc_review_value(
            huc,
            review.get("frequency_basis_from", "wflow_submodel_id"),
            fallback=wflow_submodel_id,
        )
        sfincs_domain_id = _huc_review_value(
            huc,
            review.get("sfincs_domain_id_from", "sfincs_domain_id"),
            fallback=wflow_submodel_id,
        )
        huc_id = _huc_review_value(huc, "huc_id", fallback=wflow_submodel_id)
        decisions.append(
            {
                "site_no": str(candidate["site_no"]),
                "review_status": str(review.get("review_status", "accepted_with_warning")),
                "roles": roles,
                "frequency_basis": frequency_basis,
                "wflow_submodel_id": wflow_submodel_id,
                "sfincs_domain_id": sfincs_domain_id,
                "sfincs_handoff_id": None,
                "review_notes": _huc_review_notes(candidate, huc_id, wflow_submodel_id, roles, review),
            }
        )
    return decisions


def _review_huc_regions(review, spec, paths):
    value = review.get("source_geometry") or spec.get("discovery", {}).get("search_geometry")
    if not value:
        raise ValueError("huc_region streamgage review requires review.source_geometry or discovery.search_geometry")
    path = _location_path(paths, value)
    if not path.exists():
        raise FileNotFoundError(path)
    hucs = gpd.read_file(path)
    if hucs.empty:
        raise ValueError(f"huc_region streamgage review source geometry is empty: {path}")
    if hucs.crs is None:
        hucs = hucs.set_crs("EPSG:4326")
    return hucs.to_crs("EPSG:4326")


def _candidate_features_frame(candidate_features):
    if not candidate_features:
        return gpd.GeoDataFrame(columns=[*REVIEW_SCHEMA, "geometry"], geometry="geometry", crs="EPSG:4326")
    frame = gpd.GeoDataFrame.from_features(candidate_features, crs="EPSG:4326")
    if frame.empty:
        return frame
    if "site_no" not in frame:
        raise ValueError("USGS streamgage candidates are missing site_no")
    frame["site_no"] = frame["site_no"].astype(str)
    return frame[frame.geometry.notna() & ~frame.geometry.is_empty].copy()


def _matching_huc_region(hucs, geometry, predicate):
    if predicate == "within":
        mask = hucs.geometry.apply(lambda huc: geometry.within(huc))
    elif predicate == "intersects":
        mask = hucs.geometry.intersects(geometry)
    else:
        mask = hucs.geometry.apply(lambda huc: huc.covers(geometry))
    matches = hucs.loc[mask]
    if matches.empty:
        return None
    return matches.sort_values(by=[column for column in ("huc_level", "huc_id") if column in matches.columns]).iloc[0]


def _huc_review_value(huc, column, *, fallback):
    if column in huc:
        value = huc[column]
        if not _is_missing_review_value(value):
            return str(value)
    return str(fallback)


def _huc_review_notes(candidate, huc_id, wflow_submodel_id, roles, review):
    template = review.get(
        "review_note_template",
        "HUC-derived active discharge gage inside {huc_id}; roles={roles}; submodel={submodel}; record_years={record_years}.",
    )
    years = candidate.get("record_years")
    return str(template).format(
        site_no=str(candidate["site_no"]),
        huc_id=huc_id,
        roles=", ".join(roles),
        submodel=wflow_submodel_id,
        record_years="" if pd.isna(years) else years,
    )


def _streamgage_review_decision(properties, policy):
    site_no = str(properties.get("site_no", ""))
    site_name = str(properties.get("site_name", "")).upper()
    basin = _review_basin_assignment(site_no, site_name, policy)
    handoff_id = _review_handoff_id(site_no, policy)
    roles = _review_roles(properties, site_name, handoff_id, policy)
    status = str(policy.get("default_review_status", "accepted_with_warning"))
    sfincs_domain_id = policy.get("default_sfincs_domain_id", "greensboro_main")
    override = _site_policy_override(site_no, policy)
    if override:
        status = str(override.get("review_status", status))
        roles = list(override.get("roles", roles))
        if "sfincs_handoff_id" in override:
            handoff_id = override.get("sfincs_handoff_id")
        if "sfincs_domain_id" in override:
            sfincs_domain_id = override.get("sfincs_domain_id")
        basin = {**basin, **{key: override[key] for key in ("frequency_basis", "wflow_submodel_id") if key in override}}
    return {
        "site_no": site_no,
        "review_status": status,
        "roles": roles,
        "frequency_basis": basin["frequency_basis"],
        "wflow_submodel_id": basin["wflow_submodel_id"],
        "sfincs_domain_id": sfincs_domain_id,
        "sfincs_handoff_id": handoff_id,
        "review_notes": _review_notes(properties, roles, basin, handoff_id, policy),
    }


def _review_basin_assignment(site_no, site_name, policy):
    for rule in policy.get("basin_rules", []):
        sites = {str(value) for value in rule.get("site_nos", [])}
        pattern = str(rule.get("contains", "")).upper()
        if site_no in sites or (pattern and pattern in site_name):
            return {
                "frequency_basis": str(rule.get("frequency_basis", rule.get("wflow_submodel_id", "review_required"))),
                "wflow_submodel_id": str(rule.get("wflow_submodel_id", rule.get("frequency_basis", "review_required"))),
            }
    fallback = str(policy.get("fallback_wflow_submodel_id", "review_required"))
    return {"frequency_basis": fallback, "wflow_submodel_id": fallback}


def _review_handoff_id(site_no, policy):
    handoffs = policy.get("handoff_site_nos", {})
    if isinstance(handoffs, dict):
        return handoffs.get(site_no)
    if site_no in {str(value) for value in handoffs or []}:
        return f"handoff_{site_no}"
    return None


def _review_roles(properties, site_name, handoff_id, policy):
    if handoff_id:
        return list(policy.get("handoff_roles", ["frequency", "calibration", "validation", "sfincs_handoff"]))
    validation_only_patterns = [str(value).upper() for value in policy.get("validation_only_patterns", [])]
    if any(pattern in site_name for pattern in validation_only_patterns):
        return list(policy.get("validation_only_roles", ["validation"]))
    record_years = pd.to_numeric(pd.Series([properties.get("record_years")]), errors="coerce").iloc[0]
    long_record_years = float(policy.get("long_record_years", 50.0))
    if pd.notna(record_years) and float(record_years) >= long_record_years:
        return list(policy.get("long_record_roles", ["frequency", "validation"]))
    return list(policy.get("default_roles", ["calibration", "validation"]))


def _site_policy_override(site_no, policy):
    overrides = policy.get("site_overrides", {})
    if isinstance(overrides, dict):
        return dict(overrides.get(site_no, {}))
    for override in overrides or []:
        if str(override.get("site_no", "")) == site_no:
            return dict(override)
    return {}


def _review_notes(properties, roles, basin, handoff_id, policy):
    site_no = str(properties.get("site_no", ""))
    years = properties.get("record_years")
    role_text = ", ".join(roles)
    target = "Wflow-SFINCS handoff outlet" if handoff_id else "hydrologic evidence gage"
    template = policy.get(
        "review_note_template",
        "Policy-derived {target}; roles={roles}; submodel={submodel}; record_years={record_years}. Confirm basin assignment before production.",
    )
    return str(template).format(
        site_no=site_no,
        target=target,
        roles=role_text,
        submodel=basin["wflow_submodel_id"],
        frequency_basis=basin["frequency_basis"],
        record_years="" if pd.isna(years) else years,
    )


def fetch_nwis_streamgage_site_records(spec, paths):
    discovery = spec.get("discovery", {})
    params = _nwis_site_query_params(spec, paths)
    if discovery.get("series_catalog_output", True):
        series_params = {
            **params,
            "siteOutput": str(discovery.get("series_site_output", "Basic")),
            "seriesCatalogOutput": "true",
        }
        series_response = requests.get(
            discovery.get("url", USGS_SITE_SERVICE_URL),
            params=series_params,
            timeout=int(discovery.get("request_timeout_seconds", 60)),
        )
        series_response.raise_for_status()
        expanded_params = {
            **params,
            "siteOutput": str(discovery.get("site_output", "Expanded")),
        }
        expanded_response = requests.get(
            discovery.get("url", USGS_SITE_SERVICE_URL),
            params=expanded_params,
            timeout=int(discovery.get("request_timeout_seconds", 60)),
        )
        expanded_response.raise_for_status()
        return _merge_site_metadata(_read_rdb(series_response.text), _read_rdb(expanded_response.text))

    response = requests.get(
        discovery.get("url", USGS_SITE_SERVICE_URL),
        params={**params, "siteOutput": str(discovery.get("site_output", "Expanded"))},
        timeout=int(discovery.get("request_timeout_seconds", 60)),
    )
    response.raise_for_status()
    return _read_rdb(response.text)


def _read_candidate_features(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("features", []))


def _features_from_records(records):
    features = []
    for record in records:
        row = dict(record)
        longitude = row.pop("longitude", row.pop("lon", None))
        latitude = row.pop("latitude", row.pop("lat", None))
        features.append(
            {
                "type": "Feature",
                "properties": row,
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            }
        )
    return features


def _review_decisions(review_records):
    frame = pd.DataFrame(list(review_records or []))
    if frame.empty:
        raise ValueError("at least one streamgage review record is required")
    if "site_no" not in frame:
        raise ValueError("streamgage review records require site_no")
    frame["site_no"] = frame["site_no"].astype(str)
    if frame["site_no"].duplicated().any():
        duplicates = sorted(frame.loc[frame["site_no"].duplicated(), "site_no"].unique())
        raise ValueError("duplicate streamgage review records: " + ", ".join(duplicates))
    return {
        str(row["site_no"]): {
            key: _review_value(value)
            for key, value in row.items()
            if key != "site_no" and not _is_missing_review_value(value)
        }
        for row in frame.to_dict("records")
    }


def _review_value(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed
    return value


def _is_missing_review_value(value):
    if isinstance(value, (list, tuple, dict)):
        return False
    return pd.isna(value)


def _validate_reviewed_streamgage(site_no, properties):
    status = str(properties.get("review_status", "")).lower()
    allowed = {"accepted", "accepted_with_warning", "rejected"}
    if status not in allowed:
        raise ValueError(f"review_status for {site_no} must be one of {sorted(allowed)}")
    if status in {"accepted", "accepted_with_warning"} and not properties.get("roles"):
        raise ValueError(f"accepted streamgage {site_no} requires at least one role")


def _nwis_site_query_params(spec, paths):
    discovery = spec.get("discovery", {})
    params = {
        "format": "rdb",
        "parameterCd": _parameter_cd(spec),
        "siteStatus": _site_status(spec),
    }
    data_types = discovery.get("data_types") or discovery.get("has_data_type_cd")
    if data_types:
        params["hasDataTypeCd"] = ",".join(data_types) if isinstance(data_types, list) else str(data_types)
    bbox = discovery.get("bbox") or _bbox_from_search_geometry(
        discovery.get("search_geometry"),
        paths,
        buffer_km=discovery.get("hydrologic_buffer_km"),
    )
    if bbox:
        params["bBox"] = ",".join(str(value) for value in bbox)
    elif not any(discovery.get(key) for key in ("sites", "state_cd", "stateCd", "huc", "county_cd", "countyCd")):
        raise ValueError(
            "USGS streamgage discovery requires a bounded search geometry, bbox, sites, state, huc, or county filter"
        )
    _add_optional_major_filter(params, discovery)
    return params


def _merge_site_metadata(series_records, expanded_records):
    expanded_by_site = {
        str(record.get("site_no")): record
        for record in expanded_records
        if record.get("site_no") is not None
    }
    merged = []
    for record in series_records:
        site_metadata = expanded_by_site.get(str(record.get("site_no")), {})
        row = dict(record)
        canonical_keys = {_canonical_column(key) for key in row}
        for key, value in site_metadata.items():
            if key not in row and _canonical_column(key) in canonical_keys:
                continue
            if key not in row or row.get(key) in (None, ""):
                row[key] = value
                canonical_keys.add(_canonical_column(key))
        merged.append(row)
    return merged


def fetch_nwis_discharge_records(spec, site_no, start, end):
    records_cfg = spec.get("streamflow_records", {})
    service = _streamflow_service(spec)
    url = records_cfg.get("url") or (USGS_INSTANTANEOUS_VALUES_URL if service == "iv" else USGS_DAILY_VALUES_URL)
    params = {
        "format": "rdb",
        "sites": str(site_no),
        "parameterCd": _parameter_cd(spec),
        "startDT": pd.Timestamp(start).date().isoformat(),
        "endDT": pd.Timestamp(end).date().isoformat(),
        "siteStatus": _site_status(spec),
    }
    if service == "dv":
        params["statCd"] = str(records_cfg.get("stat_cd", "00003"))
    response = requests.get(
        url,
        params=params,
        timeout=int(records_cfg.get("request_timeout_seconds", spec.get("discovery", {}).get("request_timeout_seconds", 60))),
    )
    response.raise_for_status()
    return _parse_discharge_records(response.text, service)


def _add_optional_major_filter(params, discovery):
    aliases = {
        "sites": "sites",
        "state_cd": "stateCd",
        "stateCd": "stateCd",
        "huc": "huc",
        "county_cd": "countyCd",
        "countyCd": "countyCd",
    }
    for source, target in aliases.items():
        if discovery.get(source):
            value = discovery[source]
            params[target] = ",".join(value) if isinstance(value, list) else str(value)


def _settings(config, paths):
    collection = config.get("collection", {})
    spec = collection.get("usgs_streamgages", {})
    return {
        "config": config,
        "paths": paths,
        "start": collection.get("start", "1979-01-01"),
        "end": collection.get("end", "2022-12-31"),
        "usgs_streamgages": spec,
    }


def _parameter_cd(spec):
    return str(spec.get("discovery", {}).get("parameter_cd", "00060"))


def _site_status(spec):
    return str(spec.get("discovery", {}).get("site_status", "active"))


def _streamflow_service(spec):
    return str(spec.get("streamflow_records", {}).get("service", "dv")).strip().lower()


def _candidate_output_path(spec, paths):
    value = spec.get("candidate_output") or paths.get("usgs_streamgage_candidates_geojson")
    if value is None:
        value = Path("data/sources/usgs_streamgages/streamgage_candidates.geojson")
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    return Path(root) / path


def _location_path(paths, value):
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    if path.parts and path.parts[0] in {"data", "02_flood", "01_grid"}:
        return Path(root) / path
    return Path(paths.get("repo_root", root)) / path


def _streamflow_records_output_path(spec, paths):
    value = spec.get("streamflow_records", {}).get("output") or paths.get("usgs_streamflow_records_csv")
    if value is None:
        value = Path("data/sources/usgs_streamgages/streamflow_records.csv")
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    return Path(root) / path


def _reviewed_network_path(spec, paths):
    value = spec.get("reviewed_network") or paths.get("usgs_streamgage_network_geojson")
    if value is None:
        value = Path("data/sources/usgs_streamgages/streamgage_network.geojson")
    path = Path(value)
    if path.is_absolute():
        return path
    root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
    return Path(root) / path


def _candidate_frame(records, spec):
    raw = pd.DataFrame(list(records or []))
    if raw.empty:
        return _empty_candidate_frame()
    raw = raw.rename(columns={name: _canonical_column(name) for name in raw.columns})
    raw["site_no"] = raw["site_no"].astype(str)
    raw = _filter_discharge_records(raw, spec)
    if raw.empty:
        return _empty_candidate_frame()
    rows = []
    for _, group in raw.groupby("site_no", sort=True):
        rows.append(_candidate_row(group))
    return pd.DataFrame(rows, columns=[*REVIEW_SCHEMA, "longitude", "latitude"])


def _read_reviewed_network(spec, paths):
    path = _reviewed_network_path(spec, paths)
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    rows = []
    for feature in features:
        properties = dict(feature.get("properties", {}))
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        properties["longitude"] = coordinates[0] if len(coordinates) > 0 else None
        properties["latitude"] = coordinates[1] if len(coordinates) > 1 else None
        rows.append(properties)
    return _reviewed_network_frame(rows, spec, paths)


def _reviewed_network_frame(records, spec, paths):
    frame = pd.DataFrame(list(records or []))
    if frame.empty:
        return pd.DataFrame(columns=[*REVIEW_SCHEMA, "longitude", "latitude"])
    frame = frame.rename(columns={name: _canonical_column(name) for name in frame.columns})
    if "site_no" not in frame:
        raise ValueError("reviewed streamgage network is missing site_no")
    frame["site_no"] = frame["site_no"].astype(str)
    if spec.get("active_records_only", True) or spec.get("exclude_inactive_gages", True):
        if "status" in frame:
            frame = frame[frame["status"].fillna("").astype(str).str.lower() == "active"]
    if "review_status" in frame:
        frame = frame[frame["review_status"].fillna("").astype(str).str.lower().isin({"accepted", "accepted_with_warning"})]
    elif not spec.get("accept_unreviewed_streamgage_network", False):
        raise ValueError("reviewed streamgage network is missing review_status")
    return frame.reset_index(drop=True)


def _empty_candidate_frame():
    return pd.DataFrame(columns=[*REVIEW_SCHEMA, "longitude", "latitude"])


def _canonical_column(name):
    aliases = {
        "station_nm": "site_name",
        "site_nm": "site_name",
        "dec_lat_va": "latitude",
        "lat_va": "latitude",
        "dec_long_va": "longitude",
        "long_va": "longitude",
        "drain_area_va": "drainage_area_sqmi",
        "drain_sqkm_va": "drainage_area_sqkm",
        "parm_cd": "parameter_cd",
        "parameterCd": "parameter_cd",
        "siteStatus": "status",
        "site_status": "status",
        "begin_date": "period_start",
        "end_date": "period_end",
    }
    return aliases.get(str(name), str(name))


def _filter_discharge_records(frame, spec):
    parameter = _parameter_cd(spec)
    if "parameter_cd" in frame.columns:
        frame = frame[frame["parameter_cd"].astype(str) == parameter]
    if spec.get("active_records_only", True) or spec.get("exclude_inactive_gages", True):
        if "status" in frame.columns:
            frame = frame[frame["status"].fillna("").astype(str).str.lower() == "active"]
    required = {"site_no", "site_name", "longitude", "latitude"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("USGS streamgage records are missing required columns: " + ", ".join(sorted(missing)))
    return frame.dropna(subset=["longitude", "latitude"])


def _candidate_row(group):
    first = group.iloc[0]
    period_start = _min_date(group.get("period_start"))
    period_end = _max_date(group.get("period_end"))
    record_years = _record_years(period_start, period_end)
    return {
        "site_no": str(first["site_no"]),
        "site_name": str(first["site_name"]),
        "status": str(first.get("status", "active")).lower(),
        "drainage_area_sqmi": _number(first.get("drainage_area_sqmi")),
        "period_start": period_start,
        "period_end": period_end,
        "record_years": record_years,
        "completeness_score": _completeness_score(group, period_start, period_end),
        "roles": [],
        "frequency_basis": None,
        "wflow_submodel_id": None,
        "sfincs_domain_id": None,
        "sfincs_handoff_id": None,
        "review_status": "candidate",
        "review_notes": "",
        "longitude": _number(first["longitude"]),
        "latitude": _number(first["latitude"]),
    }


def _min_date(values):
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.min().date().isoformat()


def _max_date(values):
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date().isoformat()


def _record_years(period_start, period_end):
    if not period_start or not period_end:
        return None
    days = (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days
    return round(max(days, 0) / 365.25, 2)


def _completeness_score(group, period_start, period_end):
    if "count_nu" not in group.columns or not period_start or not period_end:
        return None
    counts = pd.to_numeric(group["count_nu"], errors="coerce").dropna()
    if counts.empty:
        return None
    expected_days = max((pd.Timestamp(period_end) - pd.Timestamp(period_start)).days + 1, 1)
    return round(min(float(counts.sum()) / expected_days, 1.0), 3)


def _number(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_geojson(frame):
    features = []
    for _, row in frame.iterrows():
        properties = {name: _json_value(row[name]) for name in REVIEW_SCHEMA}
        features.append(
            {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        _json_value(row["longitude"]),
                        _json_value(row["latitude"]),
                    ],
                },
            }
        )
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        indent=2,
    ) + "\n"


def _json_value(value):
    if isinstance(value, list):
        return value
    if value is pd.NA or pd.isna(value):
        return None
    return value


def _read_rdb(text):
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(lines) > 1 and _is_rdb_type_row(lines[1]):
        del lines[1]
    if not lines:
        return []
    return pd.read_csv(StringIO("\n".join(lines)), sep="\t", dtype=str).to_dict("records")


def _is_rdb_type_row(line):
    tokens = line.split("\t")
    return all(re.fullmatch(r"\d*[snd](\[\d+\])?", token.strip()) for token in tokens)


def _parse_discharge_records(text, service):
    rows = _read_rdb(text)
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={name: _canonical_column(name) for name in frame.columns})
    if "time" not in frame:
        for column in ["datetime", "dateTime"]:
            if column in frame:
                frame = frame.rename(columns={column: "time"})
                break
    discharge_column = _discharge_column(frame)
    required = {"site_no", "time"}
    missing = required - set(frame.columns)
    if missing or discharge_column is None:
        missing_list = sorted(missing | ({"discharge_cfs"} if discharge_column is None else set()))
        raise ValueError("USGS discharge records are missing required columns: " + ", ".join(missing_list))
    out = pd.DataFrame(
        {
            "site_no": frame["site_no"].astype(str),
            "time": pd.to_datetime(frame["time"], errors="coerce", format="mixed"),
            "discharge_cfs": pd.to_numeric(frame[discharge_column], errors="coerce"),
            "source": f"usgs_{service}",
        }
    )
    out = out.dropna(subset=["site_no", "time", "discharge_cfs"]).sort_values(["site_no", "time"])
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out.to_dict("records")


def _discharge_column(frame):
    candidates = []
    for column in frame.columns:
        text = str(column)
        if "00060" not in text:
            continue
        if text.endswith("_cd") or text.endswith("_approval_cd") or text.endswith("_qualifiers"):
            continue
        candidates.append(column)
    if candidates:
        return candidates[0]
    for column in ["discharge_cfs", "value"]:
        if column in frame:
            return column
    return None


def _bbox_from_search_geometry(value, paths, *, buffer_km=None):
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        root = paths.get("location_root") or paths.get("repo_root") or Path.cwd()
        path = Path(root) / path
    if not path.exists():
        return None
    import geopandas as gpd

    frame = gpd.read_file(path).to_crs("EPSG:4326")
    if buffer_km is not None and float(buffer_km) != 0.0 and not frame.empty:
        projected_crs = frame.estimate_utm_crs() or "EPSG:3857"
        buffered = frame.to_crs(projected_crs).geometry.union_all().buffer(float(buffer_km) * 1000.0)
        bounds = gpd.GeoSeries([buffered], crs=projected_crs).to_crs("EPSG:4326").total_bounds
    else:
        bounds = frame.total_bounds
    return [round(float(value), 6) for value in bounds]
