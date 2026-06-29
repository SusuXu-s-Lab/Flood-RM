from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import time

import pandas as pd


COLLECTORS = {
    "aorc": "collect_sources.aorc:collect",
    "aorc_sst": "source_collection_v2.stochastic_boundary.rainfall:collect",
    "cora": "collect_sources.cora:collect",
    "era5": "collect_sources.era5:collect",
    "era5_waves": "collect_sources.era5:collect",
    "nwm": "collect_sources.nwm:collect",
    "usgs": "collect_sources.usgs:collect",
    "usgs_streamgages": "collect_sources.usgs:collect",
    "hurdat2": "collect_sources.hurdat2:collect",
    "lcra_hydromet": "collect_sources.lcra_hydromet:collect",
    "stream_geo": "collect_sources.stream_geo:collect",
    "stream_geo_nldi": "collect_sources.stream_geo:collect",
    "ssurgo": "collect_sources.ssurgo:collect",
    "national_hydrography": "collect_sources.national_hydrography:collect",
}

ORDER = tuple(COLLECTORS)


@dataclass(frozen=True)
class Step:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    spec: dict

    def settings(self, config: dict, paths: dict) -> dict:
        return {"config": config, "paths": paths, "source": self.name, "start": self.start, "end": self.end, "spec": self.spec}


@dataclass(frozen=True)
class Plan:
    config: dict
    paths: dict
    start: pd.Timestamp
    end: pd.Timestamp
    steps: tuple[Step, ...]

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps)

    def has(self, name: str) -> bool:
        return name in self.source_names

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({"source": s.name, "start": s.start, "end": s.end} for s in self.steps)


def plan(config: dict, paths: dict, *, start=None, end=None, sources: list[str] | tuple[str, ...] | None = None) -> Plan:
    collection = config.get("collection", {})
    base_start = pd.Timestamp(start or collection.get("start", "1979-01-01"))
    base_end = pd.Timestamp(end or collection.get("end", "2022-12-31"))
    if base_end < base_start:
        raise ValueError("collection end must be on or after start")
    wanted = tuple(sources or ORDER)
    steps = []
    for name in wanted:
        if name not in collection:
            continue
        spec = collection.get(name) or {}
        a, b = _source_window(name, spec, base_start, base_end)
        steps.append(Step(name, a, b, spec))
    return Plan(config, paths, base_start, base_end, tuple(steps))


def run(collection_plan: Plan, *, skip_existing=True, stop_on_error=True) -> pd.DataFrame:
    rows = []
    for step in collection_plan.steps:
        tic = time.monotonic()
        try:
            artifact = _collector(step.name)(step.settings(collection_plan.config, collection_plan.paths), skip_existing=skip_existing)
            rows.append({**artifact.row(), "duration_seconds": round(time.monotonic() - tic, 2)})
        except Exception as exc:
            rows.append({"source": step.name, "kind": "collection", "status": "failed", "duration_seconds": round(time.monotonic() - tic, 2), "error": f"{type(exc).__name__}: {exc}"})
            if stop_on_error:
                raise
    return pd.DataFrame(rows)


def _collector(name: str):
    module_name, function_name = COLLECTORS[name].split(":")
    return getattr(import_module(module_name), function_name)


def _source_window(name: str, spec: dict, base_start: pd.Timestamp, base_end: pd.Timestamp):
    start = pd.Timestamp(spec.get("start", spec.get("start_date", base_start)))
    end = pd.Timestamp(spec.get("end", spec.get("end_date", base_end)))
    if start < base_start or end > base_end or end < start:
        raise ValueError(f"{name} collection window must be within {base_start}..{base_end}")
    return start, end


collection_plan = plan
collect_sources = run
