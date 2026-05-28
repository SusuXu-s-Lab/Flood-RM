from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SourceCollectionStep:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    spec: dict

    @property
    def start_date(self):
        return self.start.date().isoformat()

    @property
    def end_date(self):
        return self.end.date().isoformat()


@dataclass(frozen=True)
class SourceCollectionPlan:
    config: dict
    paths: dict
    start: pd.Timestamp
    end: pd.Timestamp
    steps: tuple[SourceCollectionStep, ...]

    @property
    def source_names(self):
        return tuple(step.name for step in self.steps)

    def has(self, name):
        return name in self.source_names

    def step(self, name):
        for step in self.steps:
            if step.name == name:
                return step
        raise KeyError(f"source is not configured: {name}")

    def settings_for(self, name):
        step = self.step(name)
        return {
            "config": self.config,
            "paths": self.paths,
            "start": step.start,
            "end": step.end,
            name: step.spec,
        }

    def summary_rows(self):
        return [
            {
                "source": step.name,
                "start": step.start_date,
                "end": step.end_date,
            }
            for step in self.steps
        ]


source_order = ("cora", "nwm", "aorc_sst", "era5_waves")


def build_source_collection_plan(config, paths, *, start=None, end=None):
    base_start, base_end = _collection_window(config, start=start, end=end)
    collection = config.get("collection", {})
    steps = []
    for name in source_order:
        if name not in collection:
            continue
        spec = collection.get(name) or {}
        source_start, source_end = _source_window(name, spec, base_start, base_end)
        steps.append(
            SourceCollectionStep(
                name=name,
                start=source_start,
                end=source_end,
                spec=spec,
            )
        )
    return SourceCollectionPlan(
        config=config,
        paths=paths,
        start=base_start,
        end=base_end,
        steps=tuple(steps),
    )


def _collection_window(config, start=None, end=None):
    collection = config.get("collection", {})
    start_ts = pd.Timestamp(start or collection.get("start", "1979-01-01"))
    end_ts = pd.Timestamp(end or collection.get("end", "2022-12-31"))
    if end_ts < start_ts:
        raise ValueError("end date must be on or after start date")
    return start_ts, end_ts


def _source_window(name, spec, base_start, base_end):
    if name == "nwm":
        return _bounded_window(base_start, base_end, spec, "start", "end", "nwm")
    if name == "aorc_sst":
        return _bounded_window(base_start, base_end, spec, "start_date", "end_date", "aorc_sst")
    return base_start, base_end


def _bounded_window(base_start, base_end, spec, start_key, end_key, label):
    start_ts = pd.Timestamp(spec.get(start_key, base_start))
    end_ts = pd.Timestamp(spec.get(end_key, base_end))
    if start_ts < base_start or end_ts > base_end or end_ts < start_ts:
        raise ValueError(f"{label} collection dates must stay within the base collection window")
    return start_ts, end_ts
