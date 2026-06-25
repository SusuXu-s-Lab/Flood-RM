"""Collect source artifacts for design-event forcing."""

from design_events.collect_sources.workflow import (
    SourceCollectionPlan,
    SourceCollectionStep,
    build_source_collection_plan,
)
from design_events.collect_sources.workflow import prepare_collection_prerequisites
from design_events.collect_sources.workflow import run_collect
from design_events.collect_sources.aorc_sst import collect_aorc_wflow_baseline_warmup
from design_events.collect_sources.usgs_streamgages import build_reviewed_streamgage_decisions

collect_warmup = collect_aorc_wflow_baseline_warmup

__all__ = [
    "SourceCollectionPlan",
    "SourceCollectionStep",
    "build_reviewed_streamgage_decisions",
    "build_source_collection_plan",
    "collect_aorc_wflow_baseline_warmup",
    "collect_warmup",
    "prepare_collection_prerequisites",
    "run_collect",
]
