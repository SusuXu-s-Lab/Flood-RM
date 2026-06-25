#!/usr/bin/env python3
"""Run a location's declared notebook pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import nbformat
from nbclient import NotebookClient


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("grid", "flood", "all")

class PipelineError(RuntimeError):
    """User-facing pipeline configuration or execution error."""

@dataclass(frozen=True)
class NotebookRun:
    source: Path
    output: Path
    log: Path

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("location", help="Location folder under locations/, for example marshfield.")
    parser.add_argument("--stage", choices=STAGES, default="all", help="Notebook stage to run.")
    parser.add_argument("--in-place", action="store_true", help="Write executed outputs back to source notebooks.")
    parser.add_argument("--dry-run", action="store_true", help="Print the notebook order without executing.")
    parser.add_argument("--kernel", default="python3", help="Kernel name used for notebook execution.")
    parser.add_argument("--timeout", type=int, default=None, help="Per-cell timeout in seconds; omit for no timeout.")
    failure = parser.add_mutually_exclusive_group()
    failure.add_argument("--stop-on-error", dest="stop_on_error", action="store_true", help="Stop at first failure.")
    failure.add_argument(
        "--continue-on-error",
        dest="stop_on_error",
        action="store_false",
        help="Continue executing later notebooks after a failure.",
    )
    parser.set_defaults(stop_on_error=True)
    return parser.parse_args(argv)


def location_root(repo_root: Path, location: str) -> Path:
    root = repo_root / "locations" / location
    if not root.is_dir():
        raise PipelineError(f"Unknown location {location!r}; expected {root}.")
    return root

def load_pipeline_module(repo_root: Path, location: str) -> ModuleType:
    root = location_root(repo_root, location)
    path = root / "pipeline.py"
    if not path.exists():
        raise PipelineError(f"Missing pipeline manifest for {location!r}: {path}.")
    spec = importlib.util.spec_from_file_location(f"flood_rm_{location}_pipeline", path)
    if spec is None or spec.loader is None:
        raise PipelineError(f"Could not import pipeline manifest: {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def resolve_notebooks(repo_root: Path, location: str, stage: str) -> tuple[Path, ...]:
    if stage not in STAGES:
        raise PipelineError(f"Unknown stage {stage!r}; expected one of {', '.join(STAGES)}.")
    root = location_root(repo_root, location)
    module = load_pipeline_module(repo_root, location)
    pipelines = getattr(module, "PIPELINE_NOTEBOOKS", None)
    if not isinstance(pipelines, dict):
        raise PipelineError(f"{root / 'pipeline.py'} must define PIPELINE_NOTEBOOKS.")
    if stage not in pipelines:
        raise PipelineError(f"{root / 'pipeline.py'} does not define stage {stage!r}.")

    notebooks: list[Path] = []
    for value in pipelines[stage]:
        path = Path(value)
        if path.is_absolute():
            notebook = path
        else:
            notebook = root / path
        if notebook.suffix != ".ipynb":
            raise PipelineError(f"Pipeline entry is not a notebook: {value!r}.")
        if not notebook.exists():
            raise PipelineError(f"Pipeline notebook does not exist: {notebook}.")
        notebooks.append(notebook.resolve())
    return tuple(notebooks)

def output_notebook_path(location_root_: Path, notebook: Path, *, in_place: bool) -> Path:
    if in_place:
        return notebook
    return location_root_ / "data" / "pipeline" / "executed_notebooks" / notebook.relative_to(location_root_)

def log_path(location_root_: Path, notebook: Path) -> Path:
    return location_root_ / "data" / "pipeline" / "logs" / notebook.relative_to(location_root_).with_suffix(".log")

def build_runs(repo_root: Path, location: str, stage: str, *, in_place: bool) -> tuple[NotebookRun, ...]:
    root = location_root(repo_root, location).resolve()
    return tuple(
        NotebookRun(
            source=notebook,
            output=output_notebook_path(root, notebook, in_place=in_place),
            log=log_path(root, notebook),
        )
        for notebook in resolve_notebooks(repo_root, location, stage)
    )

@contextmanager
def temporary_env(values: dict[str, str]):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

def execution_env(repo_root: Path, location: str) -> dict[str, str]:
    root = location_root(repo_root, location)
    env = {"FLOOD_RM_LOCATION": location}
    config = root / "config.yaml"
    if config.exists():
        env["FLOOD_RM_LOCATION_CONFIG"] = str(config)
    return env

def execute_notebook(run: NotebookRun, *, kernel: str, timeout: int | None, env: dict[str, str]) -> None:
    run.output.parent.mkdir(parents=True, exist_ok=True)
    run.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with run.log.open("w", encoding="utf-8") as log:
        log.write(f"source: {run.source}\n")
        log.write(f"output: {run.output}\n")
        log.write(f"kernel: {kernel}\n")
        log.write(f"cwd: {run.source.parent}\n")
        log.write("status: running\n\n")
        try:
            notebook = nbformat.read(run.source, as_version=4)
            client = NotebookClient(
                notebook,
                kernel_name=kernel,
                timeout=timeout,
                resources={"metadata": {"path": str(run.source.parent)}},
            )
            with temporary_env(env):
                client.execute()
            nbformat.write(notebook, run.output)
        except Exception:
            log.write("status: failed\n")
            log.write(f"duration_seconds: {time.monotonic() - started:.2f}\n\n")
            log.write(traceback.format_exc())
            raise
        else:
            log.write("status: succeeded\n")
            log.write(f"duration_seconds: {time.monotonic() - started:.2f}\n")

def run_pipeline(args: argparse.Namespace, *, repo_root: Path = REPO_ROOT) -> int:
    runs = build_runs(repo_root, args.location, args.stage, in_place=args.in_place)
    root = location_root(repo_root, args.location)
    if args.dry_run:
        print(f"{args.location}:{args.stage}")
        for index, run in enumerate(runs, start=1):
            print(f"{index:02d}. {run.source.relative_to(root)} -> {run.output.relative_to(root)}")
        return 0
    
    env = execution_env(repo_root, args.location)
    failed = False
    for index, run in enumerate(runs, start=1):
        label = run.source.relative_to(root)
        print(f"[{index}/{len(runs)}] executing {label}")
        try:
            execute_notebook(run, kernel=args.kernel, timeout=args.timeout, env=env)
        except Exception as exc:
            failed = True
            print(f"failed: {label}")
            print(f"log: {run.log}")
            if args.stop_on_error:
                raise PipelineError(f"Notebook failed: {label}") from exc
        else:
            print(f"wrote {run.output}")
    return 1 if failed else 0

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_pipeline(args)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
