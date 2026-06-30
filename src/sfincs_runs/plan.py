from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from paths import location_or_repo_path_from_paths


@dataclass(frozen=True)
class BaselineBuildPlan:
    study_location: str
    build_kind: str
    truth_set_kind: str
    notebook_path: Path
    base_model_root: Path
    data_catalog: Path
    grid_footprint_source: Path | None
    required_sources: tuple[str, ...]

    def summary_rows(self):
        return [
            {"item": "study_location", "value": self.study_location},
            {"item": "build_kind", "value": self.build_kind},
            {"item": "truth_set_kind", "value": self.truth_set_kind},
            {"item": "notebook", "value": self.notebook_path.as_posix()},
            {"item": "base_model_root", "value": self.base_model_root.as_posix()},
        ]


@dataclass(frozen=True)
class StaticIntakePlan:
    study_location: str
    model_crs: str
    static_root: Path
    raw_root: Path
    data_catalog: Path
    grid_footprint_source: Path | None
    required_static_inputs: tuple[str, ...]

    def summary_rows(self):
        return [
            {"item": "study_location", "value": self.study_location},
            {"item": "model_crs", "value": self.model_crs},
            {"item": "static_root", "value": self.static_root.as_posix()},
            {"item": "raw_root", "value": self.raw_root.as_posix()},
            {"item": "data_catalog", "value": self.data_catalog.as_posix()},
            {
                "item": "grid_footprint_source",
                "value": "" if self.grid_footprint_source is None else self.grid_footprint_source.as_posix(),
            },
            {"item": "required_static_inputs", "value": ", ".join(self.required_static_inputs)},
        ]


def build_static_intake_plan(config, paths):
    return StaticIntakePlan(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name")),
        model_crs=str(config.get("project", {}).get("model_crs", "EPSG:26919")),
        static_root=Path(paths["static_root"]),
        raw_root=Path(paths["raw_root"]),
        data_catalog=Path(paths["data_catalog"]),
        grid_footprint_source=_grid_footprint_source(config, paths),
        required_static_inputs=("terrain", "landcover", "coastline", "ssurgo"),
    )

def build_baseline_build_plan(config, paths):
    coastal_waves = bool(config.get("coastal_waves", False))
    build_kind = "wave_coupled" if coastal_waves else "regular_grid"
    truth_set_kind = "wave_coupled_truth_set" if coastal_waves else "hydrodynamic_truth_set"
    notebook_key = "build_sfincs_wave_coupled" if coastal_waves else "build_sfincs"
    notebook_default = (
        "02_flood/04/a_build_waves.ipynb"
        if coastal_waves
        else "02_flood/04/a_build_standard.ipynb"
    )
    return BaselineBuildPlan(
        study_location=str(paths.get("location_name") or config.get("project", {}).get("name")),
        build_kind=build_kind,
        truth_set_kind=truth_set_kind,
        notebook_path=_repo_path(paths, config.get("notebooks", {}).get(notebook_key, notebook_default)),
        base_model_root=Path(paths["base_model_root"]),
        data_catalog=Path(paths["data_catalog"]),
        grid_footprint_source=_grid_footprint_source(config, paths),
        required_sources=("era5_waves",) if coastal_waves else (),
    )

def _repo_path(paths, value):
    return location_or_repo_path_from_paths(paths, value)

def _grid_footprint_source(config, paths):
    value = config.get("grid_footprint", {}).get("source")
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    if (len(path.parts) == 1 or path.parts[0] in {"data", "02_flood", "01_grid"}) and paths.get("location_root") is not None:
        return Path(paths["location_root"]) / path
    return _repo_root(paths) / path

def _repo_root(paths):
    if paths.get("repo_root") is not None:
        return Path(paths["repo_root"])
    return Path(paths["root"]).parent
