"""Lazy native adapters for the NLR Distribution Suite stack.

The local package owns artifact contracts and study-specific audit logic.  It
uses the suite packages directly for the domains they own:

* SHIFT: OSM parcel/road acquisition, feeder graph construction, phase/voltage/
  equipment mapping, and GDM ``DistributionSystem`` assembly.
* DiTTo: parsing/writing distribution model formats such as OpenDSS.
* GDM: validated distribution-system objects, quantities, JSON serialization,
  graph/connectivity access.
* ERAD: GDM-to-asset conversion, hazard systems, fragility probability models,
  and resilience simulation.

Imports are intentionally lazy so documentation, audits, and artifact utilities
remain importable in environments that have not installed the full simulator
stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Mapping


class NativeDependencyError(ImportError):
    """Raised when a requested native suite package is not installed."""


def require_module(import_path: str, *, package: str, purpose: str) -> Any:
    """Import a native suite module and emit an actionable error on failure."""

    try:
        return import_module(import_path)
    except ImportError as exc:  # pragma: no cover - depends on optional stack.
        raise NativeDependencyError(
            f"{purpose} requires `{package}`. Install the native NLR suite extra, "
            f"for example `pip install -e '.[suite]'`, then retry."
        ) from exc


# ---- GDM -----------------------------------------------------------------


def distribution_system_class() -> type[Any]:
    """Return the native GDM ``DistributionSystem`` class."""

    try:
        from gdm import DistributionSystem  # type: ignore
    except ImportError:
        module = require_module("gdm.distribution", package="grid-data-models", purpose="GDM system loading")
        DistributionSystem = module.DistributionSystem
    return DistributionSystem


def read_gdm_json(path: str | Path) -> Any:
    """Load a GDM system using native ``DistributionSystem.from_json``."""

    return distribution_system_class().from_json(Path(path))


def write_gdm_json(system: Any, path: str | Path, *, overwrite: bool = True) -> Path:
    """Serialize a native GDM system using ``system.to_json``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    system.to_json(path, overwrite=overwrite)
    return path


def load_gdm_dataset(
    *,
    dataset_name: str,
    version: str,
    source_name: str | None = None,
    system_type: type[Any] | None = None,
) -> Any:
    """Load a published GDM example dataset through ``gdmloader``.

    This mirrors the ERAD/GDM documentation path and avoids local URL/S3 layout
    code in this repository.
    """

    source_module = require_module("gdmloader.source", package="gdmloader", purpose="GDM dataset loading")
    constants = require_module("gdmloader.constants", package="gdmloader", purpose="GDM dataset loading")
    loader = source_module.SystemLoader()
    source = getattr(constants, source_name or "GCS_CASE_SOURCE")
    loader.add_source(source)
    return loader.load_dataset(
        source_name=source.name,
        system_type=system_type or distribution_system_class(),
        dataset_name=dataset_name,
        version=version,
    )


# ---- DiTTo ---------------------------------------------------------------


@dataclass(frozen=True)
class DittoModelIO:
    """Native DiTTo reader/writer adapter for one in-memory GDM system."""

    system: Any
    source_path: Path | None = None


def read_opendss(master_dss: str | Path) -> DittoModelIO:
    """Parse OpenDSS with DiTTo's native OpenDSS ``Reader``."""

    reader_module = require_module(
        "ditto.readers.opendss.reader",
        package="nrel-ditto",
        purpose="OpenDSS parsing",
    )
    path = Path(master_dss)
    reader = reader_module.Reader(path)
    return DittoModelIO(system=reader.get_system(), source_path=path)


def write_opendss(
    system: Any,
    output_path: str | Path,
    *,
    separate_substations: bool = False,
    separate_feeders: bool = False,
    **writer_kwargs: Any,
) -> Path:
    """Write a native GDM system to OpenDSS using DiTTo's OpenDSS ``Writer``."""

    writer_module = require_module(
        "ditto.writers.opendss.write",
        package="nrel-ditto",
        purpose="OpenDSS writing",
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    writer = writer_module.Writer(system)
    writer.write(
        output_path=output_path,
        separate_substations=separate_substations,
        separate_feeders=separate_feeders,
        **writer_kwargs,
    )
    return output_path


# ---- SHIFT ---------------------------------------------------------------


def shift_distance(value: float, units: str = "meter") -> Any:
    """Construct the native GDM/Infrasys distance quantity used by SHIFT."""

    try:
        from infrasys.quantities import Distance  # type: ignore
    except ImportError:
        from gdm.quantities import Distance  # type: ignore
    return Distance(value, units)


def shift_geo_location(longitude: float, latitude: float) -> Any:
    """Construct native ``shift.GeoLocation``."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT geospatial/model build")
    return shift.GeoLocation(longitude=float(longitude), latitude=float(latitude))


def load_shift_test_catalog(name: str = "p1rhs7_1247.json") -> Any:
    """Load SHIFT's packaged GDM equipment catalog by native GDM JSON syntax."""

    shift = require_module("shift", package="nrel-shift", purpose="SHIFT equipment catalog loading")
    models = Path(shift.__file__).resolve().parent.parent.parent / "tests" / "models"
    return read_gdm_json(models / name)


# ---- ERAD ----------------------------------------------------------------


def asset_system_from_gdm(distribution_system: Any) -> Any:
    """Convert a GDM ``DistributionSystem`` with native ``AssetSystem.from_gdm``."""

    systems = require_module("erad.systems", package="nrel-erad", purpose="ERAD GDM asset conversion")
    return systems.AssetSystem.from_gdm(distribution_system)


def run_erad_hazard(asset_system: Any, hazard_system: Any, *, curve_set: str | None = None) -> Any:
    """Run ERAD's native ``HazardSimulator`` for a prepared asset/hazard system."""

    runner = require_module("erad.runner", package="nrel-erad", purpose="ERAD hazard simulation")
    simulator = runner.HazardSimulator(asset_system=asset_system)
    kwargs = {"curve_set": curve_set} if curve_set else {}
    return simulator.run(hazard_system=hazard_system, **kwargs)


def native_dependency_versions() -> dict[str, str]:
    """Best-effort versions for optional native suite dependencies."""

    import importlib.metadata as md

    names = {
        "grid-data-models": "grid-data-models",
        "nrel-ditto": "NREL-ditto",
        "nrel-shift": "nrel-shift",
        "nrel-erad": "NREL-erad",
        "gdmloader": "gdmloader",
    }
    out: dict[str, str] = {}
    for label, dist_name in names.items():
        try:
            out[label] = md.version(dist_name)
        except md.PackageNotFoundError:
            out[label] = "not_installed"
    return out
