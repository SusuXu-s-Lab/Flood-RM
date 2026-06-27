"""ERAD-derived flood-depth fragility curves for grid assets."""
import csv
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from paths import default_location_config_path, find_repo_root
from study_location import define_location

project_root = Path(__file__).resolve().parents[3]
repo_root = find_repo_root(Path(__file__).resolve())

def power_grid_root():
    definition = define_location(default_location_config_path(repo_root))
    path = Path(definition.grid.get("power_grid_root", "data/power_grid"))
    return path if path.is_absolute() else definition.root / path

shared_fragility_dir = project_root / "artifacts" / "fragility"
default_curves_csv = shared_fragility_dir / "erad_flood_depth_curves.csv"
fragility_dir = power_grid_root() / "fragility"
default_mapping_csv = fragility_dir / "asset_type_mapping.csv"

@dataclass(frozen=True)
class FloodDepthFragilityCurve:
    """SciPy lognormal CDF parameters imported from ERAD."""

    erad_asset_type: str
    distribution: str
    shape: float
    loc_m: float
    scale_m: float
    source_version: str
    source_commit: str

    def failure_probability(self, depth_m):
        if depth_m is None:
            return 0.0
        x = float(depth_m)
        if not math.isfinite(x) or x <= self.loc_m:
            return 0.0
        if self.distribution != "lognorm":
            raise ValueError(f"Unsupported fragility distribution: {self.distribution}")
        z = math.log((x - self.loc_m) / self.scale_m) / self.shape
        return min(max(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))), 0.0), 1.0)

@lru_cache(maxsize=None)
def load_flood_depth_curves(path=default_curves_csv):
    curves = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            curve = FloodDepthFragilityCurve(
                erad_asset_type=row["erad_asset_type"],
                distribution=row["distribution"],
                shape=float(row["shape"]),
                loc_m=float(row["loc_m"]),
                scale_m=float(row["scale_m"]),
                source_version=row["source_version"],
                source_commit=row["source_commit"],
            )
            curves[curve.erad_asset_type] = curve
    return curves

@lru_cache(maxsize=None)
def load_asset_type_mapping(path=default_mapping_csv):
    """Marshfield Asset Registry type -> ERAD asset type."""
    with Path(path).open(newline="", encoding="utf-8") as f:
        return {row["local_asset_type"]: row["erad_asset_type"] for row in csv.DictReader(f)}

def erad_asset_type(local_asset_type, mapping=None):
    key = str(local_asset_type).strip()
    asset_mapping = mapping or load_asset_type_mapping()
    try:
        return asset_mapping[key]
    except KeyError as exc:
        raise KeyError(f"No ERAD fragility mapping for local asset type {key!r}") from exc

def line_local_asset_type(line_class):
    """Asset Registry line class -> local fragility asset type."""
    value = (line_class or "").strip().lower()
    if value == "underground":
        return "line_underground"
    if value == "fuse":
        return "line_fuse"
    if value == "overhead":
        return "line_overhead"
    return "line_other"

def failure_probability(local_asset_type, depth_m, *, curves=None, mapping=None):
    """ERAD-derived flood failure probability for a Marshfield asset."""
    active_curves = curves or load_flood_depth_curves()
    curve_key = erad_asset_type(local_asset_type, mapping=mapping or load_asset_type_mapping())
    return active_curves[curve_key].failure_probability(depth_m)
