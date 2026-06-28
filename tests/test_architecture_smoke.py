from __future__ import annotations

import ast
import importlib
import inspect
import json
from collections import Counter
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
NOTEBOOK_ROOT = REPO_ROOT / "locations"

LOCAL_ROOTS = {
    "design_events",
    "fiat_runs",
    "power",
    "sfincs_runs",
    "study_location",
    "wflow_runs",
}

KNOWN_DUPLICATE_PUBLIC_NAMES = {
    "design_events.build_events.probability": {
        "build_inland_catalog",
        "build_joint_catalog",
        "build_tail",
    },
    "design_events.collect_sources": {"collect_warmup"},
    "power.exports": {"control_registry", "export_base"},
    "sfincs_runs.build_base": {
        "build_domains",
        "create_handoffs",
        "set_observations",
        "validate_physics",
    },
    "wflow_runs": {
        "build_meteo",
        "plan_handoff",
        "plan_streamflow",
        "plan_warmup",
        "prepare_handoff",
        "prepare_instates",
        "require_handoff",
        "validate_geometry",
        "validate_instates",
        "validate_staticmaps",
    },
}

SIGNATURE_GUARDS = {
    ("design_events.build_events.workflow", "load_runtime"): {"location_root"},
    ("design_events.collect_sources.workflow", "load_runtime"): {"location_root"},
    ("fiat_runs", "load_notebook_runtime"): {"location_root"},
    ("power.exports", "build_event_window_bundle"): {
        "event_start",
        "horizon_hours",
        "load_profiles",
        "blocks",
        "sandbox_id",
    },
    ("power.exports", "export_base"): {"registry_dir", "output_dir", "debug_csv"},
    ("power.exports", "export_powermodels_onm"): {
        "opendss_root",
        "asset_registry_dir",
        "blocks",
        "switches",
        "der_inventory",
        "load_profiles",
        "output_dir",
    },
    ("power.resilience", "build_blocks"): {"buses", "lines", "loads", "sources", "switches", "location_id"},
    ("power.resilience", "build_der_inventory"): {"facility_rows", "location_id"},
    ("power.resilience", "build_load_matches"): {
        "facility_rows",
        "asset_rows",
        "control_unit_rows",
        "location_id",
    },
    ("power.resilience", "load_inputs"): {
        "critical_facility_records",
        "load_match_by_facility",
        "load_profile_assignments_path",
        "oedi_profile_cache_dir",
    },
    ("power.resilience", "size_der"): {"smart_ds_compat_dir", "reopt_client"},
    ("power.resilience", "solve_switches"): {"feeder", "k_switches"},
    ("sfincs_runs.build_base", "build_domains"): {"config", "paths"},
    ("sfincs_runs.build_base", "plan_inland_sfincs_base"): {"config", "paths"},
    ("sfincs_runs.build_base.region_notebook", "load_runtime"): {"location_root"},
    ("sfincs_runs.config", "load_sfincs_runtime"): {"location_root"},
    ("sfincs_runs.scenarios", "plan_example"): {"config", "paths"},
    ("sfincs_runs.scenarios", "stage_inland_coupled_example_forcing"): {"config", "paths"},
    ("sfincs_runs.scenarios.event_forcing", "run_model"): {"run_root"},
    ("wflow_runs", "build_meteo"): {"config", "location_root", "event_id"},
    ("wflow_runs", "ensure_dynamic_handoff"): {"config", "location_root", "event_id"},
    ("wflow_runs", "plot_event_precipitation_peak_discharge"): {"catalog", "location_root"},
    ("wflow_runs.notebook", "load_runtime"): {"location_root"},
}


def source_module_names():
    names = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(SRC_ROOT).with_suffix("")
        if relative.name == "__init__":
            parts = relative.parts[:-1]
        else:
            parts = relative.parts
        if parts:
            names.append(".".join(parts))
    return tuple(names)


def notebook_paths():
    return tuple(
        path
        for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb"))
        if "data" not in path.parts and ".ipynb_checkpoints" not in path.parts
    )


def notebook_local_imports():
    imports = set()
    for notebook_path in notebook_paths():
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source") or "")
            tree = ast.parse(source or "\n", filename=f"{notebook_path}:cell{cell_index}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".", 1)[0] in LOCAL_ROOTS:
                            imports.add((notebook_path, alias.name, None))
                elif isinstance(node, ast.ImportFrom):
                    module_name = node.module or ""
                    if module_name.split(".", 1)[0] in LOCAL_ROOTS:
                        for alias in node.names:
                            if alias.name != "*":
                                imports.add((notebook_path, module_name, alias.name))
    return tuple(sorted(imports, key=lambda item: (str(item[0]), item[1], item[2] or "")))


@pytest.mark.parametrize("module_name", source_module_names())
def test_public_source_modules_import_with_reference_location(module_name):
    """Smoke-test source imports without downloads, credentials, notebooks, or solvers."""

    importlib.import_module(module_name)


@pytest.mark.parametrize(("notebook_path", "module_name", "symbol_name"), notebook_local_imports())
def test_notebook_local_imports_resolve(notebook_path, module_name, symbol_name):
    """Notebook-facing imports are the public workflow interface."""

    module = importlib.import_module(module_name)
    if symbol_name is not None:
        assert hasattr(module, symbol_name), f"{notebook_path} imports missing {module_name}.{symbol_name}"


@pytest.mark.parametrize(("module_name", "symbol_name"), sorted(SIGNATURE_GUARDS))
def test_notebook_facing_signature_keeps_required_parameters(module_name, symbol_name):
    module = importlib.import_module(module_name)
    obj = getattr(module, symbol_name)
    signature = inspect.signature(obj)
    parameter_names = set(signature.parameters)

    assert SIGNATURE_GUARDS[(module_name, symbol_name)] <= parameter_names


@pytest.mark.parametrize("module_name", sorted(KNOWN_DUPLICATE_PUBLIC_NAMES))
def test_public_facades_do_not_gain_new_duplicate_exports(module_name):
    module = importlib.import_module(module_name)
    names = list(getattr(module, "__all__", ()))
    duplicates = {name for name, count in Counter(names).items() if count > 1}

    assert duplicates <= KNOWN_DUPLICATE_PUBLIC_NAMES[module_name]
