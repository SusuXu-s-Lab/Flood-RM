#!/usr/bin/env python3
"""Write the fully resolved Location Configuration for inspection.

The per-location ``config.yaml`` and its included detail files are the
configuration source. This tool merges everything the way ``define_location``
does at runtime and writes a readable ``config.resolved.yaml`` next to the
location's ``config.yaml`` so stakeholders can see and then override every
knob.

Usage:
    python tests/flood_rm/show_resolved_config.py greensboro
    python tests/flood_rm/show_resolved_config.py
    python tests/flood_rm/show_resolved_config.py austin --stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from study_location import define_location  # noqa: E402


RESOLVED_CONFIG_FILENAME = "config.resolved.yaml"
RESOLVED_CONFIG_HEADER = (
    "# ==========================================================================\n"
    "# GENERATED FILE - DO NOT EDIT BY HAND.\n"
    "#\n"
    "# Fully resolved Location Configuration for {name} (flood_setting={flood_setting}):\n"
    "# config.yaml + included detail files, exactly as define_location() merges\n"
    "# them at runtime. There are no hidden methodology defaults in this file.\n"
    "#\n"
    "# To change a value: edit config.yaml or the relevant included detail file,\n"
    "# then regenerate with:\n"
    "#     python tests/flood_rm/show_resolved_config.py {name}\n"
    "# ==========================================================================\n"
)


def _locations() -> list[str]:
    base = repo_root / "locations"
    return sorted(p.name for p in base.iterdir() if (p / "config.yaml").exists())


def write_resolved_config(definition, out_path=None) -> Path:
    target = Path(out_path) if out_path is not None else definition.root / RESOLVED_CONFIG_FILENAME
    header = RESOLVED_CONFIG_HEADER.format(
        name=definition.name,
        flood_setting=definition.config.get("flood_setting", "coastal"),
    )
    target.write_text(
        header + yaml.safe_dump(definition.config, sort_keys=True, default_flow_style=False),
        encoding="utf-8",
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "location",
        nargs="?",
        help="Study Location name (default: every location under locations/).",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the resolved config instead of writing config.resolved.yaml.",
    )
    args = parser.parse_args()

    names = [args.location] if args.location else _locations()
    for name in names:
        config_path = repo_root / "locations" / name / "config.yaml"
        if not config_path.exists():
            parser.error(f"no config.yaml for location {name!r} at {config_path}")
        definition = define_location(config_path)
        if args.stdout:
            print(f"# ----- {name} -----")
            print(yaml.safe_dump(definition.config, sort_keys=True))
        else:
            out = write_resolved_config(definition)
            print(f"{name}: wrote {out.relative_to(repo_root)}")
    if not args.stdout:
        print(f"\nGenerated {RESOLVED_CONFIG_FILENAME} is a read-only reference; "
              "edit config.yaml or detail files to change settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
