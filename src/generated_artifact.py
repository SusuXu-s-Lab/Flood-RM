"""Stamp machine-written YAML so it is never mistaken for hand-edited config.

Generated manifests, data catalogs, and recipe files live side by side with
hand-authored Location Configuration under ``locations/``. Without a marker, a
stakeholder (or an AI reading ``git ls-files``) cannot tell which files are safe
to edit and which are overwritten on the next run. Every machine-written YAML
should go through :func:`write_generated_yaml` so it carries a visible notice.

The notice is a YAML comment, which ``yaml.safe_load`` ignores, so adding it is
safe for every consumer (HydroMT data catalogs, Wflow recipes, domain-set
manifests, ...).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit by hand. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)


def generated_yaml_text(data, *, source: str, sort_keys: bool = False) -> str:
    """Return YAML text for ``data`` prefixed with the generated-file notice."""
    return _GENERATED_NOTICE.format(source=source) + yaml.safe_dump(
        data, sort_keys=sort_keys
    )


def write_generated_yaml(path, data, *, source: str, sort_keys: bool = False) -> Path:
    """Write ``data`` to ``path`` as YAML stamped with the generated-file notice."""
    path = Path(path)
    path.write_text(
        generated_yaml_text(data, source=source, sort_keys=sort_keys),
        encoding="utf-8",
    )
    return path
