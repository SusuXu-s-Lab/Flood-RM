"""Warn against overwrites of notebook-generated files. 
"""

from __future__ import annotations
from pathlib import Path
import yaml

_GENERATED_NOTICE = (
    "# GENERATED FILE — do not edit. Overwritten when {source} runs.\n"
    "# Source of truth is the location config and the code that produces this file.\n"
)

def generated_yaml_text(data, *, source: str, sort_keys: bool = False) -> str:
    return _GENERATED_NOTICE.format(source=source) + yaml.safe_dump(
        data, sort_keys=sort_keys
    )

def write_generated_yaml(path, data, *, source: str, sort_keys: bool = False) -> Path:
    path = Path(path)
    path.write_text(
        generated_yaml_text(data, source=source, sort_keys=sort_keys),
        encoding="utf-8",
    )
    return path