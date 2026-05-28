import json
from pathlib import Path
import numpy as np

def parse_sfincs_inp(path):
    """Read sfincs.inp as a simple key/value dictionary."""
    path = Path(path)
    values = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip().lower()] = value.strip()
    return values

def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

def write_sfincs_inp(template_lines, out_path, *, overrides, remove_keys):
    """Rewrite sfincs.inp from a base template plus scenario-specific keys."""
    out_path = Path(out_path)
    written_keys = set()
    new_lines = []

    for raw_line in template_lines:
        if "=" not in raw_line:
            new_lines.append(raw_line)
            continue
        key, _ = raw_line.split("=", 1)
        key_clean = key.strip().lower()
        if key_clean in remove_keys:
            continue
        if key_clean in overrides:
            value = overrides[key_clean]
            new_lines.append(f"{key.rstrip():<21} = {value}")
            written_keys.add(key_clean)
        else:
            new_lines.append(raw_line)
    for key, value in overrides.items():
        if key in written_keys or key in remove_keys:
            continue
        new_lines.append(f"{key:<21} = {value}")
    out_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

def count_nonempty_lines(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())

def write_bzs(path, series, n_bnd, *, start_seconds=0.0):
    """Write one water-level time series to every boundary point."""
    path = Path(path)
    time_s = float(start_seconds) + np.arange(len(series), dtype=float) * 3600.0
    values = np.repeat(series.to_numpy(dtype=float)[:, None], n_bnd, axis=1)
    with path.open("w", encoding="utf-8") as stream:
        for t, row in zip(time_s, values):
            parts = [f"{t:8.1f}"] + [f"{value:7.3f}" for value in row]
            stream.write(" ".join(parts).rstrip() + "\n")
