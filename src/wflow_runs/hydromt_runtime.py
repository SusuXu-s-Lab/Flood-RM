from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import sys


def _describe_hydromt_command(command: str, location_root: Path) -> tuple[str, str, str]:
    try:
        command_parts = _resolve_hydromt_command(command, location_root)
    except RuntimeError as exc:
        return command, "missing", str(exc)
    runner_status = "configured"
    if command_parts:
        runner = Path(command_parts[0])
        if (runner.name.startswith("hydromt") or runner.name.startswith("python")) and ".venv" in runner.parts:
            runner_status = "project_venv"
        elif len(command_parts) >= 3 and command_parts[1:3] == ["-m", "hydromt.cli.main"]:
            runner_status = "active_python"
        elif command_parts[0] == "hydromt":
            runner_status = "path"
    return shlex.join(command_parts), runner_status, ""


def _resolve_hydromt_command(command: str, location_root: Path) -> list[str]:
    command_parts = shlex.split(command)
    if not command_parts:
        raise ValueError("Cannot run an empty HydroMT-Wflow command.")
    if command_parts[0] != "hydromt":
        return command_parts

    for candidate in _project_hydromt_candidates(location_root):
        if _valid_console_script(candidate):
            return [str(candidate), *command_parts[1:]]

    for candidate in _project_python_candidates(location_root):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return [str(candidate), "-m", "hydromt.cli.main", *command_parts[1:]]

    hydromt_on_path = shutil.which("hydromt")
    if hydromt_on_path:
        return [hydromt_on_path, *command_parts[1:]]

    if importlib.util.find_spec("hydromt.cli.main") is not None:
        return [sys.executable, "-m", "hydromt.cli.main", *command_parts[1:]]

    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "python", "-m", "hydromt.cli.main", *command_parts[1:]]

    raise RuntimeError(_hydromt_missing_message(command, location_root))


def _valid_console_script(path: Path) -> bool:
    if not (path.exists() and os.access(path, os.X_OK)):
        return False
    try:
        first_line = path.open("rb").readline(512).decode("utf-8", errors="ignore").strip()
    except OSError:
        return False
    if not first_line.startswith("#!"):
        return True
    try:
        interpreter = shlex.split(first_line[2:].strip())[0]
    except (IndexError, ValueError):
        return False
    interpreter_path = Path(interpreter)
    if not interpreter_path.is_absolute():
        return True
    if not interpreter_path.exists():
        return False
    script_venv = _venv_root_for_script(path)
    interpreter_venv = _venv_root_for_script(interpreter_path)
    if script_venv is not None and interpreter_venv is not None:
        return script_venv == interpreter_venv
    return True


def _project_hydromt_candidates(location_root: Path) -> tuple[Path, ...]:
    script_name = "hydromt.exe" if os.name == "nt" else "hydromt"
    return tuple(path / script_name for path in _project_venv_bin_dirs(location_root))


def _project_python_candidates(location_root: Path) -> tuple[Path, ...]:
    script_name = "python.exe" if os.name == "nt" else "python"
    return tuple(path / script_name for path in _project_venv_bin_dirs(location_root))


def _project_venv_bin_dirs(location_root: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for start in (Path(location_root), Path.cwd()):
        resolved = start.resolve()
        roots.extend([resolved, *resolved.parents])

    unique_roots = []
    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        unique_roots.append(root)

    script_dirs = []
    for root in unique_roots:
        if os.name == "nt":
            script_dir = root / ".venv" / "Scripts"
        else:
            script_dir = root / ".venv" / "bin"
        if script_dir.exists():
            script_dirs.append(script_dir)
    return tuple(script_dirs)


def _venv_root_for_script(path: Path) -> Path | None:
    parent = path.parent
    if parent.name not in {"bin", "Scripts"}:
        return None
    venv_root = parent.parent
    return venv_root if venv_root.name == ".venv" else None


def _hydromt_missing_message(command: str, location_root: Path) -> str:
    tried = [
        "PATH: hydromt",
        *[str(path) for path in _project_hydromt_candidates(location_root)],
        *[f"{path} -m hydromt.cli.main" for path in _project_python_candidates(location_root)],
        f"{sys.executable} -m hydromt.cli.main",
    ]
    return (
        "HydroMT CLI executable not found for Wflow event replay. "
        "The notebook generated a HydroMT command, but the active kernel cannot spawn it. "
        "Activate the project environment or install the HydroMT-Wflow CLI, then rerun the cell. "
        f"Generated command: {command}. Tried: {', '.join(tried)}"
    )


def _hydromt_subprocess_env(location_root: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    debug_value = env.get("DEBUG")
    if debug_value is not None and not str(debug_value).lstrip("-").isdigit():
        env["DEBUG"] = "0"
    env["MPLCONFIGDIR"] = "/tmp/matplotlib"
    venv_dirs = _project_venv_bin_dirs(location_root or Path.cwd())
    existing_path = env.get("PATH", "")
    env["PATH"] = os.pathsep.join([*(str(path) for path in venv_dirs), existing_path])
    return env
