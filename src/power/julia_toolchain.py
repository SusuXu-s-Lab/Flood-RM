"""Artifact-first Python adapters for repo-local Julia toolchains."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from os import PathLike


@dataclass(frozen=True)
class JuliaToolchainResult:
    """Result emitted by a Julia toolchain adapter."""

    output_json: Path
    command: list[str]
    stdout: str
    stderr: str
    summary: dict


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_powermodels_onm_smoke(
    *,
    network_dss: Path | str,
    settings_json: Path | str,
    output_json: Path | str,
    julia_executable: PathLike[str] | str = "julia",
    julia_channel: str = "1.10",
    repo_root: Path | str | None = None,
    project_dir: Path | str = "julia/onm",
    script_path: Path | str = "scripts/julia/powermodels_onm_smoke.jl",
) -> JuliaToolchainResult:
    """Run the PowerModelsONM parser smoke gate through the Julia project."""

    root = Path(repo_root) if repo_root is not None else _repo_root()
    output_path = Path(output_json)
    command = [
        str(julia_executable),
        f"+{julia_channel}",
        f"--project={project_dir}",
        str(script_path),
        str(network_dss),
        str(settings_json),
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(output_path.read_text(encoding="utf-8"))
    return JuliaToolchainResult(
        output_json=output_path,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        summary=summary,
    )
