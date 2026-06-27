# Refactor Task Dispatch

These prompts are the active workload for the next domain-agent pass.

Use `docs/agent_work_packages.md` as the ownership contract before starting any task.
The task files are the agent prompts; the work-package file controls exact source/test
ownership and cross-domain coordination.

## Assignments

| Task | Agent | Source ownership | Report |
| --- | --- | --- | --- |
| `4a.md` | Power/grid construction and audit | `src/power/**` | `docs/agent_reports/power.md` |
| `4b.md` | Design events and source collection | `src/design_events/**` | `docs/agent_reports/design_events.md` |
| `4c.md` | SFINCS/SnapWave setup and scenarios | `src/sfincs_runs/**` | `docs/agent_reports/sfincs_runs.md` |
| `4d.md` | Wflow coupling and handoff | `src/wflow_runs/**` | `docs/agent_reports/wflow_runs.md` |
| `4e.md` | Shared infrastructure | `src/study_location.py`, `src/aoi.py`, selected shared runtime/plumbing files | `docs/agent_reports/shared_infrastructure.md` |

## Not Assigned To Domain Agents

- `src/fiat_runs/**`
- notebooks under `locations/**`
- cross-domain tests such as `tests/test_architecture_smoke.py`, `tests/cluster/**`, and
  broad `tests/flood_rm/**` notebook/artifact-contract tests
- roadmap, trace, inventory, and task coordination docs

These belong to the reserved integration lane in `docs/agent_work_packages.md`.
