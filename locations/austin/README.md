# Austin YAML Map

Edit these files:

- `config.yaml` - location identity, flood mode, event drivers, and included YAML files.
- `smartds.yaml` - SMART-DS region paths, AOI source, and evaluation footprint.
- `sfincs.yaml` - SFINCS settings, forcing, parameters, and HydroMT-SFINCS recipes.
- `wflow.yaml` - Wflow settings, forcing, parameters, handoff rules, and native HydroMT-Wflow `steps:`.

Do not edit generated YAML under `data/`.

## Pipeline Commands

Preview the SMART-DS grid plot and inland Wflow-SFINCS flood workflow:

```bash
uv run python scripts/run_pipeline.py austin --stage all --dry-run
```

Run individual stages:

```bash
uv run python scripts/run_pipeline.py austin --stage grid
uv run python scripts/run_pipeline.py austin --stage flood
```

Executed notebook copies are written to `data/pipeline/executed_notebooks/`.
Logs are written to `data/pipeline/logs/`. Add `--in-place` only when source
notebooks should be overwritten with executed outputs.
