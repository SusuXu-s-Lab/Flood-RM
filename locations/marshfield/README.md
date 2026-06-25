# Marshfield YAML Map

Edit these files:

- `config.yaml` - location identity, flood mode, event drivers, sea-level-rise offsets, and included YAML files.
- `grid.yaml` - generated-grid paths, AOI source, source anchors, and grid artifacts. **Not a SMART-DS dataset.**
- `sfincs.yaml` - regular SFINCS settings, forcing, parameters, structures, and HydroMT-SFINCS recipes.
- `snapwave.yaml` - SnapWave settings, wave forcing, runup gauges, and wave-coupled HydroMT-SFINCS recipes.

Do not edit generated YAML under `data/`.

## Pipeline Commands

Preview the full generated-grid and coastal/SnapWave flood workflow:

```bash
uv run python scripts/run_pipeline.py marshfield --stage all --dry-run
```

Run individual stages:

```bash
uv run python scripts/run_pipeline.py marshfield --stage grid
uv run python scripts/run_pipeline.py marshfield --stage flood
```

Executed notebook copies are written to `data/pipeline/executed_notebooks/`.
Logs are written to `data/pipeline/logs/`. Add `--in-place` only when source
notebooks should be overwritten with executed outputs.
