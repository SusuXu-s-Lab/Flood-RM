# Pipeline Scripts
`run_pipeline.py` executes the notebook order declared by each location's `pipeline.py`.

## Pipeline Instructions
The default stage is `all`, so this is equivalent:

```bash
uv run python scripts/run_pipeline.py <location>
```
i.e., uv run python scripts/run_pipeline.py marshfield

## Run one stage examples:
```bash
uv run python scripts/run_pipeline.py marshfield --stage grid
uv run python scripts/run_pipeline.py greensboro --stage flood
uv run python scripts/run_pipeline.py austin --stage all
```

## Outputs
By default, notebooks are written to:

```text
locations/<name>/data/pipeline/executed_notebooks/
```

# Get SMART-DS Data
`get_smartds.py` downloads SMART-DS feeder data from the public OEDI bucket. Regions: `sfo`, `austin`, `greensboro`. Omit `--execute` for a dry plan (lists files, downloads nothing).

```bash
uv run python scripts/get_smartds.py greensboro --execute
```

Useful flags: `--year`, `--scenario`, `--format`, `--subregion`, `--output-root`.

# Get Structures
`get_structures.py` pulls coastal-structure GIS data for a location's region and derives SFINCS structure layers. Omit `--execute` for a dry plan.

```bash
uv run python scripts/get_structures.py marshfield --execute
```

Useful flags: `--provider`, `--source-file` (ingest a local GeoJSON instead of querying), `--buffer-km`, `--no-derive` (raw pull only).

# ONM Export
`onm_export.py` builds a self-contained PowerModelsONM/DynaGrid run bundle zip. Output defaults to `artifacts/<location>_onm.zip`.

```bash
uv run python scripts/onm_export.py --location greensboro
```

Useful flags: `--export-scope {subregion,pilot,full,both}`, `--subregion-feeder` (repeatable), `--event-id`, `--force` (overwrite existing zip).