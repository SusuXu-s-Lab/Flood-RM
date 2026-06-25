# Pipeline Scripts
`run_pipeline.py` executes the notebook order declared by each location's `pipeline.py`.

## Instructions
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