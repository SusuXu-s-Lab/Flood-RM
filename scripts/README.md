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

# Source-acquisition Workflow Tools
`get_structures.py` is the auditable, region-agnostic front of the SFINCS Structure
Layer setup: it resolves a location's Study Area bbox, pulls coastal-structure GIS
features from a registered public provider (MassGIS by default), records full
provenance in a manifest, and derives the SFINCS weir/thin-dam source layers with the
same reviewed logic the build notebook consumes. Providers live in a registry, so the
method that produced Marshfield's structures re-runs for any coastal location.

```bash
# Make an existing download auditable + (re)derive its SFINCS sources:
uv run python scripts/get_structures.py marshfield \
  --source-file locations/marshfield/data/static/structures/evidence/massgis_public_structures_2015_marshfield.geojson --execute

# Live-fetch a new region from a verified ArcGIS REST feature layer:
uv run python scripts/get_structures.py <location> --service-url <layer-url> --execute
```

Runs are a dry plan unless `--execute` is passed. Outputs land under
`locations/<name>/data/static/structures/` (`evidence/` for the raw pull + manifest,
`sources/` for the derived weir/thin-dam layers). Provider source links are recorded in
`artifacts/data_links.txt`.
