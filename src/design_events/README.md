# Design Events Module Guide
----

Use these files first if you are trying to understand the overall catalog workflow.

- `workflow.py`  
  Runs the core reference-bundle workflow. This is the best starting point for understanding how records become paired POT samples, fitted laws, sampled events, realized drivers, timing metadata, and reviewer bundle outputs.

- `notebook.py`  
  Contains the notebook-facing runtime loader, coastal policy helpers, source inventory, and inland catalog/replay materialization.

- `runtime.py`  
  Turns a location YAML file and artifact manifests into the runtime configuration used by the workflow. Look here for path handling and catalog plan setup.

- `catalog.py`  
  Defines the main Event Catalog tables and validation checks. This file writes the core outputs: `events.csv`, `drivers.csv`, and `audit.json`.

- `build.py`  
  Contains notebook-facing builders for coastal, compound, and inland Wflow-coupled catalogs. This is the bridge between fitted records and the wide catalog outputs used downstream.

## Probability, Records, and Event Sampling

Use these files when working with event probabilities, fitted records, return periods, or sampled event magnitudes.

- `probability.py`  
  Joint-exceedance probability, joint return periods, severity bands, importance weights, joint-law objects, mixture-law objects, and catalog index selection.

- `records.py`  
  Driver records, fitted marginals, paired POT/co-occurrence samples, coastal NTR handling, config-to-record-spec normalization, and member-library construction.

- `extreme_value.py`  
  POT and block-maxima fitting, distribution selection, return values, bootstrap confidence bands, and EVA plotting helpers.

- `mixture.py`  
  Fits and samples storm-type mixtures, especially when different coastal storm populations have different dependence structures.

- `storm_type.py`  
  Classifies historical storms as tropical cyclones, nor’easters, other non-tropical storms, or unresolved events.

## Realization and Timing

Use these files when sampled event magnitudes need to be connected back to real forcing members and timing metadata.

- `realization.py`  
  Selects analog events, applies scale factors, controls member reuse, and creates long-form driver realization rows.

- `timing.py`  
  Adds timing information such as rainfall peak timing, storm loading patterns, compound lags, inland rainfall timing, and timing summaries.

- `selection.py`  
  Builds event distribution summaries, stress/training selections, compound-stress pairings, antecedent soil-moisture attachments, and related selection artifacts.

## Coastal and Inland Builders

Use these files when tracing location-specific catalog behavior.

- `coastal.py`  
  Handles coastal NTR/tide behavior, sampled coastal peaks, surge hydrograph templates, member artifacts, and realization audit metadata.

- `inland.py`  
  Handles inland rainfall-first reference-bundle normalization, streamflow/soil-moisture role metadata, Wflow-coupled event artifacts, handoff files, and inland audit manifests.

- `streamgage.py`  
  Builds USGS streamgage event-member tables for the inland external-boundary fluvial path. Notebooks should import streamgage member construction from this module directly.

- `peaks.py`  
  Loads boundary water levels, detrends records, extracts peaks, writes marginal catalogs, and creates threshold-model sensitivity artifacts.

## Outputs, Handoffs, and Review Tools

Use these files when preparing catalog outputs for notebooks, models, reviewers, or tests.

- `handoff.py`  
  Converts long `drivers` rows into the wide per-driver columns expected by operational notebooks and SFINCS/Wflow handoffs.

- `audit.py`  
  Builds a compact audit summary from a wide Event Catalog, including formulas, sampling mass, realization reuse, and probability-weight checks.

- `diagnostics.py`  
  Creates table-only diagnostics from `events`, `drivers`, and `audit`, including severity distribution, probability-weight checks, reuse, scale factors, and timing coverage.

- `plotting.py`  
  Contains notebook plots and visual diagnostics for records, fits, event distributions, timing, catalog coverage, copula fits, streamflow, and coastal/inland pairing checks.
