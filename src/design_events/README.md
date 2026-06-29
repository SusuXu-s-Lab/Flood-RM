# Design Events Module Guide
----
## Probability, Records, and Event Sampling

Use these files when working with event probabilities, fitted records, return periods, or sampled event magnitudes.

- `probability.py`  
  Joint-exceedance probability, joint return periods, severity bands, importance weights, joint-law objects, mixture-law objects, and catalog index selection.

- `records.py`  
  Driver records, fitted marginals, paired POT/co-occurrence samples, coastal NTR handling, and member-library construction.

- `extreme_value.py`  
  POT and block-maxima fitting, distribution selection, return values, bootstrap confidence bands, and EVA plotting helpers.

- `driver_records.py`  
  Loads real driver records from configuration and converts them into aligned time series and paired observations.

- `mixture.py`  
  Fits and samples storm-type mixtures, especially when different coastal storm populations have different dependence structures.

- `storm_type.py`  
  Classifies historical storms as tropical cyclones, nor’easters, other non-tropical storms, or unresolved events.
----
## Realization and Timing

Use these files when sampled event magnitudes need to be connected back to real forcing members and timing metadata.

- `realization.py`  
  Selects analog events, applies scale factors, controls member reuse, and creates long-form driver realization rows.

- `timing.py`  
  Adds timing information such as rainfall peak timing, storm loading patterns, compound lags, inland rainfall timing, and timing summaries.

- `selection.py`  
  Builds event distribution summaries, stress/training selections, compound-stress pairings, antecedent soil-moisture attachments, and related selection artifacts.
----
## Coastal and Inland Builders

Use these files when tracing location-specific catalog behavior.

- `coastal.py`  
  Handles coastal NTR/tide behavior, sampled coastal peaks, surge hydrograph templates, member artifacts, and realization audit metadata.

- `inland.py`  
  Handles inland rainfall-first catalogs, streamflow and soil-moisture roles, Wflow-coupled event artifacts, handoff files, and inland audit manifests.

- `peaks.py`  
  Loads boundary water levels, detrends records, extracts peaks, writes marginal catalogs, and creates threshold-model sensitivity artifacts.