Apply the <500-line target per file as a pressure test, not as a hard law. When a file naturally contains multiple responsibilities, prefer splitting it into named modules over compressing it into dense code. When a file is already cohesive and readable, reduce redundancy without forcing an artificial split. Public API compatibility and tests beat line count.

You are refactoring a Python codebase to reduce line count and remove redundancy while preserving behavior.

Primary objective:
- Rewrite each target file to be much shorter, cleaner, and more hand-written-looking.
- For this file, target fewer than 500 physical lines if possible.
- Do not “code golf.” The code should look like a good engineer wrote it quickly and clearly.
- Preserve public function names, signatures, return dict keys, constants used by callers, output files, error behavior where tests or call sites depend on it, and visual semantics of plots.

Before editing:
1. Identify the public API:
   - Top-level functions/classes not prefixed with `_`.
   - Any constants imported elsewhere.
   - Any return dict keys used in tests or callers.
2. Search the repo for call sites before renaming or deleting anything.
3. Run existing tests, or at least run import/syntax checks.
4. Create a quick map of the file:
   - imports
   - public functions/classes
   - private helpers
   - repeated logic blocks
   - comments/docstrings that are explanatory versus outdated or redundant

Rewrite principles:
- Prefer deleting, merging, and simplifying over adding abstractions.
- Use standard Python library tools and existing third-party APIs directly instead of introducing one-off helper functions.
- Keep small helpers only when they collapse repeated nontrivial logic used at least twice.
- Inline helpers that are used once and are only 1–3 obvious lines.
- Remove defensive branches that only guard impossible states unless tests or real data require them.
- Remove repeated imports, repeated module docstrings, overly spaced blank lines, and long comments that restate code.
- Keep comments only for domain-specific decisions, non-obvious math, external data quirks, or behavior that future maintainers could accidentally break.
- Favor comprehensions, `dict.setdefault`, `zip`, `sum`, `min`/`max`, `any`, `next`, `sorted`, `enumerate`, `Path`, `csv.DictReader`, `json`, `dataclasses`, and existing pandas/geopandas/shapely methods over custom loops/helpers where readability improves.
- Avoid machine-looking code:
  - no giant one-liners
  - no nested ternaries beyond very simple cases
  - no cryptic variable names
  - no unnecessary metaprogramming
  - no clever abstractions that hide the domain

For plotting/geospatial files specifically:
- Collapse repeated GeoDataFrame creation into one flexible helper:
  - point GeoDataFrames from lon/lat-style columns
  - line GeoDataFrames from from/to lon/lat columns
  - optional alias columns when needed by downstream code
- Merge duplicated switch segment logic:
  - one function should handle sectionalizing versus tie switches
  - support raw lon/lat tuples and projected shapely Points only if both modes are still required
- Merge repeated plotting calls:
  - repeated `LineCollection(...)` calls should use a small loop or data table
  - repeated point plotting should use one concise helper or direct loop
  - legends should be built from compact data lists, not long repeated `Line2D(...)` blocks
- Merge repeated hull/geometry logic:
  - one helper for polygonal geometry extraction
  - one helper for visible hull differencing
- Merge repeated domain/AABB logic:
  - one helper to choose UTM from points/domains
  - one helper to build buffered AABB domains from projected points
  - one helper to merge grouped `SfincsDomain` objects into a union AABB
- Merge repeated GeoJSON-writing logic:
  - one small function for `FeatureCollection`
  - one small function for writing JSON plus newline
- Keep public plotting functions readable even if helpers become denser.

Specific cleanup opportunities in `plotting.py`:
- Deduplicate all imports into one import section.
- Remove repeated section-level triple-quoted strings unless they are the module docstring.
- Merge `_line_gdf` and `_line_gdf_dft` into one `line_gdf(rows, aliases=False)` or similar.
- Replace `_bus_segment` with direct dictionary `.get` logic unless used in multiple places after rewrite.
- Merge `_tiny_tick` and `_tiny_tick_3857` into one function parameterized by delta and point type, or inline the simple tick construction.
- Merge `_switch_segments_3857` and `_sfo_style_switch_line_layers` if practical.
- Collapse `_overview_legend` and `_detail_legend` using compact specs.
- Consolidate `write_smart_ds_power_extent`, `write_marshfield_power_extent`, and `write_sfincs_domains` around shared GeoJSON payload writing.
- Consolidate `cluster_smart_ds_by_subregion`, `cluster_marshfield_by_feeder`, and `cluster_buses_kmeans` around shared “points to buffered AABB domain” logic.
- Consolidate `merge_overlapping_aabbs` and `merge_by_centroid_proximity` around shared union-find grouping and shared domain-merging output logic.

Line-count policy:
- Count physical lines after formatting.
- Blank lines count.
- Comments count.
- Generated/minified code is not allowed.
- It is acceptable to keep a file above 500 only when preserving behavior would otherwise require unreadable code. In that case, explain what prevented the reduction and the next safe split.

Validation requirements:
1. Run formatter/linter if configured by the repo.
2. Run tests touching the edited file.
3. Run:
   `python -m py_compile path/to/file.py`
4. Import the module from repo root.
5. Compare public API before and after:
   - public names
   - function signatures
   - dataclass fields
6. For functions returning manifests/dicts, preserve keys and value types.
7. For plotting functions, preserve output path behavior and figure-saving behavior.
8. Do not silently remove functionality unless it is unreachable, duplicated, or proven unused by repo search/tests.

Deliverable:
- Rewrite the file in place.
- Keep a short summary at the end:
  - original line count
  - new line count
  - largest reductions
  - behavior-preservation checks run
  - any risky changes or TODOs