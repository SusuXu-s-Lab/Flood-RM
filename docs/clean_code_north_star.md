# Clean Code North Star — Flood-RM

## Purpose

This document defines what *clean* means in this repository.

Flood-RM is a scientific Python codebase for compound-flood design events and grid
resilience. Clean code here reads like code a careful researcher-engineer would write:
direct, vectorized, explicit about scientific assumptions, reproducible, and light on
architecture. The goal of a cleanup is never to make the code look more "enterprise."
The goal is to make it **smaller, flatter, easier to trace from a notebook, and harder
to get a wrong number out of.**

The desired end state is not a framework. It is a transparent backend that supports
clean notebooks and defensible science.

### The reference standard

`src/design_events/build_events/probability/` is the canonical example of the target
style. When in doubt, open `exceedance.py` and `realization.py` and copy their shape:

- **science-first docstrings** that state the estimator, the formula, and the assumptions;
- **input coercion at the top** (`u = np.asarray(u, dtype=float)`) instead of type-annotation noise;
- **duck-typed interfaces** (any object with `.cdf(u)` and `.simulate(n, seeds=...)` is a copula) instead of base classes;
- **explicit `raise ValueError(...)`** with the offending value, never a silent empty return;
- **frozen result dataclasses** (`AndExceedanceLabels`) instead of dict-of-everything;
- **keyword-only options** after `*` so call sites read like sentences.

House conventions, applied throughout:

- **No `->` return annotations.** Lean on clear names, docstrings, and runtime coercion.
  Annotate a parameter only when the name alone is ambiguous; never let annotations
  become the longest thing on a line.
- **Fewer lines.** Prefer one vectorized expression over a loop, one function over a
  class with one method, deleting code over wrapping it.
- **Units in names.** `water_level_m`, `peak_flow_cfs`, `depth_m`, `lag_hours`. A bare
  `depth` or `value` in a scientific signature is a smell.

---

## Scientific non-negotiables

These are specific to numerical/geospatial research code and matter more than any style rule.

1. **Determinism.** Every stochastic step takes an explicit `seed` and uses
   `np.random.default_rng(seed)`. Never call the legacy global `np.random.seed` /
   `np.random.choice`. Thread the seed through the call chain (`seeds=[int(seed)]`),
   exactly as the copula stages do.
2. **Fail loudly on bad science.** A missing column, an empty fit sample, or a
   non-finite rate must raise. Returning an empty `DataFrame`, zeros, or `NaN` silently
   corrupts return periods downstream and the error surfaces hundreds of lines later.
3. **Vectorize the hot path.** `DataFrame.iterrows`, `for _, row in ...`, and Python
   loops over arrays are red flags in any per-event or per-cell computation.
4. **dtype and domain hygiene.** Coerce to `float` at boundaries. Clip uniforms to
   `[eps, 1-eps]` before `ppf`/`cdf`. Wrap `1/x` in `np.errstate(divide="ignore")`.
   Use log-space distances for heavy-tailed drivers (rainfall, discharge).
5. **Preserve spatio-temporal structure.** Scalars (a Driver Probability Index) are for
   labeling; the realized forcing is always a real observed field scaled to the target,
   never a flat scalar broadcast over space and time.
6. **Cheap invariants, asserted.** Probabilities land in `[0, 1]`; weights sum to 1;
   return curves are monotone. Check these where it costs one line.

---

## Core principles

### 1. Prefer direct scientific Python over architecture

Use a class only for durable state, a real invariant, or a fitted object with several
methods that belong together (a marginal, a copula, a fragility curve). Otherwise, a
function.

Prefer

```python
events = (
    pd.read_csv(path, parse_dates=["event_time"])
    .query("water_level_m > @threshold_m")
    .sort_values("event_time")
)
```

Avoid

```python
loader = EventDataLoader(path)
validator = EventDataValidator(loader)
pipeline = EventProcessingPipeline(validator)
events = pipeline.execute_with_context(threshold=threshold_m)
```

### 2. Reproducible randomness, threaded explicitly

Every sampler in Flood-RM (copula simulation, analog selection, tie-breaking on ranks,
lag draws) must be seedable and deterministic. This is a correctness requirement, not a
style preference: a design catalog that cannot be regenerated bit-for-bit is not
defensible.

Prefer

```python
def draw_relative_lags(n, observed_lags=None, *, default_lag_hours=0.0, seed=0):
    """Draw per-event timing lags from observed inter-driver lags."""
    rng = np.random.default_rng(int(seed))
    if observed_lags is None or len(observed_lags) == 0:
        return np.full(int(n), float(default_lag_hours))
    pool = np.asarray(observed_lags, dtype=float)
    pool = pool[np.isfinite(pool)]
    return rng.choice(pool, size=int(n), replace=True)
```

Avoid

```python
def draw_relative_lags(n, observed_lags=None):
    np.random.seed(0)                      # global state, not reproducible per call
    return np.random.choice(observed_lags, size=n)
```

### 3. Separate science from plumbing

A formula should be findable. Path handling, manifest writing, logging, and validation
should not bury the estimator. One function selects events; another writes them.

Prefer

```python
def select_design_events(events, return_period):
    """Design events at a target return period (years)."""
    return events.loc[events["return_period_years"] == return_period].copy()


def write_design_events(events, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(path, index=False)
```

Avoid

```python
def process_design_events(events, return_period, output_path=None, validate=True,
                          create_dirs=True, write=True, plot=False, audit=True):
    ...
```

### 4. Duck-typed interfaces over class hierarchies

`exceedance.py` accepts *any* object exposing `cdf(u)` and `simulate(n, seeds=[...])` —
a fitted `pyvinecopulib.Vinecop`, an analytic Gaussian copula, or a test stub. State the
protocol in the docstring. Do not introduce an abstract base class for a single
implementation.

Prefer

```python
def and_joint_survival(u, *, copula=None, reference=None, method="auto"):
    """AND survival P(all drivers exceed u).

    `copula` is any object with `.cdf(u)` (m*d uniforms -> m) and
    `.simulate(n, seeds=[...])`. With a Monte-Carlo `reference` sample we count
    draws exceeding `u` in every dimension; that path is robust for vines whose
    analytic CDF is itself a QMC estimate.
    """
    ...
```

Avoid

```python
class AbstractCopula(ABC):
    @abstractmethod
    def cdf(self, u): ...
    @abstractmethod
    def simulate(self, n): ...

class VineCopulaAdapter(AbstractCopula):   # one implementation, wrapping a library type
    ...
```

### 5. Fail loudly; never return empty science

Silent fallbacks are dangerous in a design-event pipeline: an empty catalog or a zero
survival sails through the AND labeling and only explodes when SFINCS gets no forcing.

Prefer

```python
def and_return_period(survival, event_rate):
    """AND survival -> (annual exceedance prob, return period in years)."""
    survival = np.asarray(survival, dtype=float)
    rate = float(event_rate)
    if not (np.isfinite(rate) and rate > 0):
        raise ValueError(f"event_rate must be finite and > 0, got {event_rate!r}")
    aep = np.clip(rate * survival, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        period = np.where(survival > 0, 1.0 / (rate * survival), np.inf)
    return aep, period
```

Avoid

```python
def and_return_period(survival, event_rate):
    try:
        return 1.0 / (event_rate * survival)
    except Exception:
        logger.warning("bad survival; returning empty")
        return pd.DataFrame()              # corrupts everything downstream
```

### 6. Vectorize aggregations; pandas, not loops

Asset-exposure and damage summaries are the most common loop-to-pandas wins in this repo.

Prefer

```python
exposure = (
    impacts.loc[impacts["depth_m"] > 0]
    .groupby("asset_type", as_index=False)
    .agg(
        exposed_assets=("asset_id", "nunique"),
        mean_depth_m=("depth_m", "mean"),
        max_depth_m=("depth_m", "max"),
        damage_usd=("damage_usd", "sum"),
    )
)
```

Avoid

```python
rows = []
for asset_type in set(impacts["asset_type"]):
    subset = impacts[impacts["asset_type"] == asset_type]
    flooded = subset[subset["depth_m"] > 0]
    rows.append({
        "asset_type": asset_type,
        "exposed_assets": len(set(flooded["asset_id"])),
        "mean_depth_m": flooded["depth_m"].sum() / len(flooded),
        "max_depth_m": max(flooded["depth_m"]),
        "damage_usd": sum(flooded["damage_usd"]),
    })
exposure = pd.DataFrame(rows)
```

### 7. Prefer scipy/numpy idioms over hand-rolled math

A scalar `math.erf` fragility evaluation is fine for one asset, but for a flood-depth
grid sampled at thousands of assets, use the vectorized distribution directly.

Prefer

```python
from scipy.stats import lognorm

def failure_probability(depth_m, curve):
    """P(asset failure) at flood depth, from a lognormal fragility curve."""
    depth_m = np.asarray(depth_m, dtype=float)
    return lognorm.cdf(depth_m, curve.shape, loc=curve.loc_m, scale=curve.scale_m)
```

Avoid

```python
def failure_probability(depth_m, curve):
    out = []
    for d in depth_m:                      # Python loop over assets
        z = math.log((d - curve.loc_m) / curve.scale_m) / curve.shape
        out.append(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return out
```

### 8. Gridded data with xarray; named dims, not positional axes

Sample a flood-depth raster at asset coordinates with a single vectorized `.sel`. Do not
index `array[row, col]` in a loop with hand-computed affine transforms.

Prefer

```python
def sample_depth_at_assets(assets, flood_depth):
    """Nearest flood depth (m) at each asset point. `flood_depth` has dims (y, x)."""
    points = dict(asset=assets.index)
    return flood_depth.sel(
        x=xr.DataArray(assets.geometry.x, dims="asset", coords=points),
        y=xr.DataArray(assets.geometry.y, dims="asset", coords=points),
        method="nearest",
    ).to_series()
```

Avoid

```python
def sample_depth_at_assets(assets, depth_array, transform):
    depths = []
    for _, asset in assets.iterrows():
        col = int((asset.geometry.x - transform.c) / transform.a)
        row = int((asset.geometry.y - transform.f) / transform.e)
        depths.append(depth_array[row, col])
    return depths
```

### 9. Vector geometry with geopandas; spatial joins, not point loops

Prefer

```python
def assets_in_flooded_area(assets, flood_polygons):
    """Assets intersecting any flooded polygon (CRS must already match)."""
    return gpd.sjoin(assets, flood_polygons, how="inner", predicate="intersects")
```

Avoid

```python
def assets_in_flooded_area(assets, flood_polygons):
    hits = []
    for _, asset in assets.iterrows():
        for _, poly in flood_polygons.iterrows():
            if poly.geometry.contains(asset.geometry):
                hits.append(asset)
                break
    return gpd.GeoDataFrame(hits)
```

### 10. Small explicit dispatch over factories

A driver marginal is either an extreme-value tail (POT) or a bounded empirical CDF. A
two-branch function with explicit errors beats a registry.

Prefer

```python
def fit_index_marginal(values, *, event_rate, kind="pot", ev_type="pot"):
    """Fit a driver-role-aware marginal for one Driver Probability Index.

    kind="pot": AIC-selected EV tail (Exp/GPD) for conditioning extremes.
    kind="empirical": bounded empirical CDF for antecedent/state drivers.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        raise ValueError("need at least 3 finite values to fit a marginal")
    if kind == "empirical":
        return EmpiricalMarginal(v)
    if kind != "pot":
        raise ValueError(f"unknown marginal kind {kind!r}; use 'pot' or 'empirical'")
    params, dist_name = fit_best_distribution(v, ev_type, criterium="AIC")
    return HistoricalPeakMarginal(dist_name=dist_name, params=params,
                                  extremes_rate=float(event_rate), method=ev_type)
```

Avoid

```python
class MarginalFactory:
    def __init__(self):
        self._registry = {}
    def register(self, kind, builder): ...
    def create(self, kind, values, **kw):
        return self._registry[kind](values, **kw)
```

### 11. Dataclasses for result bundles and config

A fitted model or a labeled result is a real domain object with invariants. A `dict`
returned from a sampler loses its column contract; a mutable `ConfigManager` hides which
settings are required.

Prefer

```python
@dataclass(frozen=True)
class AndExceedanceLabels:
    survival: np.ndarray
    joint_aep: np.ndarray
    return_period_years: np.ndarray
    severity_band: pd.Series


@dataclass(frozen=True)
class SfincsConfig:
    model_root: Path
    crs: str
    resolution_m: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def read_sfincs_config(path):
    """Read SFINCS configuration from YAML, validating required fields."""
    raw = yaml.safe_load(Path(path).read_text())
    return SfincsConfig(
        model_root=Path(raw["model_root"]),
        crs=raw["crs"],
        resolution_m=float(raw["resolution_m"]),
        start_time=pd.Timestamp(raw["start_time"]),
        end_time=pd.Timestamp(raw["end_time"]),
    )
```

Avoid

```python
class ConfigManager:
    def __init__(self, path):
        self.path, self.config, self.loaded = path, {}, False
    def load(self):
        if not self.loaded:
            self.config = yaml.safe_load(open(self.path)); self.loaded = True
        return self
    def get(self, key, default=None):
        return self.config.get(key, default)   # required keys hidden behind defaults
```

### 12. Thin, domain-specific plotting helpers

The caller owns the figure and the saving. A helper draws one scientific thing onto an
axis and labels it with units.

Prefer

```python
def plot_water_level_timeseries(ax, water_levels, *, label=None):
    """Plot water level through time on a caller-owned axis."""
    ax.plot(water_levels["event_time"], water_levels["water_level_m"], label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Water level [m]")


def plot_fragility_curve(ax, curve, depths_m):
    """Overlay a lognormal flood-depth fragility curve."""
    ax.plot(depths_m, failure_probability(depths_m, curve), label=curve.erad_asset_type)
    ax.set_xlabel("Flood depth [m]")
    ax.set_ylabel("P(failure)")
```

Avoid

```python
class PlotManager:
    def create_plot(self, data, x_col, y_col, title=None, output_path=None,
                    theme=None, backend="matplotlib"):
        fig, ax = plt.subplots()
        ax.plot(data[x_col], data[y_col])
        if output_path: fig.savefig(output_path)
        return fig, ax
```

### 13. pathlib for paths

Prefer

```python
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
catalog_path = output_dir / "driver_catalog.csv"
```

Avoid

```python
if not output_dir.endswith("/"):
    output_dir += "/"
catalog_path = output_dir + "driver_catalog.csv"
```

### 14. Numerical hygiene as small, named guards

The recurring tail-math patterns deserve to be visible, not buried.

Prefer

```python
clip_eps = 1e-12

u = np.clip(u, clip_eps, 1.0 - clip_eps)          # safe before ppf / cdf
x = m.ppf(u)

# heavy-tailed driver: measure analog distance in log space
distance = np.abs(np.log(np.maximum(pool, clip_eps)) - np.log(max(target, clip_eps)))
```

Avoid

```python
x = m.ppf(u)                  # ppf(1.0) -> inf, ppf(0.0) -> -inf, NaNs propagate
distance = np.abs(pool - target)   # linear distance crushes the tail that matters
```

### 15. Compatibility aliases are one line, never a second implementation

Notebook imports are public; preserve the *name*, not a fork of the logic.

Acceptable

```python
def build_grid(*args, **kwargs):
    """Backward-compatible alias for notebooks. Prefer build_synthetic_grid."""
    return build_synthetic_grid(*args, **kwargs)
```

Not acceptable

```python
def build_grid(...):
    # 90 lines of the old implementation copied so both paths "still work"
    ...
```

---

## Function and class shape

### Good function shape

```python
def build_driver_catalog(sources, output_dir, *, station_id, start_year, end_year):
    """Build a driver catalog for one station and write it to disk."""
    ...
```

- a clear verb, units in the names;
- keyword-only options after `*`;
- input coerced at the top, errors raised early;
- returns one useful object, the same type every time;
- no `->`; no hidden globals; no mode flag that changes the return type.

Avoid: `def run(data=None, config=None, mode=None, flag=True, **extra):` — vague name,
optional everything, behavior switched by booleans, type-by-mode returns.

### When a class earns its place

Use a class for a fitted/stateful domain object: a marginal, a copula model, a fragility
curve, a frozen config or result bundle. `DriverDependenceModel` (vine + marginals +
rate) is a good class because the parts must travel together and share invariants.

Do not use a class to group functions. A `PlotHelper.plot_line(ax, x, y)` is just
`plot_line(ax, x, y)` with extra ceremony.

---

## Module naming

Name modules after responsibilities and scientific stages, not abstractions.

Good: `exceedance.py`, `dependence.py`, `design_catalog.py`, `realization.py`,
`fragility.py`, `extreme_value.py`, `return_curve.py`, `paired_observations.py`,
`baseline_network.py`.

Avoid: `utils.py`, `helpers.py`, `manager.py`, `processor.py`, `pipeline.py`, `core.py`,
`common.py`. Allowed only when no clearer domain name exists.

---

## Replacement, deletion, and the no-bloat rule

### Before adding any abstraction, all of these must be yes

1. It removes more code than it adds.
2. It makes notebook-to-backend tracing easier.
3. A new contributor understands it faster than the old code.
4. It is used in at least two real places.
5. It represents a real domain concept.

Forbidden unless explicitly justified in the PR: new Manager / Factory / Adapter /
Registry / Orchestrator / Pipeline classes; new compatibility layers; new broad
`utils.py` / `core.py`; new config systems; new inheritance trees or exception
hierarchies.

### Replace custom code with a library only when

it is shorter, idiomatic, easier to test, **does not change scientific semantics**, adds
no heavy dependency, and is covered by a test. Never swap a vetted estimator for a
library look-alike without proving equivalence.

### Delete code only when at least one holds

no call sites; duplicates another implementation; wraps one obvious library call; is
stale compatibility code unused by notebooks/scripts/tests; is unreachable. If uncertain,
mark it `DEFER_DELETE` in `docs/reduction_candidates.md` rather than guessing.

---

## The reader-traceability test

A new user should be able to answer, for any function, without reading five layers:

1. Which notebook step calls this?
2. What scientific concept does it implement (the estimator, the formula)?
3. What does it read, and what does it write?
4. Which lines are science and which are plumbing?
5. How would I test it without running a full SFINCS/Wflow model?

If the answer needs a tour through wrappers and managers, it is not clean yet.

---

## What a cleanup PR must show

- fewer source lines, or a justified small increase with clear simplification;
- fewer public names, fewer vague modules, fewer one-use wrappers, fewer compatibility branches;
- a clearer notebook → backend → artifact trace;
- preserved notebook-facing APIs and preserved scientific behavior (same numbers, same seeds);
- the same or better tests, including a pure-function smoke test for any changed estimator.

A PR that only moves code without reducing complexity is not a cleanup PR. **A passing
test suite is necessary but not sufficient: the task succeeds only if the edited code is
smaller, flatter, easier to trace from notebooks, and closer to idiomatic scientific
Python.** Score every diff against `docs/refactor_acceptance_rubric.md` before finalizing.
