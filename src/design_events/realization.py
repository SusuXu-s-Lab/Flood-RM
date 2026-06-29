"""
Field-Preserving Realization (Layer 2): a sampled scalar index to an observed field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

clip_eps = 1e-12

def select_analog(targets, member_values, *, pool_size=75, sigma_scale=0.5, sigma_min=1e-6,
                  sigma_max=np.inf, reuse_penalty_lambda=0.15, log_space=True, seed=0):
    """Pick an observed member analog + scale factor for each target index value.

    For each target: restrict to the ``pool_size`` nearest observed members (log distance
    for heavy-tailed rainfall/discharge), draw one by Gaussian distance weight times a
    reuse penalty, scale its field by ``K = target/value``. Returns ``(template_idx, scale)``.
    """
    targets = np.asarray(targets, dtype=float)
    values = np.asarray(member_values, dtype=float)
    rng = np.random.default_rng(int(seed))

    finite = np.isfinite(values) & ((values > 0) if log_space else np.isfinite(values))
    valid_idx = np.flatnonzero(finite)
    if valid_idx.size == 0:
        raise ValueError("no valid member_values to select analogs from")

    def transform(x):
        return np.log(np.maximum(x, clip_eps)) if log_space else np.asarray(x, dtype=float)

    values_t = transform(values)
    valid_t = values_t[valid_idx]
    pool = max(1, min(int(pool_size), valid_idx.size))
    usage = np.zeros(values.size, dtype=int)

    chosen = np.empty(targets.size, dtype=int)
    scale = np.empty(targets.size, dtype=float)
    for i, target in enumerate(targets):
        if not np.isfinite(target) or (log_space and target <= 0):
            j = int(valid_idx[int(np.argmax(values[valid_idx]))])
            chosen[i], scale[i] = j, 1.0
            usage[j] += 1
            continue
        target_t = float(transform(np.array([target]))[0])
        near = np.argsort(np.abs(valid_t - target_t))[:pool]
        pool_t = valid_t[near]
        spread = float(np.std(pool_t)) if pool_t.size > 1 else sigma_min
        sigma = float(np.clip(sigma_scale * spread, sigma_min, sigma_max))
        distance_w = np.exp(-0.5 * ((pool_t - target_t) / sigma) ** 2)
        reuse_w = np.exp(-reuse_penalty_lambda * usage[valid_idx[near]])
        weights = distance_w * reuse_w
        if not np.isfinite(weights).any() or weights.sum() <= 0.0:
            weights = np.ones(near.size, dtype=float)
        weights = weights / weights.sum()
        j = int(valid_idx[near[int(rng.choice(near.size, p=weights))]])
        chosen[i] = j
        scale[i] = float(target / values[j]) if values[j] > 0 else 1.0
        usage[j] += 1
    return chosen, scale


def draw_lags(n, observed_lags=None, *, default_lag_hours=0.0, seed=0):
    """Draw per-event timing lags from observed inter-driver lags (preserves realistic
    relative peak timing); constant fallback when no observed pool is supplied."""
    if observed_lags is None or len(observed_lags) == 0:
        return np.full(int(n), float(default_lag_hours), dtype=float)
    rng = np.random.default_rng(int(seed))
    pool = np.asarray(observed_lags, dtype=float)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return np.full(int(n), float(default_lag_hours), dtype=float)
    return rng.choice(pool, size=int(n), replace=True)


select_analog_realization = select_analog
draw_relative_lags = draw_lags


def realize_driver(catalog, members, *, driver, target_column, index_column,
                   member_id_column="member_id", member_file_column="member_file",
                   time_column=None, design_method="scaled_analog", observed_lags=None,
                   default_lag_hours=0.0, log_space=True, pool_size=75,
                   reuse_penalty_lambda=0.15, seed=0):
    """Long-form Field-Preserving Realization rows for one driver.

    Returns a ``drivers.csv`` slice: one row per event for ``driver`` carrying ``member_id``
    / ``member_file`` / ``member_time`` (the field pointers), ``template_value``,
    ``scale_factor`` (``K = target/value``), and ``lag_hours``. Stays keyed to ``event_id``.
    """
    members = members.reset_index(drop=True)
    if index_column not in members:
        raise ValueError(f"members missing index column {index_column!r}")
    if member_id_column not in members:
        raise ValueError(f"members missing member id column {member_id_column!r}")

    targets = pd.to_numeric(catalog[target_column], errors="coerce").to_numpy(dtype=float)
    member_values = pd.to_numeric(members[index_column], errors="coerce").to_numpy(dtype=float)
    template_idx, scale = select_analog(
        targets, member_values, pool_size=pool_size,
        reuse_penalty_lambda=reuse_penalty_lambda, log_space=log_space, seed=seed,
    )
    selected = members.iloc[template_idx].reset_index(drop=True)

    rows = pd.DataFrame({
        "event_id": catalog["event_id"].to_numpy(),
        "driver": driver,
        "x": targets,
        "member_id": selected[member_id_column].astype(str).to_numpy(),
        "template_value": member_values[template_idx],
        "scale_factor": scale,
        "lag_hours": draw_lags(len(catalog), observed_lags=observed_lags,
                               default_lag_hours=default_lag_hours, seed=seed + 1),
        "realization_policy": design_method,
    })
    if member_file_column in members:
        rows["member_file"] = selected[member_file_column].astype(str).to_numpy()
    if time_column is not None and time_column in members:
        rows["member_time"] = selected[time_column].astype(str).to_numpy()
    return rows


def attach_field_preserving_realization(
    catalog,
    members,
    *,
    driver,
    target_column,
    index_column,
    member_id_column="member_id",
    member_file_column="member_file",
    time_column=None,
    design_method="scaled_analog",
    observed_lags=None,
    default_lag_hours=0.0,
    log_space=True,
    pool_size=75,
    reuse_penalty_lambda=0.15,
    seed=0,
):
    """Attach wide ``<driver>_*`` realization columns mapping each target index to an observed field.

    The wide-schema bridge the SFINCS staging and Wflow/SFINCS handoff consume (the long-form
    sibling is ``realize_driver``). Each row's ``target_column`` (a sampled Driver Probability
    Index) is matched to an observed member analog (``member_id``/``member_file`` field pointers)
    plus a scalar ``scale_factor`` and timing ``lag``; the field is staged + scaled downstream so
    spatio-temporal structure is preserved.
    """
    catalog = catalog.copy()
    members = members.reset_index(drop=True)
    if index_column not in members:
        raise ValueError(f"members missing index column {index_column!r}")
    if member_id_column not in members:
        raise ValueError(f"members missing member id column {member_id_column!r}")

    targets = pd.to_numeric(catalog[target_column], errors="coerce").to_numpy(dtype=float)
    member_values = pd.to_numeric(members[index_column], errors="coerce").to_numpy(dtype=float)
    template_idx, scale = select_analog(
        targets, member_values, pool_size=pool_size,
        reuse_penalty_lambda=reuse_penalty_lambda, log_space=log_space, seed=seed,
    )
    selected = members.iloc[template_idx].reset_index(drop=True)

    catalog[f"{driver}_template_member_id"] = selected[member_id_column].astype(str).to_numpy()
    catalog[f"{driver}_template_value"] = member_values[template_idx]
    catalog[f"{driver}_scale_factor"] = scale
    catalog[f"{driver}_design_method"] = design_method
    catalog[f"{driver}_member_id"] = selected[member_id_column].astype(str).to_numpy()
    if member_file_column in members:
        catalog[f"{driver}_member_file"] = selected[member_file_column].astype(str).to_numpy()
    if time_column is not None and time_column in members:
        catalog[f"{driver}_member_time"] = selected[time_column].astype(str).to_numpy()
    catalog[f"{driver}_realization_lag_hours"] = draw_lags(
        len(catalog), observed_lags=observed_lags, default_lag_hours=default_lag_hours, seed=seed + 1
    )
    return catalog


__all__ = [
    "select_analog",
    "select_analog_realization",
    "draw_lags",
    "draw_relative_lags",
    "realize_driver",
    "attach_field_preserving_realization",
    "clip_eps",
]
