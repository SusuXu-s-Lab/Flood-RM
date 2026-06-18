"""Field-Preserving Realization bridge: sampled Driver Probability Index -> observed field.

Layer 2 of the two-layer framework. The copula stage samples scalar driver indices
(target rainfall depth, water-level peak, ...). This bridge maps each target to an
*observed* member analog plus a scalar scale factor and timing lag, so the physical
forcing is the real spatio-temporal field (AORC SST rainfall ``netampr``, an observed
hydrograph) scaled to the target — never a uniform scalar. It is the multivariate,
copula-driven generalization of the analog+scale realization already used for coastal
water level (``hydrographs.build_surge_event_members``) and inland streamflow
(``inland_event_catalog._design_streamflow_members``): select a nearby observed event
(kernel-weighted, with a reuse penalty for diversity) and scale its field by
``K = target / observed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

clip_eps = 1e-12


def select_analog_realization(
    targets,
    member_values,
    *,
    pool_size=75,
    sigma_scale=0.5,
    sigma_min=1e-6,
    sigma_max=np.inf,
    reuse_penalty_lambda=0.15,
    log_space=True,
    seed=0,
):
    """Pick an observed member analog and scale factor for each target index value.

    For each target, restrict to the ``pool_size`` nearest observed members, draw one
    by Gaussian distance weights times a reuse penalty (diversity), and scale its field
    by ``target / member_value``. Mirrors ``build_surge_event_members``. ``log_space``
    measures distance in log space (suited to heavy-tailed rainfall/discharge);
    set ``False`` for near-zero-bounded drivers such as water level.

    Returns ``(template_index, scale_factor)`` arrays aligned with ``targets``.
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


def draw_relative_lags(n, observed_lags=None, *, default_lag_hours=0.0, seed=0):
    """Draw per-event timing lags from observed inter-driver lags.

    Preserves realistic relative timing between driver peaks instead of forcing them to
    coincide. Falls back to a constant lag when no observed lags are supplied.
    """
    if observed_lags is None or len(observed_lags) == 0:
        return np.full(int(n), float(default_lag_hours), dtype=float)
    rng = np.random.default_rng(int(seed))
    pool = np.asarray(observed_lags, dtype=float)
    pool = pool[np.isfinite(pool)]
    if pool.size == 0:
        return np.full(int(n), float(default_lag_hours), dtype=float)
    return rng.choice(pool, size=int(n), replace=True)


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
    """Attach ``<driver>_*`` realization columns mapping each target index to an observed field.

    Each catalog row's ``target_column`` (a sampled Driver Probability Index) is matched
    to an observed member analog whose ``member_id``/``member_file`` point to the real
    spatio-temporal field, plus a scalar ``scale_factor`` and timing ``lag``. The field
    itself is staged (and scaled by ``scale_factor``) downstream by the SFINCS/Wflow
    forcing step, so spatial and temporal structure is preserved by construction.
    """
    catalog = catalog.copy()
    members = members.reset_index(drop=True)
    if index_column not in members:
        raise ValueError(f"members missing index column {index_column!r}")
    if member_id_column not in members:
        raise ValueError(f"members missing member id column {member_id_column!r}")

    targets = pd.to_numeric(catalog[target_column], errors="coerce").to_numpy(dtype=float)
    member_values = pd.to_numeric(members[index_column], errors="coerce").to_numpy(dtype=float)
    template_idx, scale = select_analog_realization(
        targets,
        member_values,
        pool_size=pool_size,
        reuse_penalty_lambda=reuse_penalty_lambda,
        log_space=log_space,
        seed=seed,
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
    catalog[f"{driver}_realization_lag_hours"] = draw_relative_lags(
        len(catalog), observed_lags=observed_lags, default_lag_hours=default_lag_hours, seed=seed + 1
    )
    return catalog
