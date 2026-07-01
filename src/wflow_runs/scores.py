from __future__ import annotations

import numpy as np
import pandas as pd

from wflow_runs.observations import event_iv_records, observed_event_site_flow
from wflow_runs.output import gauge_output_map, read_wflow_event_output_csv


def hydrograph_scores(simulated: pd.Series, observed: pd.Series) -> dict:
    joined = pd.concat([simulated.rename("sim"), observed.rename("obs")], axis=1).dropna()
    if joined.empty:
        return {"n": 0, "nse": np.nan, "kge": np.nan, "peak_bias_fraction": np.nan, "volume_bias_fraction": np.nan}
    obs = joined["obs"].to_numpy(dtype=float)
    sim = joined["sim"].to_numpy(dtype=float)
    denom = np.sum((obs - obs.mean()) ** 2)
    nse = 1.0 - np.sum((sim - obs) ** 2) / denom if denom else np.nan
    r = np.corrcoef(sim, obs)[0, 1] if len(joined) > 1 and np.std(sim) and np.std(obs) else np.nan
    alpha = np.std(sim) / np.std(obs) if np.std(obs) else np.nan
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) else np.nan
    kge = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2) if np.isfinite([r, alpha, beta]).all() else np.nan
    return {
        "n": int(len(joined)),
        "nse": float(nse) if np.isfinite(nse) else np.nan,
        "kge": float(kge) if np.isfinite(kge) else np.nan,
        "peak_bias_fraction": float((sim.max() - obs.max()) / obs.max()) if obs.max() else np.nan,
        "volume_bias_fraction": float((sim.sum() - obs.sum()) / obs.sum()) if obs.sum() else np.nan,
    }


def usgs_calibration_table(event_id, *, events_root, wflow_base_root, event_streamflow_iv_root, submodel_id=None) -> pd.DataFrame:
    sim = read_wflow_event_output_csv(event_id, events_root=events_root, submodel_id=submodel_id)
    gauges = gauge_output_map(event_id, events_root=events_root, wflow_base_root=wflow_base_root, layer="gauges_usgs", submodel_id=submodel_id)
    records = event_iv_records(event_id, event_streamflow_iv_root)
    if sim.empty or gauges.empty or records.empty:
        return pd.DataFrame()
    rows = []
    for _, gauge in gauges.iterrows():
        q_col = str(gauge["q_column"])
        if q_col not in sim:
            continue
        obs = observed_event_site_flow(event_id, gauge["site_no"], sim.index, event_streamflow_iv_root=event_streamflow_iv_root)
        scores = hydrograph_scores(sim[q_col].astype(float), obs)
        if scores["n"] > 0:
            rows.append({"event_id": event_id, "site_no": str(gauge["site_no"]), "q_column": q_col, **scores})
    return pd.DataFrame(rows)
