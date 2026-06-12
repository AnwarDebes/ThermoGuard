"""Forecast accuracy: physics-only vs ML-only vs hybrid.

We simulate ground-truth indoor temperatures for a representative
vulnerable archetype (low-income pre-1980, no AC) under each of three
historical heatwave templates, then evaluate three forecasters at
horizons 1 h, 6 h, 12 h, 24 h.

Training window: first 50% of the event (early observation period
when the heatwave begins to ramp up; this is when a HHWS must produce
forecasts).
Test window: last 50%.

The hybrid forecaster is expected to dominate at long horizons because
(a) the physics backbone respects energy conservation, eliminating the
long-horizon drift that affects pure-ML, while (b) the ML residual
captures occupant behaviour, opportunistic ventilation, and weather
features the physics model does not directly include.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from thermoguard import (
    ARCHETYPES, simulate_indoor_temperature,
    heatwave_pacific_northwest_2021, heatwave_european_2003, heatwave_chicago_1995,
    PhysicsForecaster, MLResidualForecaster, HybridForecaster,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_one_event(event_name: str, weather, archetype_key: str,
                  observation_noise_std: float = 0.5, seed: int = 0) -> dict:
    """Returns per-horizon errors for each forecaster on this event."""
    rng = np.random.default_rng(seed)
    arch = ARCHETYPES[archetype_key]
    dt_s = weather.dt_h * 3600

    # Ground truth
    truth = simulate_indoor_temperature(
        archetype_key, weather.T_out_c, weather.solar_w_m2,
        dt_s=dt_s, T_in_init=24.0,
    )
    T_in_truth = truth["T_in"] + rng.normal(0, observation_noise_std,
                                              size=truth["T_in"].shape)
    # Train / test
    n = len(weather.T_out_c); half = n // 2

    physics = PhysicsForecaster(arch)
    res_phys = physics.forecast(
        weather.T_out_c[half:], weather.solar_w_m2[half:], dt_s=dt_s,
        T_in_init=float(T_in_truth[half - 1]), T_env_init=float(T_in_truth[half - 1]) - 1.0,
        timestamps_hours=weather.timestamps_hours[half:],
        T_in_truth=truth["T_in"][half:],
    )

    ml = MLResidualForecaster(alpha=0.5).fit(
        T_out=weather.T_out_c[:half], solar=weather.solar_w_m2[:half],
        rh=weather.rh_pct[:half], T_in=T_in_truth[:half],
        timestamps_h=weather.timestamps_hours[:half],
    )
    res_ml = ml.forecast(
        T_out_future=weather.T_out_c[half:], solar_future=weather.solar_w_m2[half:],
        rh_future=weather.rh_pct[half:],
        T_in_init_window=T_in_truth[:half],
        timestamps_h=weather.timestamps_hours[half:],
        T_in_truth=truth["T_in"][half:],
    )

    hyb = HybridForecaster(arch, alpha=0.5).fit(
        T_out=weather.T_out_c[:half], solar=weather.solar_w_m2[:half],
        rh=weather.rh_pct[:half], T_in_truth=T_in_truth[:half], dt_s=dt_s,
        timestamps_h=weather.timestamps_hours[:half],
        T_in_init=float(T_in_truth[0]), T_env_init=float(T_in_truth[0]) - 1.0,
    )
    res_hyb = hyb.forecast(
        T_out_future=weather.T_out_c[half:], solar_future=weather.solar_w_m2[half:],
        rh_future=weather.rh_pct[half:], dt_s=dt_s,
        T_in_init=float(T_in_truth[half - 1]),
        T_env_init=float(T_in_truth[half - 1]) - 1.0,
        timestamps_h=weather.timestamps_hours[half:],
        T_in_truth=truth["T_in"][half:],
    )

    horizons = (1, 6, 12, 24)
    return {
        "event": event_name,
        "archetype": archetype_key,
        "physics": res_phys.per_horizon_mae(horizons),
        "ml":      res_ml.per_horizon_mae(horizons),
        "hybrid":  res_hyb.per_horizon_mae(horizons),
        "trajectories": {
            "timestamps_hours": weather.timestamps_hours[half:].tolist(),
            "truth": truth["T_in"][half:].tolist(),
            "physics": res_phys.T_in_pred.tolist(),
            "ml":      res_ml.T_in_pred.tolist(),
            "hybrid":  res_hyb.T_in_pred.tolist(),
        },
    }


def main(out_path: Path = RESULTS_DIR / "forecast_accuracy.json",
         seeds=(0, 1, 2), archetypes=("low_income_pre1980_no_ac",
                                       "care_home_no_ac",
                                       "midcentury_walkup_window_ac")):
    events = {
        "PNW_2021": heatwave_pacific_northwest_2021,
        "European_2003": heatwave_european_2003,
        "Chicago_1995": heatwave_chicago_1995,
    }
    rows = []
    overall = time.time()
    for arch in archetypes:
        for ev_name, ev_fn in events.items():
            for seed in seeds:
                w = ev_fn(seed=seed)
                row = run_one_event(ev_name, w, arch, seed=seed)
                row["seed"] = seed
                rows.append(row)
                phys_mae = row["physics"].get("mae_overall", float("nan"))
                ml_mae = row["ml"].get("mae_overall", float("nan"))
                hyb_mae = row["hybrid"].get("mae_overall", float("nan"))
                print(f"  arch={arch[:18]:18s} ev={ev_name:14s} seed={seed} "
                      f"MAE physics={phys_mae:.2f}  ml={ml_mae:.2f}  hybrid={hyb_mae:.2f}")
    print(f"\nTotal: {time.time() - overall:.1f}s")

    # Aggregate by (archetype, event, model) -> mean MAE at each horizon
    summary = {}
    for arch in archetypes:
        for ev_name in events:
            rel = [r for r in rows if r["archetype"] == arch and r["event"] == ev_name]
            for model in ("physics", "ml", "hybrid"):
                key = f"{arch}/{ev_name}/{model}"
                horizons = {}
                for h in (1, 6, 12, 24):
                    vals = [r[model].get(f"mae_h{h}", np.nan) for r in rel]
                    vals = [v for v in vals if np.isfinite(v)]
                    if vals:
                        horizons[f"mae_h{h}_mean"] = float(np.mean(vals))
                        horizons[f"mae_h{h}_std"]  = float(np.std(vals))
                overall_vals = [r[model].get("mae_overall", np.nan) for r in rel]
                horizons["mae_overall_mean"] = float(np.mean(overall_vals))
                horizons["mae_overall_std"]  = float(np.std(overall_vals))
                summary[key] = horizons

    print("\n=== Summary (mean MAE over 3 seeds, degC) ===")
    for arch in archetypes:
        print(f"  {arch}:")
        for ev_name in events:
            line = f"    {ev_name:14s} "
            for model in ("physics", "ml", "hybrid"):
                key = f"{arch}/{ev_name}/{model}"
                m = summary[key]["mae_overall_mean"]
                line += f"{model}={m:.2f}  "
            print(line)

    # Strip the (large) trajectories from the saved JSON; save only the per-event
    # forecasts of one archetype for figure plotting.
    saved = []
    for r in rows:
        record = {k: v for k, v in r.items() if k != "trajectories"}
        saved.append(record)
    # Keep one set of trajectories for the figure: low_income_pre1980_no_ac PNW_2021 seed 0
    keep = next((r for r in rows
                  if r["archetype"] == "low_income_pre1980_no_ac"
                  and r["event"] == "PNW_2021"
                  and r["seed"] == 0), None)
    out_path.write_text(json.dumps({
        "per_run": saved, "summary": summary,
        "trajectories": keep["trajectories"] if keep else None,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
