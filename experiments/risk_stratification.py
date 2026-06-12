"""Heatwave risk-stratification experiment.

For each of three reference heatwaves (PNW 2021, EU 2003, Chicago 1995)
we instantiate the same synthetic city of 1500 households, simulate
indoor temperatures with the 2R-2C physics model per archetype, and
compute per-household expected excess deaths.

We then report:
    * Total expected excess deaths
    * Concentration: the % of deaths attributable to the top-N% riskiest
      households (the long-tail signature of heatwave epidemiology)
    * Per-archetype contribution
    * Per-demographic-stratum contribution (income x age band)

The first metric is the validation point: documented heatwave excess
mortality in cities resembling Portland or Chicago is roughly
0.3-1.5% of the affected population.  Our synthetic city should
reproduce a number in this range under the historical templates.

The second metric (concentration) is the principal scientific finding:
documented heat mortality is HIGHLY concentrated -- typically >70% of
deaths come from <10% of households (CDC 2024 -- elderly + no AC +
chronic conditions).  This is what makes targeted triage possible.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from thermoguard import (
    ARCHETYPES, generate_city, simulate_indoor_temperature,
    heatwave_pacific_northwest_2021, heatwave_european_2003, heatwave_chicago_1995,
    mortality_risk, vulnerability_score,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_event(event_name: str, weather, city_size: int = 1500,
              seed: int = 0) -> dict:
    city = generate_city(n_households=city_size, seed=seed)
    risks = np.zeros(len(city), dtype=np.float64)
    archetype_breakdown = defaultdict(float)
    stratum_breakdown = defaultdict(float)

    for i, h in enumerate(city):
        sim = simulate_indoor_temperature(
            h.archetype_key, weather.T_out_c, weather.solar_w_m2,
            dt_s=weather.dt_h * 3600, T_in_init=24.0,
        )
        r = mortality_risk(h, sim["T_in"], weather.rh_pct, dt_h=weather.dt_h)
        risks[i] = r["expected_excess_deaths"]
        archetype_breakdown[h.archetype_key] += r["expected_excess_deaths"]
        # Stratum: income quintile x age band
        age_max = max(h.ages); aband = "75+" if age_max >= 75 else ("65-74" if age_max >= 65 else "0-64")
        stratum_breakdown[f"q{h.income_quintile}_{aband}"] += r["expected_excess_deaths"]

    total = float(risks.sum())
    sorted_risks = np.sort(risks)[::-1]
    cum = np.cumsum(sorted_risks)
    concentration = {}
    for pct in (1, 5, 10, 25):
        k = max(1, int(pct / 100 * len(risks)))
        concentration[f"top_{pct}pct"] = float(cum[k - 1] / total) if total > 0 else 0.0

    total_residents = sum(h.n_residents for h in city)
    return {
        "event": event_name,
        "seed": seed,
        "n_households": len(city),
        "n_residents": int(total_residents),
        "total_expected_deaths": float(total),
        "excess_mortality_per_100k_residents": float(total / total_residents * 1e5),
        "concentration": concentration,
        "by_archetype": dict(archetype_breakdown),
        "by_stratum": dict(stratum_breakdown),
    }


def main(out_path: Path = RESULTS_DIR / "risk_stratification.json",
         seeds=(0, 1, 2), city_size=1500):
    events = {
        "PNW_2021": heatwave_pacific_northwest_2021,
        "European_2003": heatwave_european_2003,
        "Chicago_1995": heatwave_chicago_1995,
    }
    rows = []
    overall = time.time()
    for ev_name, ev_fn in events.items():
        for seed in seeds:
            w = ev_fn(seed=seed)
            row = run_event(ev_name, w, city_size=city_size, seed=seed)
            rows.append(row)
            print(f"  {ev_name:14s} seed={seed} "
                  f"total deaths={row['total_expected_deaths']:.1f}  "
                  f"per 100k={row['excess_mortality_per_100k_residents']:.1f}  "
                  f"top10%={row['concentration']['top_10pct']:.2f}  "
                  f"top25%={row['concentration']['top_25pct']:.2f}")
    print(f"\nTotal: {time.time() - overall:.1f}s")

    # Aggregate
    summary = {}
    for ev_name in events:
        rel = [r for r in rows if r["event"] == ev_name]
        totals = np.array([r["total_expected_deaths"] for r in rel])
        per100k = np.array([r["excess_mortality_per_100k_residents"] for r in rel])
        c10 = np.array([r["concentration"]["top_10pct"] for r in rel])
        c25 = np.array([r["concentration"]["top_25pct"] for r in rel])
        summary[ev_name] = {
            "total_expected_deaths_mean": float(totals.mean()),
            "total_expected_deaths_std":  float(totals.std()),
            "per_100k_mean": float(per100k.mean()),
            "per_100k_std":  float(per100k.std()),
            "top10pct_mean": float(c10.mean()),
            "top25pct_mean": float(c25.mean()),
        }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: deaths={v['total_expected_deaths_mean']:.1f}+-{v['total_expected_deaths_std']:.1f}  "
              f"per100k={v['per_100k_mean']:.0f}+-{v['per_100k_std']:.0f}  "
              f"top10%={v['top10pct_mean']:.2f}  top25%={v['top25pct_mean']:.2f}")

    out_path.write_text(json.dumps({"per_run": rows, "summary": summary}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
