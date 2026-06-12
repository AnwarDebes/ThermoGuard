"""Triage optimisation experiment.

For a city of 1500 households under each of three reference heatwaves,
we compare three allocators of a limited daily intervention budget
(cooling-center visits + hydration kits):

    * random_triage:        baseline uniform-random allocation
    * greedy_triage:        risk-only sort, top-K wins
    * equity_aware_triage:  per-stratum equity floor (80% reserved)

Outputs:
    * Total expected lives saved per allocator
    * Stratum coverage (% of vulnerable in each demographic stratum served)
    * Budget vs lives saved tradeoff curve (vary budget)
    * Fairness metric: minimum coverage across strata

The headline finding is the COVERAGE-EQUITY-EFFICIENCY tradeoff:
greedy saves more lives in raw count but concentrates resources in
care homes; equity-aware loses a small fraction of total lives but
serves a broader cross-section of the vulnerable.  In practice the
choice belongs to the public-health authority -- ThermoGuard quantifies
the tradeoff rather than imposing it.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from thermoguard import (
    generate_city, simulate_indoor_temperature,
    heatwave_pacific_northwest_2021, heatwave_european_2003, heatwave_chicago_1995,
    mortality_risk, TriageBudget, greedy_triage, equity_aware_triage,
    random_triage,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_city_risks(city, weather):
    risks = np.zeros(len(city))
    for i, h in enumerate(city):
        sim = simulate_indoor_temperature(
            h.archetype_key, weather.T_out_c, weather.solar_w_m2,
            dt_s=weather.dt_h * 3600, T_in_init=24.0,
        )
        r = mortality_risk(h, sim["T_in"], weather.rh_pct, dt_h=weather.dt_h)
        risks[i] = r["expected_excess_deaths"]
    return risks


def stratum_coverage(allocation, city) -> dict:
    """For each demographic stratum, what fraction of households received
    at least one intervention?"""
    by_stratum = defaultdict(lambda: [0, 0])
    for a, h in zip(allocation, city):
        s = a["stratum"]
        by_stratum[s][1] += 1
        if a["cooling_visit"] or a["hydration_kit"]:
            by_stratum[s][0] += 1
    return {s: served / total if total else 0.0
             for s, (served, total) in by_stratum.items()}


def run_event(event_name, weather, city_size=1500, seed=0,
              budgets=((20, 40), (40, 80), (80, 160), (160, 320))):
    """Run all three allocators across multiple budget levels."""
    city = generate_city(n_households=city_size, seed=seed)
    risks = compute_city_risks(city, weather)
    total_unmitigated = float(risks.sum())

    out = {"event": event_name, "seed": seed,
           "n_households": len(city),
           "total_expected_deaths_no_intervention": total_unmitigated,
           "budgets": []}
    for n_c, n_h in budgets:
        b = TriageBudget(n_cooling_visits=n_c, n_hydration_kits=n_h)
        a_random = random_triage(city, risks, b, seed=seed)
        a_greedy = greedy_triage(city, risks, b)
        a_equity = equity_aware_triage(city, risks, b, equity_floor_fraction=0.8)

        cov_random = stratum_coverage(a_random["allocation"], city)
        cov_greedy = stratum_coverage(a_greedy["allocation"], city)
        cov_equity = stratum_coverage(a_equity["allocation"], city)

        out["budgets"].append({
            "n_cooling_visits": n_c, "n_hydration_kits": n_h,
            "random":       {"lives_saved": a_random["total_lives_saved"],
                              "min_stratum_coverage": min(cov_random.values()) if cov_random else 0.0,
                              "stratum_coverage": cov_random},
            "greedy":       {"lives_saved": a_greedy["total_lives_saved"],
                              "min_stratum_coverage": min(cov_greedy.values()) if cov_greedy else 0.0,
                              "stratum_coverage": cov_greedy},
            "equity_aware": {"lives_saved": a_equity["total_lives_saved"],
                              "min_stratum_coverage": min(cov_equity.values()) if cov_equity else 0.0,
                              "stratum_coverage": cov_equity},
        })
    return out


def main(out_path: Path = RESULTS_DIR / "triage.json",
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
            r = run_event(ev_name, w, city_size=city_size, seed=seed)
            rows.append(r)
            base = r["total_expected_deaths_no_intervention"]
            print(f"  {ev_name:14s} seed={seed} baseline deaths={base:.1f}")
            for b in r["budgets"]:
                line = (f"    budget(cv={b['n_cooling_visits']:3d},"
                        f"hk={b['n_hydration_kits']:3d}): "
                        f"random={b['random']['lives_saved']:5.2f}  "
                        f"greedy={b['greedy']['lives_saved']:5.2f}  "
                        f"equity={b['equity_aware']['lives_saved']:5.2f}  "
                        f"min_cov: greedy={b['greedy']['min_stratum_coverage']:.2f} "
                        f"equity={b['equity_aware']['min_stratum_coverage']:.2f}")
                print(line)
    print(f"\nTotal: {time.time() - overall:.1f}s")

    # Aggregate per (event, budget) -> mean lives saved across seeds
    summary = {}
    for ev_name in events:
        rel = [r for r in rows if r["event"] == ev_name]
        budget_keys = [(b["n_cooling_visits"], b["n_hydration_kits"])
                       for b in rel[0]["budgets"]]
        for bk in budget_keys:
            cv, hk = bk
            for tag in ("random", "greedy", "equity_aware"):
                vals = []
                mins = []
                for r in rel:
                    b = next(b for b in r["budgets"]
                             if (b["n_cooling_visits"], b["n_hydration_kits"]) == bk)
                    vals.append(b[tag]["lives_saved"])
                    mins.append(b[tag]["min_stratum_coverage"])
                summary[f"{ev_name}/cv={cv}_hk={hk}/{tag}"] = {
                    "lives_saved_mean": float(np.mean(vals)),
                    "lives_saved_std":  float(np.std(vals)),
                    "min_stratum_coverage_mean": float(np.mean(mins)),
                }
    print("\n=== Summary ===")
    for k, v in summary.items():
        if k.endswith("greedy") or k.endswith("equity_aware"):
            print(f"  {k}: lives={v['lives_saved_mean']:.2f}+-{v['lives_saved_std']:.2f}  "
                  f"min_cov={v['min_stratum_coverage_mean']:.2f}")

    out_path.write_text(json.dumps({"per_run": rows, "summary": summary}, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
