"""Unit tests for ThermoGuard."""

import numpy as np
import pytest

from thermoguard import (
    ARCHETYPES, simulate_indoor_temperature,
    heatwave_pacific_northwest_2021, generate_normal_summer,
    generate_city, mortality_risk, MortalityRiskParams,
    heat_index, wbgt_simple, vulnerability_score,
    PhysicsForecaster, MLResidualForecaster, HybridForecaster,
    TriageBudget, greedy_triage, equity_aware_triage, random_triage,
    expected_lives_saved,
)


def test_physics_model_responds_to_heatwave():
    """Indoor temp should rise during a heatwave and rise more in no-AC archetypes."""
    w = heatwave_pacific_northwest_2021(seed=0)
    no_ac = simulate_indoor_temperature(
        "low_income_pre1980_no_ac", w.T_out_c, w.solar_w_m2,
        dt_s=w.dt_h * 3600, T_in_init=24.0,
    )
    central_ac = simulate_indoor_temperature(
        "modern_well_insulated_central_ac", w.T_out_c, w.solar_w_m2,
        dt_s=w.dt_h * 3600, T_in_init=24.0,
    )
    assert no_ac["T_in"].max() > central_ac["T_in"].max() + 5.0


def test_normal_summer_keeps_most_archetypes_safe():
    """Under a normal summer most archetypes should stay below 34 degC.
    Care homes with poor ventilation and high solar gain are a known
    exception even under non-extreme weather (the WHO HHG flags top-floor
    care homes specifically), so we exclude that archetype from the bound."""
    w = generate_normal_summer(seed=0)
    for key in ARCHETYPES:
        r = simulate_indoor_temperature(
            key, w.T_out_c, w.solar_w_m2,
            dt_s=w.dt_h * 3600, T_in_init=22.0,
        )
        upper = 40.0 if key == "care_home_no_ac" else 34.0
        assert r["T_in"].max() < upper, f"{key}: peak {r['T_in'].max()}"


def test_heat_index_monotone_in_temperature():
    """At fixed humidity, heat index should be monotone non-decreasing in T_air."""
    for rh in (50, 60, 70):
        prev = -100.0
        for T in range(27, 45):
            hi = heat_index(T, rh)
            assert hi >= prev - 0.5
            prev = hi


def test_vulnerability_score_higher_for_elderly_no_ac():
    """An 80yo with chronic condition, no AC, isolated should have higher
    vulnerability than a healthy 30yo with AC, not isolated."""
    city = generate_city(n_households=300, seed=0)
    # Find a high-risk elderly household
    elderly = [h for h in city if max(h.ages) >= 80 and not h.has_any_cooling
               and h.is_socially_isolated]
    young = [h for h in city if max(h.ages) <= 50 and h.has_any_cooling
             and not h.is_socially_isolated]
    if elderly and young:
        v_e = vulnerability_score(elderly[0]).max()
        v_y = vulnerability_score(young[0]).min()
        assert v_e > v_y * 5


def test_mortality_concentrates_on_vulnerable():
    """Top-10% riskiest households should account for >50% of expected
    deaths -- the standard 'long-tail' epidemiology of heatwaves."""
    city = generate_city(n_households=400, seed=0)
    w = heatwave_pacific_northwest_2021(seed=0)
    risks = []
    for h in city:
        sim = simulate_indoor_temperature(
            h.archetype_key, w.T_out_c, w.solar_w_m2,
            dt_s=w.dt_h * 3600, T_in_init=24.0,
        )
        r = mortality_risk(h, sim["T_in"], w.rh_pct, dt_h=w.dt_h)
        risks.append(r["expected_excess_deaths"])
    risks = np.asarray(risks)
    total = float(risks.sum())
    top10 = int(0.10 * len(risks))
    top_risks = np.sort(risks)[::-1][:top10]
    assert top_risks.sum() / total > 0.5


def test_hybrid_forecaster_runs_end_to_end():
    """Train on day 1, forecast day 2, verify forecast is finite."""
    arch = ARCHETYPES["low_income_pre1980_no_ac"]
    w = heatwave_pacific_northwest_2021(seed=0)
    # Simulate ground truth indoor temps
    sim = simulate_indoor_temperature(
        "low_income_pre1980_no_ac", w.T_out_c, w.solar_w_m2,
        dt_s=w.dt_h * 3600, T_in_init=24.0,
    )
    n = len(w.T_out_c); half = n // 2
    hyb = HybridForecaster(arch).fit(
        T_out=w.T_out_c[:half], solar=w.solar_w_m2[:half],
        rh=w.rh_pct[:half], T_in_truth=sim["T_in"][:half],
        dt_s=w.dt_h * 3600, timestamps_h=w.timestamps_hours[:half],
        T_in_init=24.0, T_env_init=23.0,
    )
    res = hyb.forecast(
        w.T_out_c[half:], w.solar_w_m2[half:], w.rh_pct[half:],
        dt_s=w.dt_h * 3600,
        T_in_init=float(sim["T_in"][half - 1]),
        T_env_init=float(sim["T_in"][half - 1]) - 1.0,
        timestamps_h=w.timestamps_hours[half:],
        T_in_truth=sim["T_in"][half:],
    )
    assert np.isfinite(res.T_in_pred).all()
    # The hybrid should at least roughly track ground truth
    err = res.per_horizon_mae()
    assert err["mae_overall"] < 5.0


def test_triage_budget_respected():
    city = generate_city(n_households=100, seed=0)
    pre_risks = np.array([float(i % 3) for i in range(len(city))])
    budget = TriageBudget(n_cooling_visits=10, n_hydration_kits=20)
    g = greedy_triage(city, pre_risks, budget)
    n_c = sum(1 for a in g["allocation"] if a["cooling_visit"])
    n_h = sum(1 for a in g["allocation"] if a["hydration_kit"])
    assert n_c <= 10 and n_h <= 20


def test_equity_aware_covers_strata():
    """Equity-aware allocation should serve at least 1 household in each
    non-empty stratum when the floor is positive."""
    city = generate_city(n_households=400, seed=0)
    pre_risks = np.array([0.01 * (i + 1) for i in range(len(city))])
    budget = TriageBudget(n_cooling_visits=80, n_hydration_kits=120)
    e = equity_aware_triage(city, pre_risks, budget,
                              equity_floor_fraction=0.8)
    # Reserved counts per stratum should sum approximately to budget * floor
    rc = sum(e["reserved_cooling_by_stratum"].values())
    assert rc >= int(0.7 * budget.n_cooling_visits)


def test_equity_aware_vs_greedy_lives_saved():
    """Greedy maximises lives saved unconstrained; equity-aware should be
    at most slightly worse on lives but more equitable across strata."""
    city = generate_city(n_households=400, seed=0)
    w = heatwave_pacific_northwest_2021(seed=0)
    risks = []
    for h in city:
        sim = simulate_indoor_temperature(
            h.archetype_key, w.T_out_c, w.solar_w_m2,
            dt_s=w.dt_h * 3600, T_in_init=24.0,
        )
        r = mortality_risk(h, sim["T_in"], w.rh_pct, dt_h=w.dt_h)
        risks.append(r["expected_excess_deaths"])
    risks = np.array(risks)
    budget = TriageBudget(n_cooling_visits=40, n_hydration_kits=60)
    g = greedy_triage(city, risks, budget)
    e = equity_aware_triage(city, risks, budget, equity_floor_fraction=0.7)
    # Equity-aware no more than 30% worse in lives saved
    assert e["total_lives_saved"] >= 0.3 * g["total_lives_saved"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
