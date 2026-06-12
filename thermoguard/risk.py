"""Heat-mortality risk model: physiology + vulnerability.

Two physiological metrics:

* **Heat Index (NWS / Steadman 1979)**: apparent temperature combining
  air temperature and relative humidity, validated to predict heat
  illness up to ~60 degC.
* **Simplified Wet Bulb Globe Temperature (WBGT)**: a heat-stress
  metric used by occupational health.  We use the indoor-shade variant
  WBGT_in = 0.7 T_wet + 0.3 T_dry.

Vulnerability score combines:
    * Age band (75+ has 5-10x mortality risk; 65-74 has 2-3x)
    * Chronic conditions (cardiovascular, respiratory, diabetes)
    * Mobility limitation (can't relocate to cooler space)
    * AC access and ability to afford running it
    * Social isolation (no one to check in, alert delays)

Per-resident hourly mortality probability is a sigmoid of the
exposure-vulnerability product.  Calibrated so a typical heatwave
exposure (~36 degC indoor for 48h, elderly, no AC, isolated)
produces ~2% excess mortality -- consistent with documented
epidemiological excess.

References:
    Steadman 1979 J. Applied Meteorology 18(7)
    Astrom et al. 2011 Maturitas 69(2) -- excess mortality vs heat
    Gronlund 2014 Curr Epidemiology Rep. -- vulnerability factors
    Bouchama et al. 2007 Arch Intern Med 167(20) -- isolation 2x risk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .data import Household


def heat_index(T_air_c: float, rh_pct: float) -> float:
    """NWS / Steadman 1979 simplified formula for apparent temperature.

    Valid for T_air >= 27 degC, RH >= 40%.  Outside that we return T_air
    (no apparent-temperature elevation expected).
    """
    if T_air_c < 27.0 or rh_pct < 40.0:
        return float(T_air_c)
    T = T_air_c * 9.0 / 5.0 + 32.0          # convert to F for formula
    R = rh_pct
    HI_F = (
        -42.379 + 2.04901523 * T + 10.14333127 * R
        - 0.22475541 * T * R - 6.83783e-3 * T**2
        - 5.481717e-2 * R**2 + 1.22874e-3 * T**2 * R
        + 8.5282e-4 * T * R**2 - 1.99e-6 * T**2 * R**2
    )
    return float((HI_F - 32.0) * 5.0 / 9.0)


def wbgt_simple(T_air_c: float, rh_pct: float) -> float:
    """Indoor-shade WBGT approximation.

    T_wet (psychrometric) by Stull 2011 inverse formula:
        T_wet = T atan(0.151977*sqrt(RH + 8.313659))
                + atan(T + RH) - atan(RH - 1.676331)
                + 0.00391838*RH^1.5 * atan(0.023101*RH) - 4.686035
    """
    T = float(T_air_c); R = float(rh_pct)
    T_wet = (
        T * np.arctan(0.151977 * np.sqrt(R + 8.313659))
        + np.arctan(T + R) - np.arctan(R - 1.676331)
        + 0.00391838 * (R ** 1.5) * np.arctan(0.023101 * R)
        - 4.686035
    )
    return float(0.7 * T_wet + 0.3 * T)


def vulnerability_score(h: Household) -> np.ndarray:
    """Returns one vulnerability multiplier per resident (length = n_residents)."""
    out = []
    for i in range(h.n_residents):
        age = h.ages[i]
        chronic = h.chronic_conditions[i]
        mobility = h.mobility_limited[i]
        # Age band multiplier
        if age >= 75:
            v = 6.0
        elif age >= 65:
            v = 2.5
        elif age >= 55:
            v = 1.3
        else:
            v = 1.0
        if chronic:
            v *= 2.0
        if mobility:
            v *= 1.5
        if not h.has_any_cooling:
            v *= 1.8
        if h.is_socially_isolated:
            v *= 2.0
        if h.income_quintile == 1:
            v *= 1.3
        elif h.income_quintile == 2:
            v *= 1.15
        out.append(v)
    return np.array(out, dtype=np.float64)


@dataclass
class MortalityRiskParams:
    """Hyperparameters of the per-resident hourly mortality model.

    The hourly mortality probability is sigmoid((HI - threshold)
    * sensitivity + log_vulnerability) clipped to a small constant.

    Defaults calibrated so a "typical" heatwave exposure
    (HI 36-40 degC for 48 h, elderly, no AC, isolated) reproduces the
    documented 1-3% all-cause excess mortality during PNW 2021 in
    that demographic stratum.
    """
    hi_threshold_c: float = 32.0            # heat-index baseline (no excess below)
    sensitivity_per_degc: float = 0.18      # logistic slope per degC of HI
    baseline_hourly_mortality: float = 1e-6 # background hazard
    max_hourly_mortality: float = 5e-4      # cap


def per_resident_hourly_hazard(
    HI_c: float, v: float, params: MortalityRiskParams,
) -> float:
    """Per-resident excess hourly mortality probability."""
    if HI_c <= params.hi_threshold_c:
        return params.baseline_hourly_mortality
    excess = HI_c - params.hi_threshold_c
    log_v = np.log(max(v, 1e-3))
    logit = excess * params.sensitivity_per_degc + log_v - 4.5
    p_excess = 1.0 / (1.0 + np.exp(-logit))
    p = params.baseline_hourly_mortality + p_excess * params.max_hourly_mortality
    return float(min(p, params.max_hourly_mortality))


def mortality_risk(
    h: Household,
    T_in_series: np.ndarray, rh_series: np.ndarray,
    dt_h: float,
    params: MortalityRiskParams = MortalityRiskParams(),
) -> dict:
    """Compute expected mortality across all residents for a household
    over a forecast / observed indoor-temperature trajectory.

    Returns dict with per-resident exposure summary and total expected
    deaths.
    """
    vulns = vulnerability_score(h)
    # Apparent temperature -- assume indoor RH ~= outdoor RH (no AC) or 50%
    # (AC running drops it; for simplicity we shift slightly for AC).
    if h.has_any_cooling:
        rh_indoor = np.minimum(rh_series, 50.0)
    else:
        rh_indoor = rh_series.copy()
    HI_series = np.array(
        [heat_index(float(T_in_series[k]), float(rh_indoor[k]))
         for k in range(len(T_in_series))], dtype=np.float64,
    )
    total_expected_deaths = 0.0
    per_res = []
    for vi in vulns:
        hazards = np.array(
            [per_resident_hourly_hazard(float(HI_series[k]), float(vi), params)
             for k in range(len(HI_series))], dtype=np.float64,
        )
        # convert per-hour hazard to per-step hazard, then sum (with small dt this
        # approximates 1 - exp(-integral))
        step_hazard = 1.0 - np.exp(-hazards * dt_h)
        # Probability the resident dies sometime in the window
        survival = np.prod(1.0 - step_hazard)
        p_death = 1.0 - survival
        total_expected_deaths += p_death
        per_res.append({"vulnerability": float(vi),
                         "p_death": float(p_death),
                         "max_HI": float(HI_series.max())})
    return {
        "household_id": h.household_id,
        "archetype": h.archetype_key,
        "n_residents": h.n_residents,
        "max_indoor_T": float(T_in_series.max()),
        "max_HI": float(HI_series.max()),
        "expected_excess_deaths": float(total_expected_deaths),
        "per_resident": per_res,
    }
