"""Synthetic weather streams: heatwave events, normal summers, and
diurnal solar/temperature patterns.

Heatwave templates mimic real events:
    * Pacific Northwest 2021 dome: 40-46 degC daytime peaks for 3 days
    * European 2003: prolonged 35-40 degC, low humidity, 2-3 weeks
    * Chicago 1995: 38-40 degC daytime, high humidity, ~5 days

We synthesise outdoor temperature, relative humidity, and solar flux
at 10-minute resolution.

For reproducibility we use deterministic templates parameterised by
peak temperature and event length plus small Gaussian noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class WeatherSegment:
    """A time-aligned weather stream."""
    timestamps_hours: np.ndarray      # hours since start (e.g. 0, 1/6, 2/6, ...)
    T_out_c: np.ndarray               # outdoor temperature [degC]
    rh_pct: np.ndarray                # outdoor relative humidity [%]
    solar_w_m2: np.ndarray            # global horizontal solar flux [W/m^2]
    dt_h: float                       # timestep in hours


def _diurnal_temperature(t_hours: np.ndarray, t_min: float, t_max: float) -> np.ndarray:
    """Sinusoidal diurnal cycle peaking at 15:00 local."""
    phase = 2 * np.pi * (t_hours - 9.0) / 24.0   # peak at 15:00 (9 hours after sunrise=6)
    return (t_min + t_max) / 2 + (t_max - t_min) / 2 * np.sin(phase)


def _diurnal_solar(t_hours: np.ndarray, peak_w_m2: float = 850.0) -> np.ndarray:
    """Sinusoidal solar flux, zero at night."""
    hours_of_day = np.mod(t_hours, 24.0)
    flux = np.maximum(0.0, peak_w_m2 * np.sin(np.pi * (hours_of_day - 6.0) / 12.0))
    return flux


def _diurnal_humidity(t_hours: np.ndarray, t_out: np.ndarray,
                      rh_mean: float = 55.0) -> np.ndarray:
    """RH anti-correlated with temperature, mean rh_mean."""
    span = np.ptp(t_out)
    if span < 1e-6:
        return np.full_like(t_out, rh_mean)
    norm = (t_out - t_out.min()) / span        # 0..1
    return np.clip(rh_mean + 25.0 - 35.0 * norm, 15.0, 95.0)


def generate_heatwave(
    n_days: int = 5,
    peak_temp_c: float = 42.0,
    night_temp_c: float = 28.0,
    rh_mean: float = 35.0,
    solar_peak_w_m2: float = 900.0,
    dt_h: float = 1 / 6,
    noise_std_c: float = 0.4,
    seed: int = 0,
) -> WeatherSegment:
    """Multi-day extreme-heat event template.

    Defaults mimic Pacific NW 2021 dome (high peak, low humidity).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(n_days * 24 / dt_h)
    t = np.arange(n_steps) * dt_h                       # hours since start

    # Ramp-up across day 1, plateau days 2..n-1, ramp-down day n
    T_out = _diurnal_temperature(t, night_temp_c, peak_temp_c)
    # Day-to-day amplification factor: build up for the first 2 days, plateau, slight decay
    day_idx = np.minimum(t // 24, n_days - 1)
    amp = 1.0 + 0.10 * np.where(day_idx < 2, day_idx, 2 - 0.1 * (day_idx - 2))
    centre = (night_temp_c + peak_temp_c) / 2
    T_out = centre + amp * (T_out - centre)
    T_out += rng.normal(0, noise_std_c, n_steps)
    rh = _diurnal_humidity(t, T_out, rh_mean=rh_mean)
    solar = _diurnal_solar(t, peak_w_m2=solar_peak_w_m2)
    return WeatherSegment(timestamps_hours=t, T_out_c=T_out, rh_pct=rh,
                          solar_w_m2=solar, dt_h=dt_h)


def generate_normal_summer(
    n_days: int = 5,
    high_temp_c: float = 28.0,
    night_temp_c: float = 18.0,
    rh_mean: float = 60.0,
    dt_h: float = 1 / 6,
    noise_std_c: float = 0.4,
    seed: int = 0,
) -> WeatherSegment:
    """Baseline non-extreme summer week."""
    rng = np.random.default_rng(seed)
    n_steps = int(n_days * 24 / dt_h)
    t = np.arange(n_steps) * dt_h
    T_out = _diurnal_temperature(t, night_temp_c, high_temp_c) + \
        rng.normal(0, noise_std_c, n_steps)
    rh = _diurnal_humidity(t, T_out, rh_mean=rh_mean)
    solar = _diurnal_solar(t, peak_w_m2=820.0)
    return WeatherSegment(timestamps_hours=t, T_out_c=T_out, rh_pct=rh,
                          solar_w_m2=solar, dt_h=dt_h)


def heatwave_pacific_northwest_2021(seed: int = 0) -> WeatherSegment:
    """Reference template for the June 2021 Pacific Northwest heat dome
    (Portland OR / Seattle WA): 42-46 degC peaks, low humidity, 3 days."""
    return generate_heatwave(
        n_days=4, peak_temp_c=44.0, night_temp_c=27.0,
        rh_mean=30.0, solar_peak_w_m2=920.0, seed=seed,
    )


def heatwave_european_2003(seed: int = 0) -> WeatherSegment:
    """Reference template for the August 2003 European heatwave
    (Paris): 38-40 degC peaks, moderate humidity, prolonged."""
    return generate_heatwave(
        n_days=10, peak_temp_c=39.0, night_temp_c=23.0,
        rh_mean=45.0, solar_peak_w_m2=860.0, seed=seed,
    )


def heatwave_chicago_1995(seed: int = 0) -> WeatherSegment:
    """Reference template for the July 1995 Chicago heatwave:
    38-40 degC peaks, very high humidity (>50%), 5 days."""
    return generate_heatwave(
        n_days=5, peak_temp_c=39.5, night_temp_c=27.0,
        rh_mean=65.0, solar_peak_w_m2=880.0, seed=seed,
    )
