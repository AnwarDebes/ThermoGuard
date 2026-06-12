"""Building thermal model: 2R-2C lumped capacitance equivalent circuit.

A residential building is modelled as a two-resistance, two-capacitance
RC equivalent circuit:

    T_outdoor --R_ext-- T_envelope (C_env) --R_int-- T_indoor (C_in)
                                                        ^
                                                        |
                                                    Q_solar + Q_internal

State variables:
    T_env: temperature of the envelope (walls + thermal mass) [degC]
    T_in:  temperature of indoor air [degC]

Parameters per building archetype:
    R_ext: outdoor-to-envelope thermal resistance [degC/W]
    R_int: envelope-to-indoor thermal resistance [degC/W]
    C_env: envelope thermal capacitance [J/degC]
    C_in:  indoor air thermal capacitance [J/degC]
    solar_gain_fraction: fraction of outdoor solar flux that enters [-]
    internal_gain: constant internal heat gain (occupants, appliances) [W]
    AC_setpoint: target indoor temperature when AC active [degC] (None = no AC)
    AC_capacity: cooling rate when AC on [W] (0 = no AC)
    infiltration_rate: air changes per hour [1/h]

Discretisation: forward Euler with timestep dt seconds.

Calibration: when sparse indoor observations are available, parameters
(R_ext, R_int, C_env, C_in, solar_gain_fraction) can be fitted via
least-squares to match observed indoor temperature trajectories.

References:
    ISO 13790:2008 (simple hourly method for energy use in buildings)
    ASHRAE Fundamentals 2021, Chapter 18 (nonresidential cooling/heating)
    Reynders et al. 2014 "Quality of grey-box models" Energy & Buildings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# Physical constants
RHO_AIR = 1.2                       # kg/m^3
CP_AIR = 1005.0                     # J/(kg.degC)
SECONDS_PER_HOUR = 3600.0


@dataclass
class BuildingArchetype:
    """Thermal parameters of a building archetype.

    Default values are representative residential ranges from the building
    physics literature, calibrated against indoor observations from heat-
    wave studies (Quebec 2018; Cambridge MA public housing 2016).
    """
    name: str
    floor_area_m2: float                  # conditioned floor area
    R_ext: float                          # [degC/W] outdoor-to-envelope
    R_int: float                          # [degC/W] envelope-to-indoor
    C_env: float                          # [J/degC] envelope capacitance
    C_in:  float                          # [J/degC] indoor air capacitance
    solar_gain_fraction: float            # [-] portion of solar entering
    internal_gain_w: float                # [W] internal sources
    infiltration_ach: float               # [1/h] air changes per hour
    has_ac: bool                          # AC available?
    ac_setpoint_c: float                  # [degC] target T_in if AC on
    ac_capacity_w: float                  # [W] cooling power
    typical_occupancy: int                # number of residents


# A small library of representative residential archetypes -- vintage,
# construction, insulation, AC status -- spanning the most thermally
# vulnerable populations in temperate climates.
ARCHETYPES = {
    "low_income_pre1980_no_ac": BuildingArchetype(
        name="Pre-1980 low-income apartment, no AC",
        floor_area_m2=55.0,
        R_ext=0.0040, R_int=0.0040,
        C_env=1.2e7, C_in=2.5e5,
        solar_gain_fraction=0.18, internal_gain_w=140.0,
        infiltration_ach=0.9, has_ac=False,
        ac_setpoint_c=24.0, ac_capacity_w=0.0,
        typical_occupancy=2,
    ),
    "midcentury_walkup_window_ac": BuildingArchetype(
        name="Midcentury walkup with window AC",
        floor_area_m2=70.0,
        R_ext=0.0050, R_int=0.0045,
        C_env=1.5e7, C_in=3.0e5,
        solar_gain_fraction=0.16, internal_gain_w=170.0,
        infiltration_ach=0.7, has_ac=True,
        ac_setpoint_c=25.5, ac_capacity_w=2200.0,
        typical_occupancy=2,
    ),
    "modern_well_insulated_central_ac": BuildingArchetype(
        name="Modern well-insulated home with central AC",
        floor_area_m2=140.0,
        R_ext=0.0090, R_int=0.0070,
        C_env=2.5e7, C_in=4.0e5,
        solar_gain_fraction=0.10, internal_gain_w=210.0,
        infiltration_ach=0.35, has_ac=True,
        ac_setpoint_c=23.5, ac_capacity_w=4500.0,
        typical_occupancy=3,
    ),
    "care_home_no_ac": BuildingArchetype(
        name="Care home, top floor, no AC",
        floor_area_m2=95.0,
        R_ext=0.0045, R_int=0.0045,
        C_env=1.3e7, C_in=2.8e5,
        solar_gain_fraction=0.22, internal_gain_w=280.0,
        infiltration_ach=0.5, has_ac=False,
        ac_setpoint_c=24.0, ac_capacity_w=0.0,
        typical_occupancy=4,
    ),
    "tropical_slum_dwelling": BuildingArchetype(
        name="Tropical slum dwelling, no AC, poor ventilation",
        floor_area_m2=22.0,
        R_ext=0.0025, R_int=0.0030,
        C_env=6.0e6, C_in=1.4e5,
        solar_gain_fraction=0.30, internal_gain_w=110.0,
        infiltration_ach=1.4, has_ac=False,
        ac_setpoint_c=24.0, ac_capacity_w=0.0,
        typical_occupancy=4,
    ),
}


@dataclass
class RCBuildingModel:
    """Forward simulator for the 2R-2C indoor-temperature dynamics."""
    archetype: BuildingArchetype

    def step(self, T_env: float, T_in: float,
             T_out: float, solar_w_m2: float,
             dt_s: float) -> Tuple[float, float, float]:
        """Single forward-Euler step.

        Returns updated (T_env, T_in, ac_power_w).
        """
        a = self.archetype
        # Heat flows in [W]
        Q_ext = (T_out - T_env) / a.R_ext
        Q_int = (T_env - T_in) / a.R_int
        Q_solar = a.solar_gain_fraction * solar_w_m2 * a.floor_area_m2 * 0.15
        # ^ rough "south-facing" solar effective area (15% of floor area)
        Q_infil = (a.infiltration_ach / SECONDS_PER_HOUR) * \
                  (RHO_AIR * CP_AIR * a.floor_area_m2 * 2.4) * (T_out - T_in)
        # ^ V_air ≈ floor_area * 2.4m ceiling

        # AC: simple thermostat
        Q_ac = 0.0
        if a.has_ac and T_in > a.ac_setpoint_c:
            # proportional control with full power if delta > 1.0 degC
            Q_ac = -a.ac_capacity_w * min(1.0, T_in - a.ac_setpoint_c)

        # Envelope balance
        dT_env_dt = (Q_ext - Q_int) / a.C_env
        # Indoor balance
        dT_in_dt = (Q_int + Q_solar + a.internal_gain_w + Q_infil + Q_ac) / a.C_in

        T_env_new = T_env + dT_env_dt * dt_s
        T_in_new  = T_in  + dT_in_dt  * dt_s
        return T_env_new, T_in_new, -Q_ac

    def simulate(
        self,
        T_out_series: np.ndarray,           # (n_steps,) outdoor temperature [degC]
        solar_series: np.ndarray,           # (n_steps,) solar flux [W/m^2]
        dt_s: float = 600.0,                # default 10-minute step
        T_env_init: Optional[float] = None,
        T_in_init: Optional[float] = None,
    ) -> dict:
        """Simulate indoor temperature trajectory.

        Returns dict with arrays T_env, T_in, ac_power_w, all of length n_steps.
        """
        T_out_series = np.asarray(T_out_series, dtype=np.float64)
        solar_series = np.asarray(solar_series, dtype=np.float64)
        assert T_out_series.shape == solar_series.shape
        n = len(T_out_series)

        T_env_init = T_out_series[0] - 1.0 if T_env_init is None else T_env_init
        T_in_init  = T_out_series[0] + 1.0 if T_in_init  is None else T_in_init

        T_env = np.zeros(n, dtype=np.float64)
        T_in = np.zeros(n, dtype=np.float64)
        ac_p = np.zeros(n, dtype=np.float64)
        T_env[0], T_in[0] = T_env_init, T_in_init
        for k in range(n - 1):
            new_env, new_in, ac = self.step(
                T_env[k], T_in[k], T_out_series[k], solar_series[k], dt_s,
            )
            T_env[k + 1], T_in[k + 1], ac_p[k + 1] = new_env, new_in, ac
        return {"T_env": T_env, "T_in": T_in, "ac_power_w": ac_p}


def simulate_indoor_temperature(
    archetype_key: str,
    T_out_series: np.ndarray,
    solar_series: np.ndarray,
    dt_s: float = 600.0,
    T_env_init: Optional[float] = None,
    T_in_init: Optional[float] = None,
) -> dict:
    """Convenience wrapper."""
    model = RCBuildingModel(archetype=ARCHETYPES[archetype_key])
    return model.simulate(T_out_series, solar_series, dt_s=dt_s,
                          T_env_init=T_env_init, T_in_init=T_in_init)
