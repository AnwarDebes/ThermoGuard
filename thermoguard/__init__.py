"""ThermoGuard: Real-Time Indoor Heat Risk Forecasting for Vulnerable Housing.

A hybrid physics + machine-learning system for predicting indoor temperature
trajectories in residential buildings under heatwave conditions, integrating
demographic vulnerability and AC access into a mortality-risk model and an
equity-aware triage optimiser for limited intervention budgets.
"""

from .building import BuildingArchetype, RCBuildingModel, ARCHETYPES, simulate_indoor_temperature
from .weather import (
    WeatherSegment, generate_heatwave, generate_normal_summer,
    heatwave_pacific_northwest_2021, heatwave_european_2003, heatwave_chicago_1995,
)
from .data import Household, generate_housing_stock, generate_city
from .forecast import (
    PhysicsForecaster, MLResidualForecaster, HybridForecaster, ForecastResult,
)
from .risk import (
    heat_index, wbgt_simple, vulnerability_score,
    mortality_risk, MortalityRiskParams, per_resident_hourly_hazard,
)
from .triage import (
    TriageBudget, greedy_triage, equity_aware_triage, random_triage,
    expected_lives_saved,
)

__version__ = "0.1.0"

__all__ = [
    "BuildingArchetype", "RCBuildingModel", "ARCHETYPES", "simulate_indoor_temperature",
    "WeatherSegment", "generate_heatwave", "generate_normal_summer",
    "heatwave_pacific_northwest_2021", "heatwave_european_2003", "heatwave_chicago_1995",
    "Household", "generate_housing_stock", "generate_city",
    "PhysicsForecaster", "MLResidualForecaster", "HybridForecaster", "ForecastResult",
    "heat_index", "wbgt_simple", "vulnerability_score",
    "mortality_risk", "MortalityRiskParams", "per_resident_hourly_hazard",
    "TriageBudget", "greedy_triage", "equity_aware_triage", "random_triage",
    "expected_lives_saved",
]
