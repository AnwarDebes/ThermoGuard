"""Synthetic city of households for heatwave risk simulation.

A ``Household`` carries:
    * its building archetype (mapping to thermal physics)
    * resident demographics: ages, chronic conditions, mobility
    * AC access and operating status (working / broken / unaffordable)
    * social isolation flag (lives alone, no daily contact)
    * neighbourhood location (lat, lon) for spatial aggregation

The synthetic city distribution reflects real epidemiological findings:
    * Elderly (75+) account for >80% of heat mortality (CDC 2024)
    * Low-income households have 2-3x higher mortality (Lancet 2023)
    * Care-home residents have the highest per-capita risk
    * Pre-1980 housing without AC is the highest-risk archetype
    * Social isolation roughly doubles risk (NEJM 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from .building import ARCHETYPES, BuildingArchetype


@dataclass
class Household:
    """A single residential unit."""
    household_id: int
    archetype_key: str
    # Demographics
    n_residents: int
    ages: List[int]                       # one age per resident
    chronic_conditions: List[bool]        # one flag per resident
    mobility_limited: List[bool]
    # Social and built-environment
    is_socially_isolated: bool
    has_working_ac: bool                  # may differ from archetype's "has_ac"
    cannot_afford_ac_use: bool            # can't afford to RUN the AC
    income_quintile: int                  # 1 (lowest) ... 5 (highest)
    # Location
    lat: float
    lon: float

    @property
    def archetype(self) -> BuildingArchetype:
        return ARCHETYPES[self.archetype_key]

    @property
    def has_any_cooling(self) -> bool:
        return self.has_working_ac and not self.cannot_afford_ac_use


def generate_housing_stock(
    n_households: int = 1000,
    seed: int = 0,
    elderly_fraction: float = 0.18,             # share of households with >=1 person 75+
    care_home_fraction: float = 0.03,
    no_ac_fraction: float = 0.30,
    affordability_failure_fraction: float = 0.10,   # of households with AC, share who can't afford to run it
    social_isolation_among_elderly: float = 0.35,
    centre_lat: float = 45.52, centre_lon: float = -122.68,   # Portland OR default
    spread_km: float = 8.0,
) -> List[Household]:
    """Generate a realistic stock with the documented vulnerability mix."""
    rng = np.random.default_rng(seed)
    households: List[Household] = []
    keys = list(ARCHETYPES.keys())

    for i in range(n_households):
        # Pick archetype, biased by elderly / care-home shares
        r = rng.random()
        if r < care_home_fraction:
            key = "care_home_no_ac"
        elif r < care_home_fraction + 0.20:
            key = "low_income_pre1980_no_ac"
        elif r < care_home_fraction + 0.45:
            key = "midcentury_walkup_window_ac"
        elif r < care_home_fraction + 0.75:
            key = "modern_well_insulated_central_ac"
        else:
            key = "tropical_slum_dwelling"

        arch = ARCHETYPES[key]
        n_res = arch.typical_occupancy
        # If care home, draw more residents
        if key == "care_home_no_ac":
            n_res = int(rng.integers(3, 7))

        # Ages: elderly bias
        ages: List[int] = []
        has_elderly = rng.random() < elderly_fraction or key == "care_home_no_ac"
        for _ in range(n_res):
            if has_elderly and rng.random() < 0.45:
                ages.append(int(rng.integers(75, 96)))
            elif has_elderly and rng.random() < 0.25:
                ages.append(int(rng.integers(65, 75)))
            else:
                ages.append(int(rng.integers(20, 70)))

        chronic = [rng.random() < (0.05 + 0.5 * max(0, a - 65) / 30) for a in ages]
        mobility = [rng.random() < (0.02 + 0.4 * max(0, a - 70) / 25) for a in ages]

        # AC: depends on archetype + affordability
        has_working_ac = arch.has_ac and rng.random() > 0.05      # 5% broken
        cannot_afford = has_working_ac and (
            rng.random() < affordability_failure_fraction
        )

        # Income quintile correlated with archetype
        if key in ("low_income_pre1980_no_ac", "tropical_slum_dwelling"):
            q = int(rng.integers(1, 3))
        elif key == "midcentury_walkup_window_ac":
            q = int(rng.integers(2, 4))
        elif key == "care_home_no_ac":
            q = int(rng.integers(1, 4))
        else:
            q = int(rng.integers(3, 6))

        # Override no-AC base fraction (some elderly are in nominally AC-equipped
        # units but lose service or can't afford)
        if rng.random() < no_ac_fraction and key != "modern_well_insulated_central_ac":
            has_working_ac = False
            cannot_afford = False

        is_isolated = has_elderly and (rng.random() < social_isolation_among_elderly)

        # Location: roughly Gaussian around centre
        # 1 degree latitude ~= 111 km
        dlat = rng.normal(0, spread_km / 111.0)
        dlon = rng.normal(0, spread_km / (111.0 * np.cos(np.radians(centre_lat))))

        households.append(Household(
            household_id=i,
            archetype_key=key,
            n_residents=n_res,
            ages=ages,
            chronic_conditions=chronic,
            mobility_limited=mobility,
            is_socially_isolated=is_isolated,
            has_working_ac=has_working_ac,
            cannot_afford_ac_use=cannot_afford,
            income_quintile=q,
            lat=centre_lat + dlat,
            lon=centre_lon + dlon,
        ))
    return households


def generate_city(n_households: int = 1500, seed: int = 0) -> List[Household]:
    """Default city: 1500 households, Portland OR centre, vulnerability mix
    aligned with documented heatwave epidemiology."""
    return generate_housing_stock(n_households=n_households, seed=seed)
