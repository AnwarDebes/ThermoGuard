"""Equity-aware triage optimizer for limited intervention budgets.

Given:
    * a set of households with predicted per-resident heat-mortality
      risks under no intervention,
    * the marginal effect of each intervention on risk
      (e.g. cooling-center visit reduces risk by 60% during exposure),
    * a finite budget of interventions per day,

we compute an allocation that maximises expected lives saved while
respecting an *equity floor*: every demographic stratum (defined by
income quintile or age band) must receive at least its proportional
share of the budget, unless its risk is dominated.

Three allocators are provided:

* ``greedy_triage``:        risk-only, no fairness consideration.
* ``equity_aware_triage``:  enforces a per-stratum minimum allocation.
* ``random_triage``:        a uniform-random baseline for context.

The intervention effect is parameterised by:
    cooling_visit_effect:  fraction of remaining risk avoided
    hydration_kit_effect:  fraction of remaining risk avoided

We treat the per-household decision as binary (intervene or not).  In a
real deployment intensity could be continuous; we provide the binary
case as the simplest defensible scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .data import Household


@dataclass
class TriageBudget:
    """Daily intervention budget."""
    n_cooling_visits: int = 50
    n_hydration_kits: int = 100
    cooling_visit_effect: float = 0.60       # fraction of remaining risk avoided
    hydration_kit_effect: float = 0.15


def _stratum_key(h: Household) -> str:
    """Income x age-band label for equity floors."""
    age_max = max(h.ages) if h.ages else 0
    if age_max >= 75: age_band = "75+"
    elif age_max >= 65: age_band = "65-74"
    else: age_band = "0-64"
    return f"q{h.income_quintile}_{age_band}"


def expected_lives_saved(
    pre_risk: float, intervened_cooling: bool, intervened_hydration: bool,
    cfg: TriageBudget,
) -> float:
    """Lives saved on a single household = pre_risk - post_risk."""
    remaining = pre_risk
    if intervened_cooling:
        remaining *= (1.0 - cfg.cooling_visit_effect)
    if intervened_hydration:
        remaining *= (1.0 - cfg.hydration_kit_effect)
    return float(pre_risk - remaining)


def greedy_triage(
    households: List[Household],
    pre_risks: np.ndarray,                  # household-level expected excess deaths under no intervention
    budget: TriageBudget,
) -> dict:
    """Risk-only sort, allocate to top-K."""
    order = np.argsort(-pre_risks)
    cooling_set = set(int(i) for i in order[: budget.n_cooling_visits])
    # Hydration kits also go to top-risk but starting after cooling set
    rem = [i for i in order if int(i) not in cooling_set]
    hydration_set = set(int(i) for i in rem[: budget.n_hydration_kits])
    allocation = []
    total_saved = 0.0
    for i, h in enumerate(households):
        c = i in cooling_set; w = i in hydration_set
        s = expected_lives_saved(float(pre_risks[i]), c, w, budget)
        total_saved += s
        allocation.append({"household_id": h.household_id,
                            "cooling_visit": c, "hydration_kit": w,
                            "lives_saved": s, "stratum": _stratum_key(h)})
    return {"allocation": allocation, "total_lives_saved": float(total_saved)}


def equity_aware_triage(
    households: List[Household],
    pre_risks: np.ndarray,
    budget: TriageBudget,
    equity_floor_fraction: float = 0.80,
) -> dict:
    """Greedy-with-equity-floor.

    First reserve ``equity_floor_fraction`` of each resource per
    stratum proportional to that stratum's at-risk population.  Within
    each stratum, allocate by risk descending.  After the reserved
    portion, allocate the remaining ``1 - equity_floor_fraction`` of
    resources greedily by risk.
    """
    strata = [_stratum_key(h) for h in households]
    unique_strata = sorted(set(strata))
    pop_by_stratum = {s: 0 for s in unique_strata}
    for s in strata:
        pop_by_stratum[s] += 1
    total_pop = sum(pop_by_stratum.values())

    # Reserved budgets
    reserved_cooling = {
        s: int(np.round(
            budget.n_cooling_visits * equity_floor_fraction
            * pop_by_stratum[s] / total_pop
        ))
        for s in unique_strata
    }
    reserved_hydration = {
        s: int(np.round(
            budget.n_hydration_kits * equity_floor_fraction
            * pop_by_stratum[s] / total_pop
        ))
        for s in unique_strata
    }

    n = len(households)
    cooling_assigned = np.zeros(n, dtype=bool)
    hydration_assigned = np.zeros(n, dtype=bool)

    # Stratum-by-stratum allocation
    for s in unique_strata:
        idx = [i for i in range(n) if strata[i] == s]
        idx_sorted = sorted(idx, key=lambda i: -pre_risks[i])
        for j in idx_sorted[: reserved_cooling[s]]:
            cooling_assigned[j] = True
        avail = [i for i in idx_sorted if not cooling_assigned[i]]
        for j in avail[: reserved_hydration[s]]:
            hydration_assigned[j] = True

    # Remaining budget allocated globally by risk
    remaining_cooling = budget.n_cooling_visits - int(cooling_assigned.sum())
    remaining_hydration = budget.n_hydration_kits - int(hydration_assigned.sum())
    sorted_global = np.argsort(-pre_risks)
    for i in sorted_global:
        if remaining_cooling <= 0 and remaining_hydration <= 0:
            break
        if not cooling_assigned[i] and remaining_cooling > 0:
            cooling_assigned[i] = True
            remaining_cooling -= 1
            continue
        if not hydration_assigned[i] and remaining_hydration > 0:
            hydration_assigned[i] = True
            remaining_hydration -= 1

    allocation = []
    total_saved = 0.0
    for i, h in enumerate(households):
        c = bool(cooling_assigned[i]); w = bool(hydration_assigned[i])
        s = expected_lives_saved(float(pre_risks[i]), c, w, budget)
        total_saved += s
        allocation.append({"household_id": h.household_id,
                            "cooling_visit": c, "hydration_kit": w,
                            "lives_saved": s, "stratum": _stratum_key(h)})
    return {"allocation": allocation, "total_lives_saved": float(total_saved),
            "reserved_cooling_by_stratum": reserved_cooling,
            "reserved_hydration_by_stratum": reserved_hydration}


def random_triage(
    households: List[Household], pre_risks: np.ndarray,
    budget: TriageBudget, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(households)
    perm = rng.permutation(n)
    cooling_set = set(int(i) for i in perm[: budget.n_cooling_visits])
    rest = [i for i in perm if int(i) not in cooling_set]
    hydration_set = set(int(i) for i in rest[: budget.n_hydration_kits])
    allocation = []
    total_saved = 0.0
    for i, h in enumerate(households):
        c = i in cooling_set; w = i in hydration_set
        s = expected_lives_saved(float(pre_risks[i]), c, w, budget)
        total_saved += s
        allocation.append({"household_id": h.household_id,
                            "cooling_visit": c, "hydration_kit": w,
                            "lives_saved": s, "stratum": _stratum_key(h)})
    return {"allocation": allocation, "total_lives_saved": float(total_saved)}
