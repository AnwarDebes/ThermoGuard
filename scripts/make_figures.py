"""Generate every figure for the ThermoGuard paper from results JSONs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"
FIGS_DIR = ROOT / "paper" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def load(name):
    return json.loads((RESULTS_DIR / name).read_text())


def fig_archetype_trajectories():
    """Indoor temperature trajectories under PNW 2021 heatwave for all
    archetypes -- the headline 'the problem' figure."""
    from thermoguard import (
        ARCHETYPES, simulate_indoor_temperature, heatwave_pacific_northwest_2021,
    )
    w = heatwave_pacific_northwest_2021(seed=0)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    colors = ["#cf6679", "#cf914e", "#7caf3a", "#a07acf", "#9e6f4c"]
    ax.plot(w.timestamps_hours / 24.0, w.T_out_c, color="black", linewidth=1.2,
            linestyle="--", label="Outdoor")
    for (key, arch), c in zip(ARCHETYPES.items(), colors):
        r = simulate_indoor_temperature(
            key, w.T_out_c, w.solar_w_m2,
            dt_s=w.dt_h * 3600, T_in_init=24.0,
        )
        ax.plot(w.timestamps_hours / 24.0, r["T_in"], color=c,
                label=arch.name, linewidth=1.4)
    ax.axhline(40, color="red", linewidth=0.6, linestyle=":",
               label="Critical (HI 40+)")
    ax.set_xlabel("Time (days from heatwave start)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Indoor temperature trajectories during PNW 2021 heat dome")
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "archetype_trajectories.png", dpi=200)
    fig.savefig(FIGS_DIR / "archetype_trajectories.pdf")
    plt.close(fig)


def fig_forecast_accuracy():
    data = load("forecast_accuracy.json")
    s = data["summary"]
    arch = "low_income_pre1980_no_ac"
    events = ["PNW_2021", "European_2003", "Chicago_1995"]
    models = ["physics", "ml", "hybrid"]
    colors = {"physics": "#7caf3a", "ml": "#cf6679", "hybrid": "#3a7caf"}
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    width = 0.25
    x = np.arange(len(events))
    for i, m in enumerate(models):
        means = [s[f"{arch}/{e}/{m}"]["mae_overall_mean"] for e in events]
        stds  = [s[f"{arch}/{e}/{m}"]["mae_overall_std"]  for e in events]
        ax.bar(x + (i - 1) * width, means, width=width,
               yerr=stds, capsize=3, color=colors[m], label=m.upper())
    ax.set_xticks(x); ax.set_xticklabels(events)
    ax.set_ylabel("Indoor T forecast MAE (°C)")
    ax.set_title(f"Forecast accuracy on {arch}, train day 1-2 / test day 3+")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "forecast_accuracy.png", dpi=200)
    fig.savefig(FIGS_DIR / "forecast_accuracy.pdf")
    plt.close(fig)


def fig_forecast_trajectories():
    """Plot the actual predicted trajectories for one event."""
    data = load("forecast_accuracy.json")
    traj = data.get("trajectories")
    if traj is None:
        return
    t = np.array(traj["timestamps_hours"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(t - t[0], traj["truth"], color="black", linewidth=1.3,
            label="Truth (RC simulation)", linestyle=":")
    ax.plot(t - t[0], traj["physics"], color="#7caf3a", linewidth=1.4,
            label="Physics")
    ax.plot(t - t[0], traj["ml"], color="#cf6679", linewidth=1.4,
            label="ML")
    ax.plot(t - t[0], traj["hybrid"], color="#3a7caf", linewidth=1.4,
            label="Hybrid")
    ax.set_xlabel("Forecast hours")
    ax.set_ylabel("Indoor temperature (°C)")
    ax.set_title("Forecasts on low-income no-AC apartment, PNW 2021 (held-out half)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "forecast_trajectories.png", dpi=200)
    fig.savefig(FIGS_DIR / "forecast_trajectories.pdf")
    plt.close(fig)


def fig_risk_concentration():
    data = load("risk_stratification.json")
    s = data["summary"]
    events = ["PNW_2021", "European_2003", "Chicago_1995"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))

    # Left: mortality per 100k
    ax = axes[0]
    means = [s[e]["per_100k_mean"] for e in events]
    stds  = [s[e]["per_100k_std"]  for e in events]
    colors = ["#3a7caf", "#cf914e", "#cf6679"]
    ax.bar(events, means, yerr=stds, capsize=4, color=colors)
    ax.set_ylabel("Expected excess deaths per 100k residents")
    ax.set_title("Heatwave mortality across reference events")
    for i, v in enumerate(means):
        ax.text(i, v + 50, f"{v:.0f}", ha="center", fontsize=9)

    # Right: concentration (top10 / top25)
    ax = axes[1]
    top10 = [s[e]["top10pct_mean"] for e in events]
    top25 = [s[e]["top25pct_mean"] for e in events]
    width = 0.35
    x = np.arange(len(events))
    ax.bar(x - width/2, top10, width=width, color="#cf6679",
           label="Top 10% of households")
    ax.bar(x + width/2, top25, width=width, color="#3a7caf",
           label="Top 25%")
    ax.set_xticks(x); ax.set_xticklabels(events)
    ax.set_ylabel("Share of total expected deaths")
    ax.set_ylim(0, 1.0)
    ax.set_title("Concentration: how much risk lives in the upper quantiles")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGS_DIR / "risk_concentration.png", dpi=200)
    fig.savefig(FIGS_DIR / "risk_concentration.pdf")
    plt.close(fig)


def fig_triage_tradeoff():
    data = load("triage.json")
    s = data["summary"]
    events = ["PNW_2021", "European_2003", "Chicago_1995"]
    budgets = [(20, 40), (40, 80), (80, 160), (160, 320)]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0))
    for ax, event in zip(axes, events):
        budget_labels = [f"{cv}/{hk}" for cv, hk in budgets]
        greedy_vals = [s[f"{event}/cv={cv}_hk={hk}/greedy"]["lives_saved_mean"]
                       for cv, hk in budgets]
        equity_vals = [s[f"{event}/cv={cv}_hk={hk}/equity_aware"]["lives_saved_mean"]
                       for cv, hk in budgets]
        random_vals = [s[f"{event}/cv={cv}_hk={hk}/random"]["lives_saved_mean"]
                       for cv, hk in budgets]
        x = np.arange(len(budgets))
        ax.plot(x, greedy_vals, "o-", color="#cf6679", label="Greedy (risk-only)")
        ax.plot(x, equity_vals, "s-", color="#3a7caf", label="Equity-aware (floor=0.8)")
        ax.plot(x, random_vals, "x--", color="#7caf3a", label="Random baseline")
        ax.set_xticks(x); ax.set_xticklabels(budget_labels, fontsize=8)
        ax.set_xlabel("Budget (cv / hk)")
        ax.set_ylabel("Expected lives saved")
        ax.set_title(event)
        if event == events[0]:
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Triage tradeoff: efficiency vs equity across heatwave events", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "triage_tradeoff.png", dpi=200)
    fig.savefig(FIGS_DIR / "triage_tradeoff.pdf")
    plt.close(fig)


def fig_architecture():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axis("off")
    boxes = [
        ("Weather forecast\n(T_out, RH, solar)",  0.02, 0.78, 0.20, 0.14, "#cfe6f7"),
        ("Building archetype\nDB (5 types)",      0.02, 0.55, 0.20, 0.14, "#fde2e1"),
        ("Household stock\n(demographics, AC)",   0.02, 0.32, 0.20, 0.14, "#fff5cc"),
        ("2R-2C physics\nrollout",                 0.30, 0.78, 0.18, 0.14, "#dbf3d9"),
        ("Sparse indoor\nobservations",           0.30, 0.55, 0.18, 0.14, "#dbf3d9"),
        ("Hybrid forecaster\n(physics + ridge residual)",
                                                   0.55, 0.66, 0.22, 0.18, "#e7d3ff"),
        ("Heat index +\nvulnerability =\nmortality risk",
                                                   0.55, 0.36, 0.22, 0.18, "#fde2e1"),
        ("Equity-aware\ntriage optimiser",        0.82, 0.50, 0.16, 0.18, "#cfe6f7"),
    ]
    for txt, x, y, w, h, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=c, edgecolor="black",
                                   linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=8.5)
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=1.0))
    arr(0.22, 0.85, 0.30, 0.85)
    arr(0.22, 0.62, 0.30, 0.62)
    arr(0.22, 0.39, 0.55, 0.42)
    arr(0.48, 0.85, 0.55, 0.78)
    arr(0.48, 0.62, 0.55, 0.70)
    arr(0.77, 0.72, 0.82, 0.62)
    arr(0.77, 0.45, 0.82, 0.55)
    arr(0.66, 0.66, 0.66, 0.54)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("ThermoGuard system architecture", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "architecture.png", dpi=200)
    fig.savefig(FIGS_DIR / "architecture.pdf")
    plt.close(fig)


def main():
    fig_architecture()
    fig_archetype_trajectories()
    fig_forecast_accuracy()
    fig_forecast_trajectories()
    fig_risk_concentration()
    fig_triage_tradeoff()
    print(f"Wrote figures to {FIGS_DIR}")


if __name__ == "__main__":
    main()
