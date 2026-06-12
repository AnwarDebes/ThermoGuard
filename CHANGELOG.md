# Changelog

All notable changes to ThermoGuard are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-12

Reference implementation accompanying the paper.

### Added

- 2R-2C lumped-capacitance building thermal model with five residential
  archetypes spanning the documented vulnerability stratification.
- Synthetic heatwave templates reproducing PNW 2021, EU 2003, and
  Chicago 1995, plus a synthetic city generator with a configurable
  vulnerability mix.
- Three indoor-temperature forecasters: pure physics, pure ML, and a
  hybrid (physics RC backbone + closed-form ridge residual learner).
- Per-resident heat-mortality risk model: Steadman heat index, Stull
  WBGT, age, chronic conditions, mobility, AC access, social isolation,
  income quintile.
- Triage allocators (random, greedy, equity-aware with a per-stratum
  coverage floor) for limited cooling-visit and hydration-kit budgets.
- Three reproducible experiments (forecast accuracy, risk
  stratification, triage tradeoff) writing raw records to `results/`.
- Figure-generation script, LaTeX paper with figures, 9 unit tests.

[0.1.0]: https://github.com/AnwarDebes/ThermoGuard/releases/tag/v0.1.0
