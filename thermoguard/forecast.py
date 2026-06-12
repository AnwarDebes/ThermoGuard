"""Hybrid physics-informed forecasters for indoor temperature.

We provide three forecaster families:

* ``PhysicsForecaster``: rolls the 2R-2C model forward using forecast
  outdoor weather only.  No data assimilation.
* ``MLResidualForecaster``: a small linear ridge regressor trained on
  recent (outdoor, indoor) lags + diurnal features.  Pure ML baseline.
* ``HybridForecaster``: runs the physics model, then learns a residual
  correction with ridge regression on lagged (physics_pred, T_out,
  solar, RH, time-of-day).  This is the load-bearing contribution.

All three return a ``ForecastResult`` with the predicted T_in series
at the forecast timestamps, plus the per-horizon error against ground
truth when available.

Why ridge regression rather than a deep network?
    The whole pipeline is meant to run on city-scale (thousands of
    households) every 10 minutes, on the edge.  Ridge is closed-form,
    deterministic, and interpretable.  When ground truth is rich the
    residual learner can be swapped out for a tree ensemble or LSTM
    without changing the architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .building import RCBuildingModel, BuildingArchetype, ARCHETYPES


@dataclass
class ForecastResult:
    timestamps_hours: np.ndarray              # forecast timestamps (hours)
    T_in_pred: np.ndarray                     # predicted indoor temp (degC)
    T_in_truth: Optional[np.ndarray]          # ground truth if known (degC)

    def per_horizon_mae(self, horizons_hours=(1, 6, 12, 24)) -> dict:
        """Mean absolute error grouped by forecast horizon."""
        if self.T_in_truth is None:
            return {}
        out = {}
        for h in horizons_hours:
            # take all timestamps within [h-0.5, h+0.5] hour relative
            # to the issue time (assumed timestamps[0])
            offset = self.timestamps_hours - self.timestamps_hours[0]
            mask = (offset >= h - 0.5) & (offset < h + 0.5)
            if not mask.any():
                continue
            out[f"mae_h{h}"] = float(
                np.mean(np.abs(self.T_in_pred[mask] - self.T_in_truth[mask]))
            )
        out["mae_overall"] = float(
            np.mean(np.abs(self.T_in_pred - self.T_in_truth))
        )
        return out


class PhysicsForecaster:
    """Pure 2R-2C physics rollout from a known/assumed initial state."""

    def __init__(self, archetype: BuildingArchetype):
        self.model = RCBuildingModel(archetype)

    def forecast(
        self,
        T_out_future: np.ndarray,
        solar_future: np.ndarray,
        dt_s: float,
        T_in_init: float, T_env_init: float,
        T_in_truth: Optional[np.ndarray] = None,
        timestamps_hours: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        result = self.model.simulate(
            T_out_future, solar_future, dt_s=dt_s,
            T_env_init=T_env_init, T_in_init=T_in_init,
        )
        n = len(T_out_future)
        if timestamps_hours is None:
            timestamps_hours = np.arange(n) * (dt_s / 3600.0)
        return ForecastResult(
            timestamps_hours=timestamps_hours,
            T_in_pred=result["T_in"],
            T_in_truth=T_in_truth,
        )


def _diurnal_features(timestamps_hours: np.ndarray) -> np.ndarray:
    """Sin/cos features of hour-of-day."""
    hod = np.mod(timestamps_hours, 24.0)
    s = np.sin(2 * np.pi * hod / 24.0)
    c = np.cos(2 * np.pi * hod / 24.0)
    return np.stack([s, c], axis=1)


class _RidgeRegressor:
    """Tiny closed-form L2-regularised linear regression."""
    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.w: Optional[np.ndarray] = None       # (n_features + 1,) including bias

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RidgeRegressor":
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        A = X1.T @ X1 + self.alpha * np.eye(X1.shape[1])
        b = X1.T @ y
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("call fit first")
        X1 = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
        return X1 @ self.w


class MLResidualForecaster:
    """Pure-ML baseline: ridge regression on lagged features.

    Features used to predict ``T_in[t]`` (target):
        * T_out[t], T_out[t-1], T_out[t-2], T_out[t-6]
        * solar[t], solar[t-1], solar[t-3]
        * RH[t]
        * T_in[t-1], T_in[t-3]   (autoregressive)
        * sin(hour), cos(hour)
    """
    LAGS_OUT = [0, 1, 2, 6]
    LAGS_SOLAR = [0, 1, 3]
    LAGS_IN = [1, 3]

    def __init__(self, alpha: float = 1e-2):
        self.ridge = _RidgeRegressor(alpha=alpha)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    @staticmethod
    def _build_features(
        T_out: np.ndarray, solar: np.ndarray, rh: np.ndarray,
        T_in: np.ndarray, timestamps_h: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns X, y, valid_indices.  Drops the first ``max_lag`` rows."""
        max_lag = max(MLResidualForecaster.LAGS_OUT
                       + MLResidualForecaster.LAGS_SOLAR
                       + MLResidualForecaster.LAGS_IN)
        n = len(T_out)
        idx = np.arange(max_lag, n)
        cols = []
        for L in MLResidualForecaster.LAGS_OUT:
            cols.append(T_out[idx - L])
        for L in MLResidualForecaster.LAGS_SOLAR:
            cols.append(solar[idx - L])
        cols.append(rh[idx])
        for L in MLResidualForecaster.LAGS_IN:
            cols.append(T_in[idx - L])
        diurnal = _diurnal_features(timestamps_h[idx])
        X = np.stack(cols, axis=1).astype(np.float64)
        X = np.concatenate([X, diurnal], axis=1)
        y = T_in[idx]
        return X, y, idx

    def fit(self, T_out: np.ndarray, solar: np.ndarray, rh: np.ndarray,
            T_in: np.ndarray, timestamps_h: np.ndarray) -> "MLResidualForecaster":
        X, y, _ = self._build_features(T_out, solar, rh, T_in, timestamps_h)
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-9
        Xn = (X - self.feature_means) / self.feature_stds
        self.ridge.fit(Xn, y)
        return self

    def forecast(
        self,
        T_out_future: np.ndarray, solar_future: np.ndarray,
        rh_future: np.ndarray, T_in_init_window: np.ndarray,
        timestamps_h: np.ndarray,
        T_in_truth: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        """Recursive 1-step-ahead rollout starting from a warmed-up history."""
        max_lag = max(self.LAGS_OUT + self.LAGS_SOLAR + self.LAGS_IN)
        n = len(T_out_future)
        T_in_pred = np.zeros(n, dtype=np.float64)
        # we need a history buffer of length max_lag for T_in
        hist = np.concatenate([T_in_init_window[-max_lag:], np.zeros(n)])
        # extend covariates with history too
        T_out_hist = np.concatenate([T_in_init_window[-max_lag:],   # placeholder
                                      T_out_future])
        # actually, the right thing is to know history T_out, solar, rh too.
        # We expect callers to provide them prepended onto the future series.
        for t in range(n):
            global_t = max_lag + t
            cols = []
            for L in self.LAGS_OUT:
                cols.append(T_out_future[t - L] if (t - L) >= 0 else T_in_init_window[L - t - 1])
            for L in self.LAGS_SOLAR:
                cols.append(solar_future[t - L] if (t - L) >= 0 else 0.0)
            cols.append(rh_future[t])
            for L in self.LAGS_IN:
                cols.append(hist[global_t - L])
            X_row = np.array(cols + [
                np.sin(2 * np.pi * (timestamps_h[t] % 24) / 24),
                np.cos(2 * np.pi * (timestamps_h[t] % 24) / 24),
            ], dtype=np.float64)
            Xn = (X_row - self.feature_means) / self.feature_stds
            yhat = float(self.ridge.predict(Xn[None, :])[0])
            T_in_pred[t] = yhat
            hist[global_t] = yhat
        return ForecastResult(timestamps_hours=timestamps_h,
                              T_in_pred=T_in_pred, T_in_truth=T_in_truth)


class HybridForecaster:
    """Physics backbone + ridge residual correction.

    Training:
        1. Run physics model on the calibration period using known weather +
           initial state.  Get ``T_in_phys``.
        2. Compute residual r[t] = T_in_truth[t] - T_in_phys[t].
        3. Fit ridge regression: r[t] ~ phi(T_out[t], solar[t], rh[t],
           T_in_phys[t], lagged residuals, diurnal features).

    Forecasting:
        1. Roll physics forward over the forecast horizon -> ``T_in_phys``.
        2. Predict residual sequentially with the ridge model.
        3. Return T_in_pred = T_in_phys + residual_hat.
    """
    LAGS_RES = [1, 2, 3]

    def __init__(self, archetype: BuildingArchetype, alpha: float = 1e-2):
        self.physics = PhysicsForecaster(archetype)
        self.ridge = _RidgeRegressor(alpha=alpha)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

    def _features(self, T_out, solar, rh, T_in_phys, residual_hist,
                  timestamps_h, t_idx):
        cols = [
            T_out[t_idx], solar[t_idx], rh[t_idx],
            T_in_phys[t_idx], T_in_phys[max(0, t_idx - 1)],
        ]
        for L in self.LAGS_RES:
            cols.append(residual_hist[max(0, t_idx - L)])
        cols.append(np.sin(2 * np.pi * (timestamps_h[t_idx] % 24) / 24))
        cols.append(np.cos(2 * np.pi * (timestamps_h[t_idx] % 24) / 24))
        return np.array(cols, dtype=np.float64)

    def fit(
        self, T_out: np.ndarray, solar: np.ndarray, rh: np.ndarray,
        T_in_truth: np.ndarray, dt_s: float,
        timestamps_h: np.ndarray,
        T_in_init: Optional[float] = None, T_env_init: Optional[float] = None,
    ) -> "HybridForecaster":
        # Physics rollout over training window
        if T_in_init is None: T_in_init = float(T_in_truth[0])
        if T_env_init is None: T_env_init = float(T_in_truth[0]) - 1.0
        phys = self.physics.model.simulate(
            T_out, solar, dt_s=dt_s,
            T_env_init=T_env_init, T_in_init=T_in_init,
        )
        T_in_phys = phys["T_in"]
        residual = T_in_truth - T_in_phys
        n = len(T_out)
        max_lag = max(self.LAGS_RES)
        rows = []; targets = []
        for t in range(max_lag, n):
            rows.append(self._features(T_out, solar, rh, T_in_phys, residual,
                                        timestamps_h, t))
            targets.append(residual[t])
        X = np.stack(rows)
        y = np.array(targets, dtype=np.float64)
        self.feature_means = X.mean(axis=0)
        self.feature_stds = X.std(axis=0) + 1e-9
        Xn = (X - self.feature_means) / self.feature_stds
        self.ridge.fit(Xn, y)
        return self

    def forecast(
        self, T_out_future: np.ndarray, solar_future: np.ndarray,
        rh_future: np.ndarray, dt_s: float,
        T_in_init: float, T_env_init: float,
        timestamps_h: np.ndarray,
        T_in_truth: Optional[np.ndarray] = None,
        residual_history: Optional[np.ndarray] = None,
    ) -> ForecastResult:
        # Run physics forward
        phys = self.physics.model.simulate(
            T_out_future, solar_future, dt_s=dt_s,
            T_env_init=T_env_init, T_in_init=T_in_init,
        )
        T_in_phys = phys["T_in"]
        n = len(T_out_future)
        # Recursive residual prediction
        residual_hist = np.zeros(n + 10, dtype=np.float64)
        if residual_history is not None:
            k = min(len(residual_history), 10)
            residual_hist[:k] = residual_history[-k:]
            offset = 10
        else:
            offset = 0
        residual_pred = np.zeros(n, dtype=np.float64)
        for t in range(n):
            feat = self._features(T_out_future, solar_future, rh_future,
                                   T_in_phys, residual_hist[offset:offset + n],
                                   timestamps_h, t)
            featn = (feat - self.feature_means) / self.feature_stds
            r_hat = float(self.ridge.predict(featn[None, :])[0])
            residual_pred[t] = r_hat
            residual_hist[offset + t] = r_hat
        T_in_pred = T_in_phys + residual_pred
        return ForecastResult(
            timestamps_hours=timestamps_h,
            T_in_pred=T_in_pred, T_in_truth=T_in_truth,
        )
