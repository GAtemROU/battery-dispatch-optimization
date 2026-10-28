"""Tests for day-ahead rolling and forecast-then-optimize strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.battery import BatteryParams
from src.evaluation import evaluate_schedule
from src.strategies import (
    day_ahead_rolling_strategy,
    forecast_optimize_strategy,
    holdout_split,
    seasonal_naive_hour_forecast,
)


def _diurnal_prices(n_days: int = 20, start: str = "2026-08-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days * 24, freq="h")
    hour = idx.hour
    # Stable shape + mild day index drift so forecast ≠ perfect.
    day_i = np.arange(len(idx)) // 24
    p = 100 + 50 * np.sin((hour - 5) / 24 * 2 * np.pi) + 3.0 * (day_i % 5)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "timestamp_end": idx + pd.Timedelta(hours=1),
            "price_eur_per_mwh": p,
            "price_eur_per_kwh": p / 1000.0,
        }
    )


def test_holdout_split_day_aligned():
    prices = _diurnal_prices(10)
    warm, test, meta = holdout_split(prices, frac=0.5)
    assert meta["n_days_warm"] == 5
    assert meta["n_days_test"] == 5
    assert len(warm) + len(test) == len(prices)
    assert warm["timestamp"].max() < test["timestamp"].min()


def test_day_ahead_rolling_feasible():
    prices = _diurnal_prices(6)
    batt = BatteryParams()
    sched = day_ahead_rolling_strategy(prices, batt)
    m = evaluate_schedule(sched, batt, strategy_name="day_ahead_rolling")
    assert m["feasible"]
    assert abs(m["soc_final_kwh"] - batt.final_soc_kwh) < 1e-3
    assert sched.attrs["strategy"] == "day_ahead_rolling"


def test_forecast_no_leakage_and_feasible():
    prices = _diurnal_prices(16)
    warm, test, _ = holdout_split(prices, frac=0.5)
    batt = BatteryParams()
    sched = forecast_optimize_strategy(test, batt, history=warm, lookback_days=7)
    m = evaluate_schedule(sched, batt, strategy_name="forecast_optimize")
    assert m["feasible"]
    assert len(sched) == len(test)

    # Leakage check: mutating a future actual day must not change earlier actions.
    test2 = test.copy()
    # Bump last day's actual prices massively.
    last_day = pd.to_datetime(test2["timestamp"]).dt.floor("D").max()
    mask = pd.to_datetime(test2["timestamp"]).dt.floor("D") == last_day
    test2.loc[mask, "price_eur_per_kwh"] = test2.loc[mask, "price_eur_per_kwh"] + 1.0
    test2.loc[mask, "price_eur_per_mwh"] = test2.loc[mask, "price_eur_per_kwh"] * 1000.0

    sched2 = forecast_optimize_strategy(test2, batt, history=warm, lookback_days=7)
    # All hours before last day: charge/discharge identical.
    pre = ~mask.to_numpy()
    np.testing.assert_allclose(
        sched.loc[pre, "charge_kwh"].to_numpy(),
        sched2.loc[pre, "charge_kwh"].to_numpy(),
        atol=1e-8,
    )
    np.testing.assert_allclose(
        sched.loc[pre, "discharge_kwh"].to_numpy(),
        sched2.loc[pre, "discharge_kwh"].to_numpy(),
        atol=1e-8,
    )


def test_seasonal_naive_uses_only_past():
    prices = _diurnal_prices(10)
    warm, test, _ = holdout_split(prices, frac=0.5)
    day0 = test.iloc[:24]
    fc = seasonal_naive_hour_forecast(warm, day0, lookback_days=7)
    assert len(fc) == 24
    assert fc["price_eur_per_kwh"].notna().all()
