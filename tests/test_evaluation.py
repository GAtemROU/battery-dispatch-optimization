"""Tests for baselines and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.battery import BatteryParams
from src.evaluation import comparison_table, evaluate_schedule
from src.strategies import clock_window_strategy, idle_strategy


def _toy_prices(n: int = 48) -> pd.DataFrame:
    idx = pd.date_range("2026-08-28", periods=n, freq="h")
    # Synthetic diurnal: low midday, high evening.
    hour = idx.hour
    p = 100 + 40 * np.sin((hour - 4) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "timestamp_end": idx + pd.Timedelta(hours=1),
            "price_eur_per_mwh": p,
            "price_eur_per_kwh": p / 1000.0,
        }
    )


def test_idle_zero_value():
    prices = _toy_prices()
    s = idle_strategy(prices)
    m = evaluate_schedule(s, strategy_name="idle")
    assert m["total_arbitrage_eur"] == 0.0
    assert m["feasible"]
    assert m["total_charge_kwh"] == 0.0


def test_clock_window_feasible():
    prices = _toy_prices(24 * 5)
    batt = BatteryParams()
    s = clock_window_strategy(prices, batt)
    m = evaluate_schedule(s, batt, strategy_name="clock_window")
    assert m["feasible"]
    assert m["violations"]["simultaneous_charge_discharge"] == 0
    assert abs(m["soc_final_kwh"] - batt.final_soc_kwh) < 1e-3


def test_comparison_table():
    prices = _toy_prices(48)
    metrics = [
        evaluate_schedule(idle_strategy(prices), strategy_name="idle"),
        evaluate_schedule(clock_window_strategy(prices), strategy_name="clock_window"),
    ]
    table = comparison_table(metrics)
    assert list(table["strategy"]) == ["idle", "clock_window"]
    assert table.loc[0, "arbitrage_eur"] == 0.0
