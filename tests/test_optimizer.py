"""Tests for the CVXPY dispatch optimizer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.battery import BatteryParams
from src.evaluation import evaluate_schedule
from src.optimizer import optimize_dispatch


def _prices_from(values_mwh: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(values_mwh), freq="h")
    p_mwh = np.asarray(values_mwh, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "timestamp_end": idx + pd.Timedelta(hours=1),
            "price_eur_per_mwh": p_mwh,
            "price_eur_per_kwh": p_mwh / 1000.0,
        }
    )


def test_constant_prices_no_cycle():
    # With flat prices, round-trip losses → optimal is idle (value ~0).
    prices = _prices_from([100.0] * 24)
    result = optimize_dispatch(prices)
    metrics = evaluate_schedule(result.schedule, strategy_name="optimized")
    assert result.is_optimal
    assert abs(result.objective_value) < 1e-4
    assert metrics["total_charge_kwh"] < 1e-3
    assert metrics["feasible"]


def test_simple_arbitrage_two_prices():
    # Cheap then expensive: should charge then discharge.
    prices = _prices_from([50.0, 200.0] * 4)  # 8 hours
    batt = BatteryParams()
    result = optimize_dispatch(prices, batt)
    s = result.schedule
    assert result.is_optimal
    assert result.objective_value > 0
    assert s["charge_kwh"].sum() > 0
    assert s["discharge_kwh"].sum() > 0
    # No simultaneous.
    assert not ((s["charge_kwh"] > 1e-8) & (s["discharge_kwh"] > 1e-8)).any()
    metrics = evaluate_schedule(s, batt, strategy_name="optimized")
    assert metrics["feasible"]
    assert abs(metrics["soc_final_kwh"] - batt.final_soc_kwh) < 1e-3


def test_negative_prices_prefer_charge():
    prices = _prices_from([-10.0, -10.0, 150.0, 150.0])
    result = optimize_dispatch(prices)
    s = result.schedule
    assert s.loc[0:1, "charge_kwh"].sum() > 0
    assert result.objective_value > 0


def test_soc_bounds_respected():
    # Long cheap stretch cannot overfill.
    prices = _prices_from([10.0] * 10 + [300.0] * 10)
    batt = BatteryParams(capacity_kwh=100.0, max_charge_kw=50.0, max_discharge_kw=50.0)
    result = optimize_dispatch(prices, batt)
    metrics = evaluate_schedule(result.schedule, batt, strategy_name="optimized")
    assert metrics["soc_max_kwh"] <= batt.capacity_kwh + 1e-5
    assert metrics["soc_min_kwh"] >= -1e-5
    assert metrics["feasible"]


def test_empty_prices_raise():
    prices = _prices_from([])
    with pytest.raises(ValueError):
        optimize_dispatch(prices)
