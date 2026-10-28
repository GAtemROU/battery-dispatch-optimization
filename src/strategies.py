"""Baseline dispatch strategies (idle + rule-based)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.battery import DEFAULT_BATTERY, BatteryParams


@dataclass(frozen=True)
class ClockWindowHeuristic:
    """Charge/discharge on fixed local-clock windows from price EDA.

    Default windows match the Aug–Oct 2026 DE sample shape:
    - midday trough ~12–15h
    - evening peak ~18–21h
    """

    charge_hours: tuple[int, ...] = (12, 13, 14, 15)
    discharge_hours: tuple[int, ...] = (18, 19, 20, 21)
    name: str = "clock_window"


def _empty_schedule(prices: pd.DataFrame, battery: BatteryParams) -> pd.DataFrame:
    n = len(prices)
    soc0 = battery.initial_soc_kwh
    t_end = (
        prices["timestamp_end"].to_numpy()
        if "timestamp_end" in prices.columns
        else np.array([pd.NaT] * n)
    )
    p = prices["price_eur_per_kwh"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "timestamp": prices["timestamp"].to_numpy(),
            "timestamp_end": t_end,
            "price_eur_per_kwh": p,
            "price_eur_per_mwh": p * 1000.0,
            "charge_kwh": np.zeros(n),
            "discharge_kwh": np.zeros(n),
            "soc_start_kwh": np.full(n, soc0),
            "soc_end_kwh": np.full(n, soc0),
            "is_charging_mode": np.zeros(n, dtype=bool),
            "arbitrage_eur": np.zeros(n),
        }
    )


def idle_strategy(
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
) -> pd.DataFrame:
    """Baseline A: battery inactive; arbitrage value is zero."""
    battery = battery or DEFAULT_BATTERY
    if "price_eur_per_kwh" not in prices.columns:
        raise ValueError("prices must include price_eur_per_kwh")
    schedule = _empty_schedule(prices.reset_index(drop=True), battery)
    schedule.attrs["strategy"] = "idle"
    return schedule


def clock_window_strategy(
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
    heuristic: ClockWindowHeuristic | None = None,
) -> pd.DataFrame:
    """Baseline B: fixed clock-hour charge/discharge windows.

    Rules (deterministic)
    ---------------------
    1. Walk hours in chronological order.
    2. In charge hours: charge at max power if SOC room (efficiency-aware).
    3. In discharge hours: discharge at max power if SOC available.
    4. Never charge and discharge the same hour (windows are disjoint).
    5. Outside windows: idle.
    6. After the forward pass, restore final SOC == initial SOC by a
       second chronological pass that trims/adds residual energy using
       remaining headroom (charge more or discharge more where windows
       allow; if still short, use any hour with room as last resort for
       feasibility — residual is usually small).

    Step 6 keeps the comparison fair with the optimizer's terminal SOC
    constraint. Residual repair is documented via schedule attrs.
    """
    battery = battery or DEFAULT_BATTERY
    heuristic = heuristic or ClockWindowHeuristic()
    df = prices.reset_index(drop=True)
    if "price_eur_per_kwh" not in df.columns or "timestamp" not in df.columns:
        raise ValueError("prices need timestamp and price_eur_per_kwh")

    charge_set = set(heuristic.charge_hours)
    discharge_set = set(heuristic.discharge_hours)
    if charge_set & discharge_set:
        raise ValueError("charge_hours and discharge_hours must be disjoint")

    n = len(df)
    hours = pd.to_datetime(df["timestamp"]).dt.hour.to_numpy()
    p = df["price_eur_per_kwh"].to_numpy(dtype=float)

    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = np.zeros(n + 1)
    soc[0] = battery.initial_soc_kwh

    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    e_c_max = battery.max_charge_kwh
    e_d_max = battery.max_discharge_kwh
    cap = battery.capacity_kwh
    soc_target = battery.final_soc_kwh

    # Forward pass: apply windows greedily.
    for t in range(n):
        s = soc[t]
        h = int(hours[t])
        c = 0.0
        d = 0.0
        if h in charge_set:
            # Grid energy limited by power and remaining SOC room.
            room = (cap - s) / eta_c
            c = min(e_c_max, max(0.0, room))
        elif h in discharge_set:
            available = s * eta_d
            d = min(e_d_max, max(0.0, available))
        charge[t] = c
        discharge[t] = d
        soc[t + 1] = s + eta_c * c - d / eta_d

    # Terminal SOC repair: adjust later actions to hit final target.
    # Prefer reducing charge / reducing discharge first, then top-up.
    repair_log: list[str] = []
    tol = 1e-9

    def _recompute_soc_from(t0: int = 0) -> None:
        for t in range(t0, n):
            soc[t + 1] = soc[t] + eta_c * charge[t] - discharge[t] / eta_d

    # If final SOC too high: need net discharge (or less charge).
    if soc[n] > soc_target + tol:
        excess_batt = soc[n] - soc_target  # kWh in battery terms
        # 1) cut charge from the end (saves η_c * Δc in battery)
        for t in range(n - 1, -1, -1):
            if excess_batt <= tol:
                break
            if charge[t] <= tol:
                continue
            # reducing charge by dc lowers SOC path by eta_c * dc
            dc = min(charge[t], excess_batt / eta_c)
            charge[t] -= dc
            excess_batt -= eta_c * dc
            repair_log.append(f"cut_charge_t{t}={dc:.6f}")
        _recompute_soc_from(0)

        # 2) add discharge where possible
        if soc[n] > soc_target + tol:
            excess_batt = soc[n] - soc_target
            for t in range(n - 1, -1, -1):
                if excess_batt <= tol:
                    break
                if charge[t] > tol:
                    continue  # still no simultaneous
                # headroom to discharge more this hour
                # after full recompute we need local available at t
                # recompute prefix
                _recompute_soc_from(0)
                available = soc[t] * eta_d - discharge[t]
                available = min(available, e_d_max - discharge[t])
                if available <= tol:
                    continue
                # increasing discharge by dd lowers SOC by dd/eta_d
                dd = min(available, excess_batt * eta_d)
                discharge[t] += dd
                excess_batt -= dd / eta_d
                repair_log.append(f"add_discharge_t{t}={dd:.6f}")
            _recompute_soc_from(0)

    # If final SOC too low: need net charge (or less discharge).
    if soc[n] < soc_target - tol:
        deficit_batt = soc_target - soc[n]
        # 1) cut discharge from the end
        for t in range(n - 1, -1, -1):
            if deficit_batt <= tol:
                break
            if discharge[t] <= tol:
                continue
            dd = min(discharge[t], deficit_batt * eta_d)
            discharge[t] -= dd
            deficit_batt -= dd / eta_d
            repair_log.append(f"cut_discharge_t{t}={dd:.6f}")
        _recompute_soc_from(0)

        # 2) add charge where possible
        if soc[n] < soc_target - tol:
            deficit_batt = soc_target - soc[n]
            for t in range(n - 1, -1, -1):
                if deficit_batt <= tol:
                    break
                if discharge[t] > tol:
                    continue
                _recompute_soc_from(0)
                room = (cap - soc[t]) / eta_c - charge[t]
                room = min(room, e_c_max - charge[t])
                if room <= tol:
                    continue
                dc = min(room, deficit_batt / eta_c)
                charge[t] += dc
                deficit_batt -= eta_c * dc
                repair_log.append(f"add_charge_t{t}={dc:.6f}")
            _recompute_soc_from(0)

    # Clip numerical dust and enforce non-simultaneity.
    charge[charge < 1e-10] = 0.0
    discharge[discharge < 1e-10] = 0.0
    _recompute_soc_from(0)

    # Final hard check: project tiny residual onto last feasible charge/discharge.
    residual = soc[n] - soc_target
    if abs(residual) > 1e-6:
        repair_log.append(f"residual_soc_error_kwh={residual:.8f}")

    schedule = pd.DataFrame(
        {
            "timestamp": df["timestamp"].to_numpy(),
            "timestamp_end": (
                df["timestamp_end"].to_numpy()
                if "timestamp_end" in df.columns
                else np.array([pd.NaT] * n)
            ),
            "price_eur_per_kwh": p,
            "price_eur_per_mwh": p * 1000.0,
            "charge_kwh": charge,
            "discharge_kwh": discharge,
            "soc_start_kwh": soc[:-1],
            "soc_end_kwh": soc[1:],
            "is_charging_mode": charge > discharge,
            "arbitrage_eur": p * (discharge - charge),
        }
    )
    schedule.attrs["strategy"] = heuristic.name
    schedule.attrs["charge_hours"] = heuristic.charge_hours
    schedule.attrs["discharge_hours"] = heuristic.discharge_hours
    schedule.attrs["soc_repair_log"] = repair_log
    schedule.attrs["final_soc_kwh"] = float(soc[n])
    schedule.attrs["final_soc_target_kwh"] = float(soc_target)
    return schedule


def run_strategy(
    name: str,
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
) -> pd.DataFrame:
    """Dispatch by strategy name."""
    key = name.lower().strip()
    if key in {"idle", "a"}:
        return idle_strategy(prices, battery)
    if key in {"clock_window", "clock", "rule", "b"}:
        return clock_window_strategy(prices, battery)
    raise ValueError(f"Unknown strategy {name!r}")
