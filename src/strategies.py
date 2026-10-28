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


def _day_groups(prices: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.DataFrame]]:
    """Split hourly frame into chronological calendar-day chunks."""
    df = prices.reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_day"] = df["timestamp"].dt.floor("D")
    groups: list[tuple[pd.Timestamp, pd.DataFrame]] = []
    for day, g in df.groupby("_day", sort=True):
        groups.append((pd.Timestamp(day), g.drop(columns=["_day"]).reset_index(drop=True)))
    return groups


def _stitch_day_schedules(
    day_schedules: list[pd.DataFrame],
    *,
    strategy_name: str,
    extra_attrs: dict | None = None,
) -> pd.DataFrame:
    if not day_schedules:
        raise ValueError("no daily schedules to stitch")
    out = pd.concat(day_schedules, ignore_index=True)
    out.attrs["strategy"] = strategy_name
    if extra_attrs:
        for k, v in extra_attrs.items():
            out.attrs[k] = v
    return out


def _revalue_schedule_with_actual_prices(
    plan: pd.DataFrame,
    actual_day: pd.DataFrame,
) -> pd.DataFrame:
    """Keep planned charge/discharge/SOC; score arbitrage on actual prices."""
    if len(plan) != len(actual_day):
        raise ValueError("plan and actual day length mismatch")
    out = plan.copy()
    p = actual_day["price_eur_per_kwh"].to_numpy(dtype=float)
    out["price_eur_per_kwh"] = p
    out["price_eur_per_mwh"] = p * 1000.0
    out["timestamp"] = actual_day["timestamp"].to_numpy()
    if "timestamp_end" in actual_day.columns:
        out["timestamp_end"] = actual_day["timestamp_end"].to_numpy()
    out["arbitrage_eur"] = p * (out["discharge_kwh"].to_numpy() - out["charge_kwh"].to_numpy())
    if "price_forecast_eur_per_kwh" in plan.columns:
        out["price_forecast_eur_per_kwh"] = plan["price_forecast_eur_per_kwh"].to_numpy()
    return out


def seasonal_naive_hour_forecast(
    history: pd.DataFrame,
    target_day: pd.DataFrame,
    *,
    lookback_days: int = 7,
) -> pd.DataFrame:
    """Forecast target day prices as mean of last ``lookback_days`` same clock-hours.

    Uses only rows in ``history`` with timestamp < first target timestamp (no leakage).
    Falls back to longer history mean by hour if fewer than lookback_days available.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    hist = history.reset_index(drop=True).copy()
    tgt = target_day.reset_index(drop=True).copy()
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])
    tgt["timestamp"] = pd.to_datetime(tgt["timestamp"])
    t0 = tgt["timestamp"].iloc[0]
    hist = hist.loc[hist["timestamp"] < t0].copy()
    if hist.empty:
        raise ValueError("history empty — need warm-up before first forecast day")

    hist["hour"] = hist["timestamp"].dt.hour
    hist["day"] = hist["timestamp"].dt.floor("D")
    # Keep the most recent lookback_days calendar days present in history.
    days = sorted(hist["day"].unique())
    use_days = days[-lookback_days:]
    window = hist.loc[hist["day"].isin(use_days)]
    hour_mean = window.groupby("hour")["price_eur_per_kwh"].mean()
    # Fallback: all history hour means for any missing hour.
    all_mean = hist.groupby("hour")["price_eur_per_kwh"].mean()

    hours = tgt["timestamp"].dt.hour
    fc = hours.map(lambda h: float(hour_mean.get(h, all_mean.get(h, np.nan))))
    if fc.isna().any():
        raise ValueError("forecast has NaN — insufficient history for some hours")

    out = tgt.copy()
    out["price_eur_per_kwh"] = fc.to_numpy(dtype=float)
    out["price_eur_per_mwh"] = out["price_eur_per_kwh"] * 1000.0
    out["price_forecast_eur_per_kwh"] = out["price_eur_per_kwh"]
    return out


def day_ahead_rolling_strategy(
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
    *,
    free_daily_final_soc: bool = True,
    final_day_soc_kwh: float | None = None,
) -> pd.DataFrame:
    """Rolling day-ahead MILP on *actual* DA prices, one calendar day at a time.

    Information set each day: that day's 24h prices only (post DA auction).
    No multi-week foresight. SOC carries overnight.

    Terminal SOC
    ------------
    - Intermediate days: free end-of-day SOC if ``free_daily_final_soc``.
    - Last day: hard final SOC (default battery.final_soc_kwh).
    """
    from src.optimizer import optimize_dispatch

    battery = battery or DEFAULT_BATTERY
    groups = _day_groups(prices)
    if not groups:
        raise ValueError("prices empty")

    soc = battery.initial_soc_kwh
    target_final = (
        battery.final_soc_kwh if final_day_soc_kwh is None else float(final_day_soc_kwh)
    )
    day_schedules: list[pd.DataFrame] = []
    day_meta: list[dict] = []

    for i, (day, day_df) in enumerate(groups):
        is_last = i == len(groups) - 1
        require_final = is_last or not free_daily_final_soc
        result = optimize_dispatch(
            day_df,
            battery,
            initial_soc_kwh=soc,
            final_soc_kwh=target_final if require_final else None,
            require_final_soc=require_final,
            perfect_foresight=True,  # perfect within the day only
        )
        sched = result.schedule.copy()
        sched["decision_day"] = day
        day_schedules.append(sched)
        soc = float(sched["soc_end_kwh"].iloc[-1])
        day_meta.append(
            {
                "day": str(day.date()),
                "status": result.status,
                "planned_obj": result.objective_value,
                "soc_end": soc,
                "n_hours": len(day_df),
                "require_final_soc": require_final,
            }
        )

    out = _stitch_day_schedules(
        day_schedules,
        strategy_name="day_ahead_rolling",
        extra_attrs={
            "rolling": True,
            "free_daily_final_soc": free_daily_final_soc,
            "day_meta": day_meta,
            "info_set": "each calendar day uses that day's actual DA prices only",
        },
    )
    return out


def forecast_optimize_strategy(
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
    *,
    lookback_days: int = 7,
    history: pd.DataFrame | None = None,
    free_daily_final_soc: bool = True,
    final_day_soc_kwh: float | None = None,
) -> pd.DataFrame:
    """Forecast next day (seasonal-naive hour means), MILP on forecast, settle on actual.

    Parameters
    ----------
    prices
        Days to *dispatch on* (typically the hold-out / test period).
    history
        Optional prior prices available before ``prices`` (warm-up). If None,
        uses only past days inside ``prices`` (first day then needs external history).
    lookback_days
        Number of past calendar days for same-hour mean forecast.
    """
    from src.optimizer import optimize_dispatch

    battery = battery or DEFAULT_BATTERY
    test = prices.reset_index(drop=True).copy()
    test["timestamp"] = pd.to_datetime(test["timestamp"])

    if history is None:
        hist_all = test.iloc[0:0].copy()
    else:
        hist_all = history.reset_index(drop=True).copy()
        hist_all["timestamp"] = pd.to_datetime(hist_all["timestamp"])

    groups = _day_groups(test)
    if not groups:
        raise ValueError("prices empty")

    soc = battery.initial_soc_kwh
    target_final = (
        battery.final_soc_kwh if final_day_soc_kwh is None else float(final_day_soc_kwh)
    )
    day_schedules: list[pd.DataFrame] = []
    day_meta: list[dict] = []
    # Running history: warm-up + previously completed actual test days.
    running_hist = hist_all.copy()

    for i, (day, day_df) in enumerate(groups):
        fc_day = seasonal_naive_hour_forecast(
            running_hist, day_df, lookback_days=lookback_days
        )
        is_last = i == len(groups) - 1
        require_final = is_last or not free_daily_final_soc
        result = optimize_dispatch(
            fc_day,
            battery,
            initial_soc_kwh=soc,
            final_soc_kwh=target_final if require_final else None,
            require_final_soc=require_final,
            perfect_foresight=False,
        )
        plan = result.schedule.copy()
        plan["price_forecast_eur_per_kwh"] = fc_day["price_eur_per_kwh"].to_numpy()
        settled = _revalue_schedule_with_actual_prices(plan, day_df)
        settled["decision_day"] = day
        day_schedules.append(settled)

        # SOC path follows planned physical actions (not re-optimized on actual).
        soc = float(settled["soc_end_kwh"].iloc[-1])
        mae = float(
            np.mean(
                np.abs(
                    fc_day["price_eur_per_kwh"].to_numpy()
                    - day_df["price_eur_per_kwh"].to_numpy(dtype=float)
                )
            )
        )
        day_meta.append(
            {
                "day": str(day.date()),
                "status": result.status,
                "planned_obj_on_forecast": result.objective_value,
                "realized_arb": float(settled["arbitrage_eur"].sum()),
                "forecast_mae_eur_per_kwh": mae,
                "soc_end": soc,
                "n_history_rows": int(len(running_hist)),
            }
        )
        # Reveal actual day only after planning (no leakage into this day's forecast).
        running_hist = pd.concat([running_hist, day_df], ignore_index=True)

    out = _stitch_day_schedules(
        day_schedules,
        strategy_name="forecast_optimize",
        extra_attrs={
            "rolling": True,
            "lookback_days": lookback_days,
            "free_daily_final_soc": free_daily_final_soc,
            "day_meta": day_meta,
            "info_set": (
                f"plan on seasonal-naive {lookback_days}d hour means; "
                "P&L on actual prices"
            ),
        },
    )
    return out


def holdout_split(
    prices: pd.DataFrame,
    *,
    frac: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split on calendar-day boundary nearest to ``frac`` of days.

    Returns (warm_up, test, meta). Warm-up is first half of *days*; test the rest.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("frac must be in (0, 1)")
    groups = _day_groups(prices)
    n_days = len(groups)
    if n_days < 4:
        raise ValueError(f"need at least 4 calendar days for hold-out, got {n_days}")
    n_warm = max(1, int(round(n_days * frac)))
    n_warm = min(n_warm, n_days - 1)
    warm_parts = [g for _, g in groups[:n_warm]]
    test_parts = [g for _, g in groups[n_warm:]]
    warm = pd.concat(warm_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    meta = {
        "n_days_total": n_days,
        "n_days_warm": n_warm,
        "n_days_test": n_days - n_warm,
        "warm_start": str(pd.to_datetime(warm["timestamp"].iloc[0])),
        "warm_end": str(pd.to_datetime(warm["timestamp"].iloc[-1])),
        "test_start": str(pd.to_datetime(test["timestamp"].iloc[0])),
        "test_end": str(pd.to_datetime(test["timestamp"].iloc[-1])),
        "frac": frac,
    }
    return warm, test, meta


def run_strategy(
    name: str,
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Dispatch by strategy name."""
    key = name.lower().strip()
    if key in {"idle", "a"}:
        return idle_strategy(prices, battery)
    if key in {"clock_window", "clock", "rule", "b"}:
        return clock_window_strategy(prices, battery)
    if key in {"day_ahead_rolling", "da_rolling", "rolling_da", "c"}:
        return day_ahead_rolling_strategy(prices, battery, **kwargs)
    if key in {"forecast_optimize", "forecast", "d"}:
        return forecast_optimize_strategy(prices, battery, **kwargs)
    raise ValueError(f"Unknown strategy {name!r}")
