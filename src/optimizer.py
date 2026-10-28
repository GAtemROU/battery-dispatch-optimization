"""Perfect-foresight battery arbitrage optimizer (CVXPY + HiGHS MILP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from src.battery import DEFAULT_BATTERY, BatteryParams


@dataclass
class OptimizeResult:
    """Solved dispatch schedule and solver metadata."""

    schedule: pd.DataFrame
    status: str
    objective_value: float
    solver: str
    battery: BatteryParams
    is_optimal: bool
    meta: dict[str, Any]


def _validate_prices(prices: pd.DataFrame) -> pd.Series:
    if "price_eur_per_kwh" not in prices.columns:
        raise ValueError("prices must include column price_eur_per_kwh")
    if "timestamp" not in prices.columns:
        raise ValueError("prices must include column timestamp")
    if prices["price_eur_per_kwh"].isna().any():
        raise ValueError("price_eur_per_kwh contains missing values")
    if len(prices) == 0:
        raise ValueError("prices is empty")
    return prices["price_eur_per_kwh"].astype(float).reset_index(drop=True)


def optimize_dispatch(
    prices: pd.DataFrame,
    battery: BatteryParams | None = None,
    *,
    initial_soc_kwh: float | None = None,
    final_soc_kwh: float | None = None,
    require_final_soc: bool = True,
    solver: str = "HIGHS",
    time_limit_s: float | None = None,
    verbose: bool = False,
    perfect_foresight: bool = True,
) -> OptimizeResult:
    """Maximize wholesale arbitrage over a price horizon.

    Decision variables (per hour t = 0..T-1)
    ----------------------------------------
    - ``charge_kwh[t]``: energy drawn from the grid (kWh)
    - ``discharge_kwh[t]``: energy injected to the grid (kWh)
    - ``soc_kwh[t]``: state of charge at the *start* of interval t (kWh),
      length T+1 so ``soc_kwh[T]`` is the terminal SOC
    - ``is_charging[t]``: binary mode — 1 charge allowed, 0 discharge allowed

    Objective
    ---------
    ``maximize sum_t price[t] * (discharge_kwh[t] - charge_kwh[t])``

    Dynamics (README)
    -----------------
    ``SOC[t+1] = SOC[t] + η_c * charge[t] - discharge[t] / η_d``

    Constraints
    -----------
    - power/energy caps per interval
    - ``0 <= SOC <= capacity``
    - fixed initial SOC (default: battery.initial_soc_kwh)
    - optional fixed final SOC (default: battery.final_soc_kwh);
      set ``require_final_soc=False`` for free terminal SOC
    - no simultaneous charge and discharge via binaries

    Notes
    -----
    ``price_eur_per_kwh`` is the series the solver *plans against*.
    For forecast-then-optimize, pass forecast prices here and recompute
    arbitrage later with actual prices.
    """
    battery = battery or DEFAULT_BATTERY
    price = _validate_prices(prices)
    timestamps = prices["timestamp"].reset_index(drop=True)
    t_end = (
        prices["timestamp_end"].reset_index(drop=True)
        if "timestamp_end" in prices.columns
        else pd.Series([pd.NaT] * len(prices))
    )

    soc0 = battery.initial_soc_kwh if initial_soc_kwh is None else float(initial_soc_kwh)
    socT = battery.final_soc_kwh if final_soc_kwh is None else float(final_soc_kwh)
    if not (0.0 <= soc0 <= battery.capacity_kwh):
        raise ValueError(f"initial_soc_kwh={soc0} outside [0, capacity]")
    if require_final_soc and not (0.0 <= socT <= battery.capacity_kwh):
        raise ValueError(f"final_soc_kwh={socT} outside [0, capacity]")

    T = len(price)
    p = price.to_numpy()

    charge = cp.Variable(T, nonneg=True, name="charge_kwh")
    discharge = cp.Variable(T, nonneg=True, name="discharge_kwh")
    soc = cp.Variable(T + 1, nonneg=True, name="soc_kwh")
    is_charging = cp.Variable(T, boolean=True, name="is_charging")

    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    e_c_max = battery.max_charge_kwh
    e_d_max = battery.max_discharge_kwh
    cap = battery.capacity_kwh

    constraints = [
        soc[0] == soc0,
        soc <= cap,
        charge <= e_c_max * is_charging,
        discharge <= e_d_max * (1 - is_charging),
    ]
    if require_final_soc:
        constraints.append(soc[T] == socT)

    for t in range(T):
        constraints.append(
            soc[t + 1]
            == soc[t] + eta_c * charge[t] - discharge[t] / eta_d
        )

    objective = cp.Maximize(p @ (discharge - charge))
    problem = cp.Problem(objective, constraints)

    solve_kwargs: dict[str, Any] = {"solver": getattr(cp, solver), "verbose": verbose}
    # HiGHS time limit via scipy/highspy path varies; pass if supported.
    if time_limit_s is not None:
        solve_kwargs["time_limit"] = time_limit_s

    try:
        problem.solve(**solve_kwargs)
    except Exception:
        # Retry without time_limit if solver rejects the kwarg.
        solve_kwargs.pop("time_limit", None)
        problem.solve(**solve_kwargs)

    status = problem.status
    is_optimal = status == cp.OPTIMAL
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"Optimizer failed with status={status!r}. "
            "Check battery params, price length, and solver install."
        )

    charge_v = np.asarray(charge.value, dtype=float).ravel()
    discharge_v = np.asarray(discharge.value, dtype=float).ravel()
    soc_v = np.asarray(soc.value, dtype=float).ravel()
    mode_v = np.asarray(is_charging.value, dtype=float).ravel()

    # Numerical cleanup of tiny residuals from MILP tolerance.
    charge_v = np.clip(charge_v, 0.0, None)
    discharge_v = np.clip(discharge_v, 0.0, None)
    soc_v = np.clip(soc_v, 0.0, cap)
    charge_v[charge_v < 1e-8] = 0.0
    discharge_v[discharge_v < 1e-8] = 0.0

    schedule = pd.DataFrame(
        {
            "timestamp": timestamps,
            "timestamp_end": t_end,
            "price_eur_per_kwh": p,
            "price_eur_per_mwh": p * 1000.0,
            "charge_kwh": charge_v,
            "discharge_kwh": discharge_v,
            "soc_start_kwh": soc_v[:-1],
            "soc_end_kwh": soc_v[1:],
            "is_charging_mode": mode_v >= 0.5,
            "arbitrage_eur": p * (discharge_v - charge_v),
        }
    )

    obj = float(problem.value) if problem.value is not None else float("nan")

    return OptimizeResult(
        schedule=schedule,
        status=str(status),
        objective_value=obj,
        solver=solver,
        battery=battery,
        is_optimal=is_optimal,
        meta={
            "n_intervals": T,
            "horizon_start": str(timestamps.iloc[0]),
            "horizon_end": str(timestamps.iloc[-1]),
            "perfect_foresight": perfect_foresight,
            "initial_soc_kwh": soc0,
            "final_soc_kwh": float(soc_v[-1]),
            "require_final_soc": require_final_soc,
            "final_soc_target_kwh": socT if require_final_soc else None,
        },
    )
