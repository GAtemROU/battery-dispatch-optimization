"""Evaluate dispatch schedules against battery constraints and economics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.battery import DEFAULT_BATTERY, BatteryParams


REQUIRED_COLS = {
    "price_eur_per_kwh",
    "charge_kwh",
    "discharge_kwh",
    "soc_start_kwh",
    "soc_end_kwh",
}


def evaluate_schedule(
    schedule: pd.DataFrame,
    battery: BatteryParams | None = None,
    *,
    strategy_name: str | None = None,
    tol: float = 1e-6,
) -> dict[str, Any]:
    """Compute metrics and constraint-violation flags for one schedule."""
    battery = battery or DEFAULT_BATTERY
    df = schedule.reset_index(drop=True)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"schedule missing columns: {sorted(missing)}")

    charge = df["charge_kwh"].to_numpy(dtype=float)
    discharge = df["discharge_kwh"].to_numpy(dtype=float)
    soc_s = df["soc_start_kwh"].to_numpy(dtype=float)
    soc_e = df["soc_end_kwh"].to_numpy(dtype=float)
    price = df["price_eur_per_kwh"].to_numpy(dtype=float)

    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency

    # Rebuild SOC path for consistency check.
    soc_path = np.zeros(len(df) + 1)
    soc_path[0] = soc_s[0]
    for t in range(len(df)):
        soc_path[t + 1] = soc_path[t] + eta_c * charge[t] - discharge[t] / eta_d

    arb = price * (discharge - charge)
    total_arb = float(arb.sum())
    e_charge = float(charge.sum())
    e_discharge = float(discharge.sum())
    # Equivalent full cycles: average of charged/discharged battery-side energy / capacity.
    batt_in = float((eta_c * charge).sum())
    batt_out = float((discharge / eta_d).sum())
    full_cycles = 0.5 * (batt_in + batt_out) / battery.capacity_kwh

    hours_charge = int((charge > tol).sum())
    hours_discharge = int((discharge > tol).sum())

    violations = {
        "soc_below_zero": int((soc_path < -tol).sum()),
        "soc_above_capacity": int((soc_path > battery.capacity_kwh + tol).sum()),
        "charge_above_power": int((charge > battery.max_charge_kwh + tol).sum()),
        "discharge_above_power": int(
            (discharge > battery.max_discharge_kwh + tol).sum()
        ),
        "simultaneous_charge_discharge": int(
            ((charge > tol) & (discharge > tol)).sum()
        ),
        "initial_soc_mismatch": float(abs(soc_path[0] - battery.initial_soc_kwh)),
        "final_soc_mismatch": float(abs(soc_path[-1] - battery.final_soc_kwh)),
        "soc_dynamics_mismatch_max": float(
            np.max(np.abs(soc_e - soc_path[1:])) if len(df) else 0.0
        ),
    }
    violations["n_constraint_violations"] = int(
        violations["soc_below_zero"]
        + violations["soc_above_capacity"]
        + violations["charge_above_power"]
        + violations["discharge_above_power"]
        + violations["simultaneous_charge_discharge"]
        + (violations["initial_soc_mismatch"] > tol)
        + (violations["final_soc_mismatch"] > 1e-4)
    )

    name = strategy_name or schedule.attrs.get("strategy") or "unknown"

    return {
        "strategy": name,
        "n_intervals": int(len(df)),
        "total_arbitrage_eur": total_arb,
        "arbitrage_per_kwh_capacity_eur": total_arb / battery.capacity_kwh,
        "total_charge_kwh": e_charge,
        "total_discharge_kwh": e_discharge,
        "equivalent_full_cycles": float(full_cycles),
        "hours_charging": hours_charge,
        "hours_discharging": hours_discharge,
        "soc_min_kwh": float(soc_path.min()),
        "soc_max_kwh": float(soc_path.max()),
        "soc_final_kwh": float(soc_path[-1]),
        "violations": violations,
        "feasible": violations["n_constraint_violations"] == 0,
    }


def comparison_table(metrics_list: list[dict[str, Any]]) -> pd.DataFrame:
    """Tabulate strategy metrics for reporting."""
    rows = []
    for m in metrics_list:
        rows.append(
            {
                "strategy": m["strategy"],
                "arbitrage_eur": m["total_arbitrage_eur"],
                "eur_per_kwh_cap": m["arbitrage_per_kwh_capacity_eur"],
                "charge_kwh": m["total_charge_kwh"],
                "discharge_kwh": m["total_discharge_kwh"],
                "full_cycles": m["equivalent_full_cycles"],
                "hours_chg": m["hours_charging"],
                "hours_dis": m["hours_discharging"],
                "soc_min": m["soc_min_kwh"],
                "soc_max": m["soc_max_kwh"],
                "feasible": m["feasible"],
            }
        )
    table = pd.DataFrame(rows)
    # Improvement vs rule baseline when present and nonzero.
    if "clock_window" in set(table["strategy"]) or "rule" in set(table["strategy"]):
        base_names = {"clock_window", "rule", "clock"}
        base = table.loc[table["strategy"].isin(base_names), "arbitrage_eur"]
        if len(base):
            b = float(base.iloc[0])
            if abs(b) > 1e-12:
                table["improvement_vs_rule_eur"] = table["arbitrage_eur"] - b
                table["improvement_vs_rule_pct"] = 100.0 * (table["arbitrage_eur"] - b) / abs(b)
            else:
                table["improvement_vs_rule_eur"] = table["arbitrage_eur"] - b
                table["improvement_vs_rule_pct"] = np.nan
    return table
