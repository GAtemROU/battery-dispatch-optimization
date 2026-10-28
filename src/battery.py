"""Illustrative battery parameters (single configuration point)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BatteryParams:
    """Commercial battery assumptions for the simulation.

    Units
    -----
    - Energy: kWh
    - Power: kW
    - Interval length ``dt_hours``: h
    - For hourly steps, max energy per interval = power_kw * dt_hours (kWh).
    """

    capacity_kwh: float = 100.0
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    initial_soc_frac: float = 0.50
    final_soc_frac: float = 0.50
    dt_hours: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("power limits must be non-negative")
        if not (0.0 < self.charge_efficiency <= 1.0):
            raise ValueError("charge_efficiency must be in (0, 1]")
        if not (0.0 < self.discharge_efficiency <= 1.0):
            raise ValueError("discharge_efficiency must be in (0, 1]")
        if not (0.0 <= self.initial_soc_frac <= 1.0):
            raise ValueError("initial_soc_frac must be in [0, 1]")
        if not (0.0 <= self.final_soc_frac <= 1.0):
            raise ValueError("final_soc_frac must be in [0, 1]")
        if self.dt_hours <= 0:
            raise ValueError("dt_hours must be positive")

    @property
    def initial_soc_kwh(self) -> float:
        return self.initial_soc_frac * self.capacity_kwh

    @property
    def final_soc_kwh(self) -> float:
        return self.final_soc_frac * self.capacity_kwh

    @property
    def max_charge_kwh(self) -> float:
        """Max grid-side energy drawn in one interval."""
        return self.max_charge_kw * self.dt_hours

    @property
    def max_discharge_kwh(self) -> float:
        """Max grid-side energy injected in one interval."""
        return self.max_discharge_kw * self.dt_hours

    @property
    def round_trip_efficiency(self) -> float:
        return self.charge_efficiency * self.discharge_efficiency

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            {
                "initial_soc_kwh": self.initial_soc_kwh,
                "final_soc_kwh": self.final_soc_kwh,
                "max_charge_kwh": self.max_charge_kwh,
                "max_discharge_kwh": self.max_discharge_kwh,
                "round_trip_efficiency": self.round_trip_efficiency,
            }
        )
        return d


# Default params used across strategies (README §5).
DEFAULT_BATTERY = BatteryParams()
