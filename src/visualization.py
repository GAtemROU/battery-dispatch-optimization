"""Plot helpers for price EDA and later strategy figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIG_DIR = REPO_ROOT / "reports" / "figures"


def _ensure_fig_dir(path: Path | None = None) -> Path:
    fig_dir = Path(path) if path is not None else DEFAULT_FIG_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def plot_price_timeseries(
    prices: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    filename: str = "01_price_timeseries.png",
) -> Path:
    """Electricity prices over time (README figure family #1)."""
    fig_dir = _ensure_fig_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(
        prices["timestamp"],
        prices["price_eur_per_mwh"],
        color="#1f4e79",
        linewidth=0.8,
    )
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title("German day-ahead wholesale price (hourly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_price_by_hour_of_day(
    prices: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    filename: str = "02_price_by_hour_of_day.png",
) -> Path:
    """Mean ± band of price by hour of day."""
    fig_dir = _ensure_fig_dir(fig_dir)
    df = prices.copy()
    df["hour"] = df["timestamp"].dt.hour
    grouped = df.groupby("hour")["price_eur_per_mwh"]
    mean = grouped.mean()
    q10 = grouped.quantile(0.10)
    q90 = grouped.quantile(0.90)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(mean.index, q10, q90, color="#9dc3e6", alpha=0.5, label="P10–P90")
    ax.plot(mean.index, mean.values, color="#1f4e79", marker="o", label="Mean")
    ax.set_xticks(range(0, 24))
    ax.set_title("Average day-ahead price by hour of day")
    ax.set_xlabel("Hour of day (local wall clock)")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_price_by_day_of_week(
    prices: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    filename: str = "03_price_by_day_of_week.png",
) -> Path:
    """Mean price by weekday."""
    fig_dir = _ensure_fig_dir(fig_dir)
    df = prices.copy()
    # Monday=0 ... Sunday=6
    df["dow"] = df["timestamp"].dt.day_of_week
    mean = df.groupby("dow")["price_eur_per_mwh"].mean()
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(mean.index, mean.values, color="#2e75b6", edgecolor="white")
    ax.set_xticks(range(7))
    ax.set_xticklabels(labels)
    ax.set_title("Average day-ahead price by day of week")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Price (EUR/MWh)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_price_hour_dow_heatmap(
    prices: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    filename: str = "04_price_hour_dow_heatmap.png",
) -> Path:
    """Mean price heatmap: hour of day × day of week."""
    fig_dir = _ensure_fig_dir(fig_dir)
    df = prices.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.day_of_week
    pivot = df.pivot_table(
        index="hour",
        columns="dow",
        values="price_eur_per_mwh",
        aggfunc="mean",
    )
    # Ensure full 0–23 × 0–6 grid
    pivot = pivot.reindex(index=range(24), columns=range(7))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn_r")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean price (EUR/MWh)")
    ax.set_xticks(range(7))
    ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(0, 24, 2))
    ax.set_yticklabels(range(0, 24, 2))
    ax.set_title("Mean day-ahead price by hour and weekday")
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Hour of day")
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_daily_mean_price(
    prices: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    filename: str = "05_daily_mean_price.png",
) -> Path:
    """Daily mean price over the sample."""
    fig_dir = _ensure_fig_dir(fig_dir)
    daily = (
        prices.set_index("timestamp")["price_eur_per_mwh"].resample("D").mean()
    )

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily.values, color="#1f4e79", marker="o", markersize=3)
    ax.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_title("Daily mean day-ahead price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Mean price (EUR/MWh)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_price_eda_figures(prices: pd.DataFrame, fig_dir: Path | None = None) -> list[Path]:
    """Write the initial price EDA figure set to reports/figures/."""
    paths = [
        plot_price_timeseries(prices, fig_dir=fig_dir),
        plot_price_by_hour_of_day(prices, fig_dir=fig_dir),
        plot_price_by_day_of_week(prices, fig_dir=fig_dir),
        plot_price_hour_dow_heatmap(prices, fig_dir=fig_dir),
        plot_daily_mean_price(prices, fig_dir=fig_dir),
    ]
    return paths


def plot_strategy_arbitrage_comparison(
    metrics_table,
    *,
    fig_dir: Path | None = None,
    filename: str = "06_strategy_arbitrage_comparison.png",
) -> Path:
    """Bar chart of total arbitrage value across strategies (README fig #3)."""
    fig_dir = _ensure_fig_dir(fig_dir)
    df = metrics_table.copy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {
        "idle": "#9e9e9e",
        "clock_window": "#2e75b6",
        "optimized": "#c45911",
        "optimized_full_test": "#c45911",
        "day_ahead_rolling": "#548235",
        "forecast_optimize": "#7030a0",
    }
    bar_colors = [colors.get(str(s), "#4f4f4f") for s in df["strategy"]]
    ax.bar(df["strategy"].astype(str), df["arbitrage_eur"], color=bar_colors, edgecolor="white")
    ax.set_title("Wholesale arbitrage value by strategy")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total arbitrage value (EUR)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(df["arbitrage_eur"]):
        ax.text(i, v, f"{v:,.0f} €", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_dispatch_timeseries(
    schedule,
    *,
    fig_dir: Path | None = None,
    filename: str = "07_optimized_dispatch.png",
    title: str = "Optimized battery dispatch and state of charge",
    max_points: int | None = None,
) -> Path:
    """Price + charge/discharge + SOC (README fig #2).

    If max_points is set, plot only the first max_points rows (overview still
    uses full data when None). For long horizons the full series remains readable
    at dpi=150.
    """
    fig_dir = _ensure_fig_dir(fig_dir)
    df = schedule.reset_index(drop=True)
    if max_points is not None:
        df = df.iloc[:max_points].copy()

    ts = df["timestamp"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.0, 1.0]})

    axes[0].plot(ts, df["price_eur_per_mwh"], color="#1f4e79", linewidth=0.8)
    axes[0].axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    axes[0].set_ylabel("Price (EUR/MWh)")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(ts, df["charge_kwh"], width=0.03, color="#2e75b6", label="Charge")
    axes[1].bar(ts, -df["discharge_kwh"], width=0.03, color="#c45911", label="Discharge")
    axes[1].set_ylabel("Grid energy (kWh)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ts, df["soc_end_kwh"], color="#548235", linewidth=0.9)
    axes[2].set_ylabel("SOC end (kWh)")
    axes[2].set_xlabel("Date")
    axes[2].set_ylim(0, max(100.0, float(df["soc_end_kwh"].max()) * 1.05))
    axes[2].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_dispatch_zoom(
    schedule,
    *,
    start=None,
    end=None,
    fig_dir: Path | None = None,
    filename: str = "08_optimized_dispatch_zoom.png",
    title: str = "Optimized dispatch — sample week",
) -> Path:
    """Zoomed dispatch window for readability."""
    fig_dir = _ensure_fig_dir(fig_dir)
    df = schedule.reset_index(drop=True).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] < pd.Timestamp(end)]
    if df.empty:
        raise ValueError("dispatch zoom window is empty")
    return plot_dispatch_timeseries(df, fig_dir=fig_dir, filename=filename, title=title)


def plot_cumulative_arbitrage(
    schedules: dict,
    *,
    fig_dir: Path | None = None,
    filename: str = "09_cumulative_arbitrage.png",
) -> Path:
    """Cumulative arbitrage EUR over time for each strategy."""
    fig_dir = _ensure_fig_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    style = {
        "idle": ("#9e9e9e", "--"),
        "clock_window": ("#2e75b6", "-"),
        "optimized": ("#c45911", "-"),
        "optimized_full_test": ("#c45911", "-"),
        "day_ahead_rolling": ("#548235", "-"),
        "forecast_optimize": ("#7030a0", "-"),
    }
    for name, sched in schedules.items():
        ts = pd.to_datetime(sched["timestamp"])
        cum = sched["arbitrage_eur"].cumsum()
        color, ls = style.get(name, ("#333333", "-"))
        ax.plot(ts, cum, label=name, color=color, linestyle=ls, linewidth=1.2)
    ax.set_title("Cumulative wholesale arbitrage value")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative value (EUR)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_efficiency_sensitivity(
    results_df,
    *,
    fig_dir: Path | None = None,
    filename: str = "10_sensitivity_efficiency.png",
) -> Path:
    """Arbitrage vs round-trip efficiency (both legs equal η)."""
    fig_dir = _ensure_fig_dir(fig_dir)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        results_df["efficiency"],
        results_df["arbitrage_eur"],
        marker="o",
        color="#1f4e79",
        linewidth=1.5,
    )
    ax.set_title("Sensitivity: arbitrage vs charge/discharge efficiency")
    ax.set_xlabel("One-way efficiency η (= η_c = η_d)")
    ax.set_ylabel("Optimized arbitrage (EUR)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = fig_dir / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
