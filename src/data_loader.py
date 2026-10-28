"""Load and validate SMARD day-ahead price data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

# Repository root: parent of src/
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "prices-de-aug-oct.csv"
DEFAULT_PROCESSED_PATH = REPO_ROOT / "data" / "processed" / "prices_hourly.csv"

RAW_START_COL = "Start date"
RAW_END_COL = "End date"
RAW_PRICE_COL = "euro/MWh"


def load_raw_prices(path: Path | str | None = None) -> pd.DataFrame:
    """Load the original SMARD CSV without altering values.

    Expected columns: Start date, End date, euro/MWh.
    An optional leading unnamed index column is dropped if present.
    """
    csv_path = Path(path) if path is not None else DEFAULT_RAW_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw price file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Drop pandas leftover index column from a prior export, if present.
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = {RAW_START_COL, RAW_END_COL, RAW_PRICE_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns {sorted(missing)} in {csv_path}")

    return df.reset_index(drop=True)


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse timestamps, validate, convert units, resolve explicit issues.

    Returns
    -------
    clean : DataFrame
        Columns:
        - timestamp: interval start (naive local wall clock, sorted)
        - timestamp_end: interval end
        - price_eur_per_mwh: original units
        - price_eur_per_kwh: EUR/MWh / 1000
    report : dict
        Validation summary for logging and data docs.
    """
    report: dict[str, Any] = {
        "n_raw_rows": int(len(raw)),
        "issues": [],
        "actions": [],
    }

    work = raw.copy()
    work["timestamp"] = pd.to_datetime(work[RAW_START_COL], format="mixed")
    work["timestamp_end"] = pd.to_datetime(work[RAW_END_COL], format="mixed")
    work["price_eur_per_mwh"] = pd.to_numeric(work[RAW_PRICE_COL], errors="coerce")

    n_bad_price = int(work["price_eur_per_mwh"].isna().sum())
    if n_bad_price:
        report["issues"].append(f"non_numeric_price_rows={n_bad_price}")
        # Do not fabricate: drop only rows that cannot be parsed as numbers.
        work = work.dropna(subset=["price_eur_per_mwh"])
        report["actions"].append(f"dropped_non_numeric_price_rows={n_bad_price}")

    durations_h = (work["timestamp_end"] - work["timestamp"]).dt.total_seconds() / 3600.0
    bad_duration = work.loc[durations_h.sub(1.0).abs() > 1e-9]
    if len(bad_duration):
        report["issues"].append(f"non_one_hour_intervals={len(bad_duration)}")
        raise ValueError(
            f"Found {len(bad_duration)} intervals with duration != 1 hour; "
            "inspect raw file before continuing."
        )

    work = work.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    dup_mask = work["timestamp"].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    if n_dup_rows:
        # SMARD wall-clock export repeats the 02:00 hour on DST fall-back
        # without UTC offsets. Prices on the known duplicate are nearly
        # identical; average into one hourly observation and document it.
        dup_ts = work.loc[dup_mask, "timestamp"].unique()
        report["issues"].append(
            f"duplicate_start_timestamps={len(dup_ts)} "
            f"affecting_rows={n_dup_rows} "
            f"timestamps={[str(t) for t in dup_ts]}"
        )
        aggregated = (
            work.groupby("timestamp", as_index=False)
            .agg(
                timestamp_end=("timestamp_end", "max"),
                price_eur_per_mwh=("price_eur_per_mwh", "mean"),
                n_source_rows=("price_eur_per_mwh", "size"),
            )
        )
        n_collapsed = int((aggregated["n_source_rows"] > 1).sum())
        report["actions"].append(
            f"averaged_duplicate_wall_clock_hours={n_collapsed}; "
            "DST fall-back keeps one row per local hour (mean price)"
        )
        work = aggregated.drop(columns=["n_source_rows"])
    else:
        work = work[["timestamp", "timestamp_end", "price_eur_per_mwh"]]

    # Gaps after de-duplication (should be hourly continuous in local time,
    # except the spring-forward missing hour if present in other periods).
    gaps = work["timestamp"].diff().dropna()
    non_hourly = gaps[gaps != pd.Timedelta(hours=1)]
    report["n_non_hourly_gaps_after_clean"] = int(len(non_hourly))
    if len(non_hourly):
        report["issues"].append(
            "non_hourly_gaps="
            + str(
                [
                    {
                        "after": str(work.loc[i - 1, "timestamp"]),
                        "at": str(work.loc[i, "timestamp"]),
                        "delta_h": gaps.loc[i] / pd.Timedelta(hours=1),
                    }
                    for i in non_hourly.index[:10]
                ]
            )
        )

    work["price_eur_per_kwh"] = work["price_eur_per_mwh"] / 1000.0
    work = work[
        ["timestamp", "timestamp_end", "price_eur_per_mwh", "price_eur_per_kwh"]
    ].reset_index(drop=True)

    report["n_clean_rows"] = int(len(work))
    report["timestamp_min"] = str(work["timestamp"].min())
    report["timestamp_max"] = str(work["timestamp"].max())
    report["price_eur_per_mwh_min"] = float(work["price_eur_per_mwh"].min())
    report["price_eur_per_mwh_max"] = float(work["price_eur_per_mwh"].max())
    report["price_eur_per_mwh_mean"] = float(work["price_eur_per_mwh"].mean())
    report["n_negative_prices"] = int((work["price_eur_per_mwh"] < 0).sum())
    report["n_missing_prices"] = 0  # none retained after clean
    report["unit_conversion"] = "EUR/kWh = EUR/MWh / 1000"

    return work, report


def load_clean_prices(
    path: Path | str | None = None,
    *,
    save_processed: bool = True,
    processed_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """End-to-end: raw CSV → cleaned hourly series (+ optional parquet/csv write)."""
    raw = load_raw_prices(path)
    clean, report = clean_prices(raw)

    if save_processed:
        out = Path(processed_path) if processed_path is not None else DEFAULT_PROCESSED_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(out, index=False)
        report["processed_path"] = str(out)

    return clean, report
