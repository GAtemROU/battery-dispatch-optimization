# Data

## Source

- Platform: [SMARD](https://www.smard.de/en/downloadcenter/download-market-data) (German electricity market data)
- Series: Day-ahead wholesale electricity prices
- Bidding zone: Germany / Luxembourg (user-filtered export saved as DE file)
- Units in raw file: EUR/MWh (`euro/MWh` column)
- Time resolution: hourly delivery intervals (`Start date` → `End date`)
- Sample period in use: 2026-08-28 00:00 through 2026-10-28 (local wall-clock timestamps as exported)
- Licensing: follow SMARD terms of use for market data downloads; retain the original file under `raw/`

## Files

| Path | Description |
|------|-------------|
| `raw/prices-de-aug-oct.csv` | Primary input used by the project (DE day-ahead, Aug–Oct 2026) |
| `raw/prices-aug-oct.csv` | Alternate/full export retained for reference (semicolon-separated; not the default loader input) |
| `processed/prices_hourly.csv` | Cleaned hourly series written by `src.data_loader` |

## Raw schema (`prices-de-aug-oct.csv`)

- Optional leading unnamed index column (dropped on load)
- `Start date` — interval start, US-style date string (e.g. `Aug 28, 2026 12:00 AM`)
- `End date` — interval end
- `euro/MWh` — day-ahead price

## Cleaning rules

Implemented in `src/data_loader.py`:

1. Parse start/end timestamps; require 1-hour duration.
2. Coerce prices to numeric; drop non-numeric rows only (no fabricated fills).
3. Sort by start timestamp.
4. **DST fall-back duplicate:** on 2026-10-25 the export contains two rows with wall-clock start `02:00` (prices 0.01 and 0.00 EUR/MWh). The file has no UTC offset, so both map to the same naive timestamp. They are averaged into one hourly observation and documented in the loader report. This prefers a unique local-hour index over inventing timezone offsets.
5. Convert `price_eur_per_kwh = price_eur_per_mwh / 1000`.
6. Missing observations are never silently interpolated.

## Notes

- Negative prices occur in the sample and are valid market outcomes; they are kept.
- Timestamps are naive local wall clock as provided by the download — not timezone-aware Europe/Berlin.
