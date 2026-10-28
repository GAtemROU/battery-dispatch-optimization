# Battery Dispatch Optimizer

Wholesale day-ahead arbitrage for a simulated commercial battery, using real German SMARD prices.

Portfolio-style project: clean data pipeline, MILP dispatch, simple baselines, hold-out evaluation, tests, notebook, and figures.

## What it does

Given hourly day-ahead prices, schedule charge/discharge to maximize:

```text
sum_t  price_t * (discharge_t - charge_t)
```

subject to battery power, energy, efficiency, and no simultaneous charge and discharge.

**Strategies**

| Strategy | Idea |
|----------|------|
| `idle` | Do nothing (value = 0) |
| `clock_window` | Charge 12–15h, discharge 18–21h (EDA-based rule) |
| `optimized` | Full-horizon MILP (perfect foresight benchmark) |
| `day_ahead_rolling` | Each day: MILP on that day's actual DA prices only; SOC carries overnight |
| `forecast_optimize` | Seasonal-naive 7-day hour-mean forecast → MILP on forecast → P&L on actual |

`optimized` is an **upper bound**, not a deployable controller. `day_ahead_rolling` matches planning after the day-ahead auction. `forecast_optimize` is closer to deciding before prices are known.

## Main results

**Battery (illustrative):** 100 kWh, ±50 kW, η_c = η_d = 95%, start/end SOC = 50%.

**Data:** DE day-ahead prices, 2026-08-28 → 2026-10-28 (hourly SMARD export).

**Full sample** (perfect-foresight window for the first three strategies):

| Strategy | Arbitrage (EUR) | vs clock |
|----------|----------------:|---------:|
| idle | 0 | — |
| clock_window | 864 | — |
| optimized | 1,234 | +43% |

**Hold-out** (first 31 days warm-up, last 31 days scored):

| Strategy | Arbitrage (EUR) | vs clock |
|----------|----------------:|---------:|
| idle | 0 | — |
| clock_window | 351 | — |
| optimized (test window) | 531 | +51% |
| day_ahead_rolling | 529 | +51% |
| forecast_optimize | 505 | +44% |

On this sample, most value is **within-day** spread: day-ahead rolling nearly matches the test-window oracle. The naive forecast loses a modest amount (~26 EUR vs oracle on the test half).

Figures: `reports/figures/`. Tables: `reports/strategy_comparison*.csv`.

## Setup

```bash
python -m venv env
source env/bin/activate          # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Python 3.11+ (developed on 3.12). Solver: **HiGHS** via CVXPY.

## Reproduce

```bash
source env/bin/activate
python -m pytest tests/ -q
```

Full analysis path (data → strategies → figures → hold-out):

1. Open `notebooks/battery_dispatch.ipynb` with the project venv as the kernel.
2. Run all cells top to bottom.

Saved outputs (no need to re-run to inspect):

- Figures → `reports/figures/`
- Metrics → `reports/strategy_comparison.csv`, `reports/strategy_comparison_holdout.csv`
- Data notes → `data/README.md`

For a quick library smoke check from the repo root:

```bash
python -c "from src.data_loader import load_clean_prices; from src.optimizer import optimize_dispatch; p,_=load_clean_prices(); print(round(optimize_dispatch(p).objective_value, 2))"
```

## Project layout

```text
battery-dispatch-optimizer/
  data/
    raw/                 # original SMARD CSV (prices-de-aug-oct.csv)
    processed/           # cleaned hourly series
    README.md            # source, units, cleaning notes
  src/
    battery.py           # parameters (single config point)
    data_loader.py       # load / validate / unit convert
    optimizer.py         # CVXPY + HiGHS MILP
    strategies.py        # idle, clock, rolling DA, forecast
    evaluation.py        # metrics + constraint checks
    visualization.py     # figures
  tests/                 # pytest
  notebooks/battery_dispatch.ipynb
  reports/
    figures/             # PNG outputs
    *.csv                # comparison and sensitivity tables
  requirements.txt
```

## Model (short)

- **Objective:** maximize wholesale arbitrage in EUR (grid-side charge/discharge × price).
- **Dynamics:** `SOC[t+1] = SOC[t] + η_c·charge[t] − discharge[t]/η_d`
- **Constraints:** SOC ∈ [0, capacity], power caps, fixed initial SOC, optional final SOC, binary mode to forbid simultaneous charge and discharge.
- **Not included:** grid fees, taxes, imbalance, degradation, retail tariffs, forecasting ML models.

## Data notes

- Source: [SMARD](https://www.smard.de/en/downloadcenter/download-market-data) day-ahead prices (DE export under `data/raw/prices-de-aug-oct.csv`).
- Units: EUR/MWh in raw file → EUR/kWh in the model (`/ 1000`).
- Cleaning: sort, 1h intervals, no fabricated fills; one DST fall-back duplicate hour averaged (documented in `data/README.md`).
- Negative prices are kept (valid market outcomes).

## Limitations

- Perfect-foresight and even test-window optimization overstate what a live policy can earn without knowing future prices.
- Day-ahead rolling assumes next-day DA prices are known when planning (reasonable after auction; not before).
- Forecast is a simple seasonal naive — not a production price model.
- Results are **wholesale arbitrage value**, not customer bill savings or realized trading profit.
- Battery parameters are illustrative, not a real asset.

## License / use

Educational portfolio project. Follow SMARD terms for the market data files you download.
