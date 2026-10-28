# Project Brief: Battery Dispatch Optimizer

## 1. Objective

Build a small, polished, reproducible Python project demonstrating battery scheduling optimization using real German wholesale electricity price data.

The project is intended as a portfolio piece for a Data Scientist position at coneva, focused on energy flexibility, forecasting, and optimization.

The goal is to demonstrate practical skills in:
- Python, Pandas, NumPy, SciPy, and Jupyter.
- Mathematical optimization using CVXPY and a suitable solver.
- Data cleaning, validation, and exploratory analysis.
- Energy-system constraints and quantitative evaluation.
- Clear reporting, visualization, testing, and reproducibility.

Keep the project simple, technically correct, and easy to understand. Prioritize a working end-to-end result over unnecessary features.

## 2. Core Use Case

Given historical hourly German day-ahead electricity prices, determine when a simulated commercial battery should charge and discharge to maximize its wholesale-market arbitrage value.

Compare three strategies:

1. Idle battery: no charging or discharging.
2. Rule-based baseline: charge during relatively cheap hours and discharge during expensive hours.
3. Optimized schedule: use mathematical optimization to maximize arbitrage value subject to battery constraints.

The optimization model may use the entire historical price series as an idealized benchmark. Clearly explain that this assumes perfect foresight and does not represent a deployable real-time trading strategy.

Do not claim that the results represent actual customer savings or realized trading profits.

## 3. Data Source

Use SMARD, the German electricity market data platform:

https://www.smard.de/en/downloadcenter/download-market-data

Download hourly day-ahead wholesale electricity prices for the Germany/Luxembourg bidding zone, using a recent historical period with sufficient valid observations.

Start with 4–8 weeks of data.

Requirements:
- Store the original downloaded file locally under `data/raw/`.
- Document the download steps, source, selected period, units, and licensing.
- Never fabricate or silently replace missing observations.
- Handle timestamps, sorting, duplicate records, missing values, and price units explicitly.
- Convert prices from EUR/MWh to EUR/kWh where necessary.
- Make the analysis reproducible from the documented input file.
- Do not build a complex API integration unless it is straightforward and necessary.

If downloading data is not possible, create a clearly documented ingestion interface and ask the user to supply the file. Do not substitute fabricated data for real observations.

## 4. Technology Stack

Use:
- Python 3.11 or another compatible, documented version.
- Pandas and NumPy for data processing.
- CVXPY for optimization modeling.
- HiGHS as the preferred solver if compatible with the chosen CVXPY model.
- Matplotlib for visualizations.
- pytest for automated tests.
- Jupyter Notebook for exploratory analysis and final presentation.
- Git and `requirements.txt` for reproducibility.

Avoid unnecessary dependencies, web applications, databases, cloud services, Docker, or machine-learning frameworks.

## 5. Battery Assumptions

Use the following illustrative parameters, configurable in one place:

- Energy capacity: 100 kWh.
- Maximum charging power: 50 kW.
- Maximum discharging power: 50 kW.
- Charging efficiency: 95%.
- Discharging efficiency: 95%.
- Initial state of charge: 50% of capacity.
- Final state of charge: equal to initial state of charge.
- Time interval: one hour.

These are simulation assumptions, not specifications of an actual coneva customer system.

Use consistent units throughout the model. For hourly intervals, 50 kW corresponds to a maximum of 50 kWh of energy charged or discharged in one interval.

## 6. Optimization Model

Implement a linear or mixed-integer linear optimization model.

Decision variables for each time interval:
- Charging energy in kWh.
- Discharging energy in kWh.
- Battery state of charge in kWh.
- Binary charging/discharging mode variables if needed to prohibit simultaneous charging and discharging.

Objective:

Maximize the wholesale-market arbitrage value:

Sum over time of:

`price_eur_per_kwh * (discharge_kwh - charge_kwh)`

The price is in EUR/kWh and interval duration is one hour.

State-of-charge transition:

`SOC[t+1] = SOC[t] + charging_efficiency * charge[t] - discharge[t] / discharging_efficiency`

Constraints:
- State of charge remains between zero and battery capacity.
- Charging and discharging remain within their power limits.
- Initial state of charge is fixed.
- Final state of charge equals initial state of charge.
- Simultaneous charging and discharging is prohibited, preferably through binary mode variables and a MILP formulation.
- The solver must return an optimal or otherwise explicitly handled status.

Use the same assumptions and economic accounting across all strategy comparisons.

The model optimizes energy arbitrage only. Exclude grid charges, taxes, levies, market access fees, degradation costs, imbalance costs, and retail tariff effects from the initial version. State these exclusions in the report.

## 7. Baseline Strategies

### A. Idle baseline
Keep the battery inactive. Arbitrage value is zero.

### B. Rule-based baseline
Implement a simple, deterministic heuristic that:
- Uses historical price thresholds or daily price rankings.
- Respects the same battery capacity and power limits.
- Accounts for charging and discharging efficiencies.
- Maintains a valid state of charge.
- Respects the final state-of-charge condition.

Document the heuristic clearly. It is a benchmark, not necessarily an optimal controller.

### C. Optimized strategy
Use CVXPY and the selected solver to calculate the optimal schedule.

All three strategies must use consistent definitions of arbitrage value and energy flows.

## 8. Evaluation Metrics

Calculate and report:
- Total arbitrage value in EUR over the evaluated period.
- Arbitrage value per unit of battery capacity, in EUR/kWh of capacity.
- Total energy charged and discharged, in kWh.
- Equivalent full cycles, with the definition documented.
- Hours spent charging and discharging.
- Minimum and maximum state of charge.
- Constraint violations.
- Improvement over the rule-based strategy, when its value is nonzero.

Do not report percentage improvement over a zero-valued baseline.

Clearly distinguish cumulative arbitrage value from actual electricity-bill savings.

## 9. Visualizations

Produce at least three clear, report-quality figures:

1. Electricity prices over time, with useful units and date labels.
2. Optimized battery charging/discharging schedule and state of charge.
3. Comparison of arbitrage value across the three strategies.

Optionally add a sensitivity chart showing how arbitrage value changes with battery efficiency or capacity.

Use descriptive titles, axis labels, legends, units, readable date formatting, and consistent terminology.

Save the figures to `reports/figures/` so they can be inspected without rerunning the notebook.

## 10. Project Structure

Use this structure:

battery-dispatch-optimizer/
    README.md
    requirements.txt
    .gitignore
    data/
        raw/
        processed/
        README.md
    notebooks/
        battery_dispatch.ipynb
    src/
        __init__.py
        data_loader.py
        optimizer.py
        strategies.py
        evaluation.py
        visualization.py
    tests/
        test_data_loader.py
        test_optimizer.py
        test_evaluation.py
    reports/
        figures/
        report.md

Keep modules small and responsibilities clear. Avoid overengineering. If the final structure becomes unnecessarily complicated, simplify it while retaining separation between data handling, optimization, and evaluation.

## 11. Implementation Plan

### Phase 1 — Data ingestion and validation
- Create the repository and Python environment.
- Acquire the historical SMARD CSV.
- Inspect its schema and identify the actual price and timestamp columns.
- Implement data loading and cleaning.
- Validate timestamps, units, missing values, and duplicate observations.
- Produce an initial price visualization.

Deliverable: a clean, validated hourly price series and documented data source.

### Phase 2 — Optimization model
- Implement battery parameters and state transitions.
- Formulate the objective and constraints.
- Implement the CVXPY model.
- Select and test a compatible solver.
- Validate solution status and numerical feasibility.
- Test edge cases such as constant prices, insufficient data, and negative prices.

Deliverable: a working optimizer with constraint checks.

### Phase 3 — Baselines and comparison
- Implement the idle baseline.
- Implement the deterministic rule-based baseline.
- Calculate the same metrics for every strategy.
- Generate the strategy-comparison table.
- Confirm that all strategies respect battery limits and final state of charge.

Deliverable: reproducible quantitative comparisons.

### Phase 4 — Reporting and sensitivity analysis
- Generate the required plots.
- Run at least one small sensitivity analysis.
- Write a concise report describing methods, results, assumptions, and limitations.
- Explain the perfect-foresight assumption and the distinction between wholesale arbitrage value and customer savings.

Deliverable: a report that can be understood without reading the source code.

### Phase 5 — Quality and polish
- Add unit tests.
- Ensure the project runs from a clean Python environment.
- Verify that file paths are repository-relative.
- Document setup and execution commands.
- Check that the notebook runs from top to bottom.
- Remove unused dependencies, dead code, and misleading claims.

Deliverable: a clean, reproducible portfolio repository.

Target completion time: 3–5 focused working days.

## 12. Acceptance Criteria

The project is complete when:

- A real historical price dataset has been used.
- The source, period, and units are documented.
- The optimization model solves successfully.
- The battery obeys all physical constraints.
- Simultaneous charging and discharging is prohibited.
- The initial and final state of charge are consistent.
- All three strategies are evaluated using the same metric definitions.
- The report includes numerical results and at least three useful figures.
- Automated tests pass.
- The project can be reproduced using documented commands.
- Limitations and modeling assumptions are explicit.
- No results, data points, or performance claims have been fabricated.

## 13. Scope Boundaries

Do not implement in version one:
- Forecasting models.
- Real-time trading or control.
- Customer billing simulations.
- PV generation or EV charging integration.
- Web dashboards.
- Live market APIs.
- Complex battery degradation models.

These may be future extensions, but the first version must remain focused on a correct and polished optimization workflow.

## 14. Final Deliverables

Produce:
1. A clean Git repository.
2. A Jupyter notebook with the complete analysis.
3. Modular, tested Python code.
4. Saved figures.
5. A concise report in `reports/report.md`.
6. A README containing setup instructions, a project summary, the main findings, and limitations.

At the end, provide a brief summary of the implemented work, the actual numerical results obtained, the tests run, any remaining limitations, and the exact commands needed to reproduce the analysis.

Do not claim completion of any step that has not actually been executed and verified.
