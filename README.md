# Linear & Nonlinear Time Series Modeling & Forecasting

This repository contains an extensive implementation of both linear and nonlinear time-series analysis techniques applied on real and synthetic datasets.
The project includes forecasting, dynamical system reconstruction, nonlinear complexity estimation, and comparative modeling approaches.

The computational framework is nearly complete (≈85–90%).
A final report and surrogate-based statistical comparisons will be added in a future update.

## Features

### Linear Time Series Analysis

- Trend and seasonality removal
- Autocorrelation and partial autocorrelation analysis (ACF/PACF)
- AR / ARIMA model fitting and diagnostics
- Stationarity testing (ADF)
- Residual analysis
- Multi-step forecasting
- NRMSE evaluation

### Nonlinear Time Series Analysis

- Time-delay embedding
- False Nearest Neighbors (FNN)
- Correlation Dimension (D₂) estimation
- Delayed Mutual Information
- 3D attractor reconstruction
- Local prediction models (LAP, OLS, PCR-based models)
- Chaotic data generation (logistic map, Henon map)

## Datasets

- BTC/USDT cryptocurrency price series
- Henon map chaotic series

## Project Structure

```
/
├── data/
│   ├── BTCUSDT.csv
│   └── Henon.csv
│
├── code/
│   ├── LinearTSAnalysis.py
│   ├── NonLinearTimeSeriesAnalysis.py
│   ├── LinearTSAnalysis.ipynb
│   └── NonLinearTimeSeriesAnalysis.ipynb
│
└── README.md
```

## Project Status

In Progress — Approximately 85–90% Complete

Completed so far:
- Full linear analysis pipeline
- Full nonlinear analysis pipeline
- Dataset handling and visualization
- Forecasting frameworks (linear and nonlinear)

Pending:
- Generation of 40 surrogate time series
- Statistical comparison of nonlinear measures (q₀ vs q₁…q₄₀)
- Final written analysis/report

## Requirements

Python Packages:
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- scikit-learn
- nolds
- nolitsa

```
pip install numpy scipy statsmodels scikit-learn matplotlib nolds nolitsa
```

## Notes
- The nonlinear pipeline includes both deterministic and stochastic modeling techniques.
- Real and synthetic data experiments are separated for clarity and reproducibility.

- Additional documentation and visual outputs will be added in the final update.

## Author

Ilias Korompilis

## License

Academic and educational use only.

