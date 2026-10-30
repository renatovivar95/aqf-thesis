# Air Quality Forecasting for Aveiro District (Portugal)

Daily forecasting of air pollutant concentrations (NO2, O3, PM10) in the Aveiro District using statistical, machine learning, and hybrid models. This repository contains the code developed for my MSc Data Science thesis.

Four models are compared:

| Script | Model |
|---|---|
| `arimax_fourier_rep.py` | ARIMAX (SARIMAX with exogenous weather variables and Fourier terms) |
| `random_forest_rep.py` | Random Forest with lag features |
| `xgboost_rep.py` | XGBoost with lag, calendar, and (for O3) photochemical interaction features |
| `hybrid_rf_arimax_rep.py` | Hybrid Random Forest + SARIMAX on the Random Forest residuals |

## Repository structure

```
├── arimax_fourier_rep.py
├── random_forest_rep.py
├── xgboost_rep.py
├── hybrid_rf_arimax_rep.py
├── aveiro_2018_2022_final_dataset.csv   # Daily pollutant and weather data, 2018-2022
├── aveiro_weather_2018_2022.csv         # Daily Precipitation and Radiation, 2018-2022
├── requirements.txt
└── README.md
```

## Data

- **`aveiro_2018_2022_final_dataset.csv`**: daily averages for the Aveiro District from 2018-01-01 to 2022-12-31. Columns: `date`, `district`, `NO2`, `O3`, `PM10`, `Temperature`, `WindSpeed`. Values are raw daily means (missing days are not filled); each model script handles missing values itself.
- **`aveiro_weather_2018_2022.csv`**: daily `Precipitation` and `Radiation` for the same period.

The air quality data was originally stored in a local SQL Server data warehouse (`aqf_dw`, Kimball star schema). `export_dataset.py` shows the query used to extract it; it only works on the author's machine and is included for transparency. All model scripts read the CSV files directly, so no database is needed to reproduce the results.

<!-- TODO: add the original data sources (air quality monitoring network and weather data provider) -->

## Installation

Requires Python 3.9 or later.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## Usage

Run each script from the repository folder and choose the pollutant with `--pollutant`:

```bash
python arimax_model.py --pollutant PM10
python random_forest_model.py --pollutant NO2
python xgboost_model.py --pollutant O3
python hybrid_rf_sarimax.py --pollutant NO2
```

Supported pollutants: `NO2`, `O3`, `PM10`. The name must match the column name in the dataset exactly.

Optional arguments (same for all four scripts):

| Argument | Default | Description |
|---|---|---|
| `--pollutant` | `PM10` (ARIMAX), `O3` (XGBoost), `NO2` (others) | Target pollutant |
| `--data` | `aveiro_2018_2022_final_dataset.csv` | Path to the main dataset |
| `--weather` | `aveiro_weather_2018_2022.csv` | Path to the Precipitation and Radiation file |

To see the options for any script:

```bash
python xgboost_model.py --help
```

Grid searches and the daily-origin rolling evaluation can take several minutes per run, depending on the machine.

## Methodology

**Data split.** Training uses 2018–2021 and testing uses 2022. Hyperparameters and model orders are selected with time-series cross-validation inside the training period only, so the test year is never seen during model selection.

**Features.**
- Exogenous weather variables: Temperature, WindSpeed, Precipitation, Radiation.
- Lags of 1, 2, 3, and 7 days (Random Forest, XGBoost, Hybrid).
- Fourier terms for annual and weekly seasonality (ARIMAX and the SARIMAX component of the hybrid).
- Calendar features (XGBoost) and photochemical interaction features for O3 (XGBoost).

**Models.**
- **ARIMAX:** the (p, d, q) order is selected by time-series cross-validation on the training period. NO2, O3, and PM10 are modeled on a log scale (`log1p`) and transformed back for evaluation.
- **Random Forest / XGBoost:** hyperparameters are tuned with time-series cross-validation (randomized search for Random Forest, grid search with early stopping for XGBoost).
- **Hybrid RF + SARIMAX:** a Random Forest produces the main forecast and a SARIMAX models its residuals. The SARIMAX is trained on out-of-fold residuals, so it learns from out-of-sample errors of the same kind it sees in the test year.

**Evaluation.**
- **1-step-ahead:** predictions for each day of 2022 using observed values up to the previous day.
- **Rolling multi-horizon (1, 3, and 7 days):** 52 weekly windows across 2022. The tree-based models forecast recursively, using their own predictions as lag features inside each window.
- **Daily-origin rolling forecast:** a robustness check where the forecast origin moves one day at a time, so each horizon covers every day of the week.
- **Metrics:** MAE, RMSE, MAPE, and R², compared with a persistence baseline (tomorrow = today).

Future weather is taken from observations, which assumes a perfect weather forecast. The results are therefore an upper bound on what an operational system using real weather forecasts would achieve.

## Outputs

Each script creates its own results folder named after the model and pollutant, for example:

```
arimax_model_results_PM10_F2/
random_forest_model_results_NO2_F2/
xgboost_O3_F2/
hybrid_rf_arimax_results_NO2_F2/
```

Each folder contains:
- metrics (CSV)
- best hyperparameters or model order (CSV/JSON)
- feature importance (CSV and plot)
- multi-horizon results, per window and summarized (CSV)
- prediction and error plots (PNG)

## Author

Renato Andrés Vivar Orellana
MSc Data Science 
