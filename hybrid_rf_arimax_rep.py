# =============================
# 1. IMPORTS
# =============================
import os
import json
import argparse
# import pyodbc   # only needed for the (commented-out) local SQL Server connection
import itertools
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller


# =============================
# 2. CONFIG (command-line arguments)
# =============================
# Usage example:
#   python hybrid_rf_sarimax.py --pollutant NO2
#   python hybrid_rf_sarimax.py --pollutant PM10 --data aveiro_2018_2022_final_dataset.csv
parser = argparse.ArgumentParser(description="Hybrid Random Forest + SARIMAX forecasting for Aveiro air quality.")
parser.add_argument("--pollutant", type=str, default="NO2",
                    help="Target pollutant column in the dataset (default: NO2)")
parser.add_argument("--data", type=str, default="aveiro_2018_2022_final_dataset.csv",
                    help="Path to the input CSV dataset (default: aveiro_2018_2022_final_dataset.csv)")
parser.add_argument("--weather", type=str, default="aveiro_weather_2018_2022.csv",
                    help="Path to the weather CSV with Precipitation and Radiation, used only if "
                         "those columns are not already in the main dataset")
args = parser.parse_args()

target = args.pollutant
split_date = '2022-01-01'
BURN_IN = 30   # first days of OOF residuals excluded from metrics (SARIMAX initialization)

output_dir = f"hybrid_rf_arimax_results_{target}_F2"
os.makedirs(output_dir, exist_ok=True)


# =============================
# 3-6B. LOADING AND CLEANING
# =============================
# --- Original version: local SQL Server data warehouse (not available to other users) ---
# conn = pyodbc.connect(
#     "DRIVER={ODBC Driver 18 for SQL Server};"
#     "SERVER=DESKTOP-FRDCJFV;"
#     "DATABASE=aqf_dw;"
#     "Trusted_Connection=yes;"
#     "TrustServerCertificate=yes;"
# )
#
# query = f"""
# SELECT * FROM (
#     SELECT CAST(t.time_key AS DATE) AS date, s.district, i.item_name, AVG(f.value) AS avg_value
#     FROM dbo.fact_air_quality_measurement f
#     JOIN dbo.dim_time t     ON f.time_key = t.time_key
#     JOIN dbo.dim_location s ON f.location_key = s.location_key
#     JOIN dbo.dim_item i     ON f.item_key = i.item_key
#     WHERE s.district = 'aveiro'
#       AND CAST(t.time_key AS DATE) BETWEEN '2018-01-01' AND '2022-12-31'
#       AND i.item_name IN ('{target}','Temperature','WindSpeed')
#     GROUP BY CAST(t.time_key AS DATE), s.district, i.item_name
# ) AS source
# PIVOT (AVG(avg_value) FOR item_name IN ([{target}],[Temperature],[WindSpeed])) AS pivot_table
# ORDER BY date;
# """
#
# df = pd.read_sql(query, conn)

# --- Public version: load the same data from a CSV file ---
raw = pd.read_csv(args.data)
raw['date'] = pd.to_datetime(raw['date'])

if 'district' in raw.columns:
    raw = raw[raw['district'].astype(str).str.lower() == 'aveiro']
raw = raw[(raw['date'] >= '2018-01-01') & (raw['date'] <= '2022-12-31')]

required = [target, 'Temperature', 'WindSpeed']
missing = [c for c in required if c not in raw.columns]
if missing:
    raise ValueError(f"Columns {missing} not found in {args.data}. "
                     f"Available columns: {raw.columns.tolist()}")

df = raw[['date'] + required].copy()
df = df.groupby('date', as_index=False).mean()   # one row per day, as in the SQL AVG/GROUP BY
df = df.sort_values('date').set_index('date').asfreq("D")

nulls = df.isnull().sum()
df = df.drop(columns=nulls[nulls > len(df) * 0.1].index)
if 'district' in df.columns:
    df = df.drop(columns=['district'])
df = df.interpolate(method='time').ffill().bfill()

# Precipitation and Radiation: taken from the main dataset if present, otherwise from the weather CSV
extra_weather = ["Precipitation", "Radiation"]
if all(c in raw.columns for c in extra_weather):
    weather_df = raw[['date'] + extra_weather].groupby('date').mean()
else:
    weather_df = pd.read_csv(args.weather, parse_dates=['date']).set_index('date')
weather_df = weather_df[extra_weather].reindex(df.index)
weather_df = weather_df.interpolate(method='time').ffill().bfill()
df = df.join(weather_df)


# =============================
# 7. LAGS
# =============================
LAGS = [1, 2, 3, 7]

def create_lags(df, lags=LAGS):
    out = df.copy()
    for col in df.columns:
        for lag in lags:
            out[f"{col}_lag{lag}"] = df[col].shift(lag)
    return out

df_lagged = create_lags(df).dropna()


# =============================
# 8. DATE-BASED SPLIT
# =============================
train = df_lagged[df_lagged.index < split_date]
test  = df_lagged[df_lagged.index >= split_date]

X_train, y_train = train.drop(columns=[target]), train[target]
X_test,  y_test  = test.drop(columns=[target]),  test[target]
FEATURE_COLS = X_train.columns.tolist()

print(f"Target pollutant: {target}")
print(f"Train: {y_train.index.min().date()} → {y_train.index.max().date()}")
print(f"Test:  {y_test.index.min().date()} → {y_test.index.max().date()}")


# =============================
# 9. RANDOM FOREST: TUNING ON TRAIN ONLY
# =============================
param_grid = {
    'n_estimators':      [200, 300, 500],
    'max_depth':         [6, 8, 10, None],
    'min_samples_leaf':  [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'max_features':      [0.3, 0.5, 0.7, 'sqrt'],
    'max_samples':       [0.7, 0.8, 0.9],
}
tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_grid, n_iter=50, cv=tscv,
    scoring='neg_mean_absolute_error', random_state=42, n_jobs=-1, verbose=1
)
search.fit(X_train, y_train)
best_params = search.best_params_
model = search.best_estimator_   # already refit on the full train set
print("RF best params:", best_params, "| CV MAE:", -search.best_score_)


# =============================
# 10. RF OUT-OF-FOLD RESIDUALS (train)
# =============================
# Each train residual is a TRUE out-of-sample error: the RF that produces it
# did not see that day. This way the SARIMAX learns from residuals of the same
# nature as the 2022 residuals.
oof_pred = pd.Series(np.nan, index=y_train.index)
for tr_idx, va_idx in tscv.split(X_train):
    rf_fold = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    rf_fold.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
    oof_pred.iloc[va_idx] = rf_fold.predict(X_train.iloc[va_idx])

oof_pred = oof_pred.dropna().asfreq("D")
train_residuals = (y_train.loc[oof_pred.index] - oof_pred).asfreq("D")
print(f"OOF residuals: {train_residuals.index.min().date()} → {train_residuals.index.max().date()} "
      f"({len(train_residuals)} days)")
print("RF OOF MAE (should be ≈ CV MAE):", mean_absolute_error(y_train.loc[oof_pred.index], oof_pred))

# RF 1-step prediction on test (actual lags) and actual test residuals
y_pred = pd.Series(model.predict(X_test), index=y_test.index).asfreq("D")
test_residuals_true = (y_test.asfreq("D") - y_pred)


# =============================
# 11. EXOGENOUS VARIABLES FOR THE RESIDUAL SARIMAX (weather + Fourier)
# =============================
def create_fourier_terms(index, period, K):
    t = np.arange(len(index))
    out = pd.DataFrame(index=index)
    for k in range(1, K + 1):
        out[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        out[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return out

weather_features = ["Temperature", "WindSpeed", "Precipitation", "Radiation"]
exog_all = pd.concat([
    df[weather_features],
    create_fourier_terms(df.index, 365, 3),
    create_fourier_terms(df.index, 7, 1),
], axis=1)

exog_train = exog_all.loc[train_residuals.index]
exog_test  = exog_all.loc[test_residuals_true.index]


# =============================
# 12. SARIMAX ORDER SELECTION (AIC, train OOF residuals only)
# =============================
adf_p = adfuller(train_residuals)[1]
print(f"ADF p-value OOF residuals: {adf_p:.4f} (<0.05 → stationary, d=0 justified)")

# d fixed at 0: RF residuals should be stationary, and AIC
# is not comparable across models with different d.
best_aic, best_order, sarimax_results = np.inf, None, None
warnings.filterwarnings("ignore")
for p, q in itertools.product(range(0, 4), range(0, 4)):
    try:
        fit = SARIMAX(train_residuals, exog=exog_train, order=(p, 0, q),
                      seasonal_order=(0, 0, 0, 0),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        if fit.aic < best_aic:
            best_aic, best_order, sarimax_results = fit.aic, (p, 0, q), fit
    except Exception:
        continue
print("Best SARIMAX order on residuals:", best_order, "AIC:", best_aic)

lb = acorr_ljungbox(sarimax_results.resid.iloc[BURN_IN:], lags=[10, 20, 30], return_df=True)
print(lb)


# =============================
# 13. 1-STEP EVALUATION ON 2022
# =============================
# extend: frozen parameters, the state continues from 2021-12-31 and is updated
# with each day's ACTUAL residual → residual at t predicted using residuals up to t-1.
sarimax_test = sarimax_results.extend(test_residuals_true, exog=exog_test)
predicted_residuals = sarimax_test.fittedvalues
y_pred_hybrid = (y_pred + predicted_residuals).clip(lower=0)

y_persist = X_test[f"{target}_lag1"]

def metrics(actual, pred):
    return {
        "MAE":  mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "MAPE": mean_absolute_percentage_error(actual, pred) * 100,
        "R2":   r2_score(actual, pred),
    }

m_persist = metrics(y_test, y_persist)
m_rf      = metrics(y_test, y_pred)
m_hybrid  = metrics(y_test, y_pred_hybrid)

mse_persist = mean_squared_error(y_test, y_persist)
skill_rf     = 1 - mean_squared_error(y_test, y_pred) / mse_persist
skill_hybrid = 1 - mean_squared_error(y_test, y_pred_hybrid) / mse_persist

metrics_df = pd.DataFrame([
    {"metric": k, "persistence": m_persist[k], "rf": m_rf[k], "hybrid": m_hybrid[k]}
    for k in ["MAE", "RMSE", "MAPE", "R2"]
] + [{"metric": "Skill vs persistence", "persistence": 0.0, "rf": skill_rf, "hybrid": skill_hybrid}])
metrics_df.to_csv(os.path.join(output_dir, f"metrics_test_1step_{target}.csv"), index=False)
print("\n=== Test 2022, 1-step ===")
print(metrics_df)


# =============================
# 14. OVERFITTING CHECK (OOF vs test, same horizon)
# =============================
oof_hybrid = (oof_pred + sarimax_results.fittedvalues).clip(lower=0)
idx_oof = oof_pred.index[BURN_IN:]

m_rf_oof     = metrics(y_train.loc[idx_oof], oof_pred.loc[idx_oof])
m_hybrid_oof = metrics(y_train.loc[idx_oof], oof_hybrid.loc[idx_oof])

overfit_df = pd.DataFrame([
    {"model": "RF",     "metric": k, "train_oof": m_rf_oof[k],     "test": m_rf[k]}
    for k in ["MAE", "RMSE", "MAPE"]
] + [
    {"model": "Hybrid", "metric": k, "train_oof": m_hybrid_oof[k], "test": m_hybrid[k]}
    for k in ["MAE", "RMSE", "MAPE"]
])
overfit_df["train_over_test"] = overfit_df["train_oof"] / overfit_df["test"]
overfit_df.to_csv(os.path.join(output_dir, f"overfitting_check_{target}.csv"), index=False)
print("\n=== Overfitting (1-step OOF vs 1-step test) ===")
print(overfit_df)


# =============================
# 15. SAVE PARAMETERS, IMPORTANCES, DIAGNOSTICS
# =============================
best_params_serializable = {
    k: (v.item() if isinstance(v, (np.integer, np.floating)) else v) for k, v in best_params.items()
}
with open(os.path.join(output_dir, f"best_params_{target}.json"), "w") as f:
    json.dump({
        "rf_best_params": best_params_serializable,
        "rf_best_cv_mae": float(-search.best_score_),
        "sarimax_best_order": list(best_order),
        "sarimax_best_aic": float(best_aic),
        "adf_pvalue_oof_residuals": float(adf_p),
    }, f, indent=4)

importance = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
importance.rename("importance").reset_index().rename(columns={"index": "feature"}) \
    .to_csv(os.path.join(output_dir, f"feature_importance_{target}.csv"), index=False)
lb.to_csv(os.path.join(output_dir, f"ljung_box_{target}.csv"))
with open(os.path.join(output_dir, f"sarimax_residuals_summary_{target}.txt"), "w", encoding="utf-8") as f:
    f.write(sarimax_results.summary().as_text())

plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label="Actual", linewidth=2)
plt.plot(y_test.index, y_pred, label="Random Forest", alpha=0.8)
plt.plot(y_test.index, y_pred_hybrid, label="Hybrid RF + SARIMAX", linewidth=1.5)
plt.legend(); plt.grid(True)
plt.title(f"{target} - 1-step prediction (2022)")
plt.savefig(os.path.join(output_dir, f"hybrid_forecast_1step_{target}.png"), dpi=300, bbox_inches="tight")
plt.close()


# =====================================================================
# 16. ROLLING MULTI-HORIZON (52 windows, step 7, h = 1, 3, 7)
# =====================================================================
# RF: recursive (own predicted lags within the window, actual values before the origin).
# SARIMAX: get_forecast of the residual with known exogenous variables; after each window
# the state is updated with the ACTUAL residuals (y - 1-step RF).
# Limitation: the SARIMAX models the 1-step RF error; at h=3 and h=7 the recursive
# RF accumulates additional error that the residual component does not model.
HORIZONS, N_WINDOWS, STEP = [1, 3, 7], 52, 7


def recursive_forecast_rf(model, df, target, feature_cols, origin_idx, horizon=STEP):
    known_target = df[target]
    preds = {}
    for h in range(horizon):
        current_idx = origin_idx + h
        current_date = df.index[current_idx]
        row = {}
        for col in feature_cols:
            if col.startswith(f"{target}_lag"):
                lag_date = df.index[current_idx - int(col.split("lag")[-1])]
                row[col] = preds.get(lag_date, known_target.loc[lag_date])
            elif "_lag" in col:
                base_col, lag_str = col.rsplit("_lag", 1)
                row[col] = df.loc[df.index[current_idx - int(lag_str)], base_col]
            else:
                row[col] = df.loc[current_date, col]
        preds[current_date] = model.predict(pd.DataFrame([row], index=[current_date])[feature_cols])[0]
    return preds


model.n_jobs = 1
origin_idx = df.index.get_loc(pd.Timestamp(split_date))
current_sarimax = sarimax_results
rows = []

for week in range(N_WINDOWS):
    if origin_idx + STEP > len(df):
        break
    origin_date = df.index[origin_idx - 1]
    window_dates = df.index[origin_idx: origin_idx + STEP]

    rf_preds = recursive_forecast_rf(model, df, target, FEATURE_COLS, origin_idx)
    exog_window = exog_all.loc[window_dates]
    resid_pred = current_sarimax.get_forecast(steps=STEP, exog=exog_window).predicted_mean
    resid_pred.index = window_dates

    for h in HORIZONS:
        d = window_dates[h - 1]
        rf_hat = rf_preds[d]
        hybrid_hat = max(rf_hat + resid_pred.loc[d], 0)
        y_true = df.loc[d, target]
        persist_hat = df.loc[origin_date, target]
        rows.append({
            "window": week + 1, "origin_date": origin_date, "target_date": d, "horizon": h,
            "actual": y_true, "rf_pred": rf_hat, "residual_pred": resid_pred.loc[d],
            "hybrid_pred": hybrid_hat,
            "abs_error": abs(y_true - hybrid_hat), "sq_error": (y_true - hybrid_hat) ** 2,
            "pct_error": abs(y_true - hybrid_hat) / abs(y_true) if y_true != 0 else np.nan,
            "persist_pred": persist_hat,
            "persist_abs_error": abs(y_true - persist_hat), "persist_sq_error": (y_true - persist_hat) ** 2,
        })

    current_sarimax = current_sarimax.append(
        test_residuals_true.loc[window_dates], exog=exog_window, refit=False)
    origin_idx += STEP

horizon_results = pd.DataFrame(rows)
summary = horizon_results.groupby("horizon").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("pct_error", lambda s: s.mean() * 100),
    persistence_MAE=("persist_abs_error", "mean"),
    persistence_RMSE=("persist_sq_error", lambda s: np.sqrt(s.mean())),
    n=("abs_error", "count"),
).reset_index()
print("\n=== Hybrid: rolling multi-horizon ===")
print(summary)

# Check: rolling h=1 should match the 1-step test
h1 = horizon_results[horizon_results.horizon == 1].set_index("target_date")["hybrid_pred"]
print("Max. difference h=1 vs 1-step:", (h1 - y_pred_hybrid.loc[h1.index]).abs().max())

horizon_results.to_csv(os.path.join(output_dir, f"hybrid_multi_horizon_by_window_{target}.csv"), index=False)
summary.to_csv(os.path.join(output_dir, f"hybrid_multi_horizon_summary_{target}.csv"), index=False)

plt.figure(figsize=(8, 5))
sns.boxplot(data=horizon_results, x="horizon", y="abs_error")
plt.title(f"{target} - Hybrid: absolute error by horizon (52 windows)")
plt.xlabel("Horizon (days)"); plt.ylabel("Absolute error"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "hybrid_horizon_abs_error_boxplot.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="horizon", y="RMSE")
plt.title(f"{target} - Hybrid: RMSE by horizon")
plt.xlabel("Horizon (days)"); plt.ylabel("RMSE"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "hybrid_horizon_rmse_bar.png"), bbox_inches="tight")
plt.close()

print(f"\nResults saved to: {output_dir}/")


# =====================================================================
# 16B. MULTI-HORIZON ROLLING FORECAST, DAILY ORIGIN (robustness check)
# =====================================================================
# Same procedure as section 16, but the origin advances ONE day at a time.
# RF: recursive forecast restarting at each origin.
# SARIMAX: residual forecast; after each window its STATE is updated with
# the true residual of ONE day (y - RF one-step prediction), parameters frozen.
WINDOW_D, STEP_D, HORIZONS_D = 7, 1, [1, 3, 7]

origin_idx = df.index.get_loc(pd.Timestamp(split_date))
current_d = sarimax_results
rows_d, w = [], 0

while origin_idx + WINDOW_D <= len(df):
    origin_date = df.index[origin_idx - 1]
    window_dates = df.index[origin_idx: origin_idx + WINDOW_D]

    rf_preds = recursive_forecast_rf(model, df, target, FEATURE_COLS, origin_idx, horizon=WINDOW_D)
    exog_window = exog_all.loc[window_dates]
    resid_pred = current_d.get_forecast(steps=WINDOW_D, exog=exog_window).predicted_mean
    resid_pred.index = window_dates
    persist_hat = df.loc[origin_date, target]

    for h in HORIZONS_D:
        d = window_dates[h - 1]
        rf_hat = rf_preds[d]
        hybrid_hat = max(rf_hat + resid_pred.loc[d], 0)
        y_true = df.loc[d, target]
        rows_d.append({
            "window": w + 1, "origin_date": origin_date, "target_date": d,
            "target_weekday": d.day_name(), "horizon": h,
            "actual": y_true, "rf_pred": rf_hat, "residual_pred": resid_pred.loc[d],
            "hybrid_pred": hybrid_hat,
            "abs_error": abs(y_true - hybrid_hat), "sq_error": (y_true - hybrid_hat) ** 2,
            "pct_error": abs(y_true - hybrid_hat) / abs(y_true) * 100 if y_true != 0 else np.nan,
            "persist_abs_error": abs(y_true - persist_hat),
            "persist_sq_error": (y_true - persist_hat) ** 2,
        })

    # update the SARIMAX state with the TRUE residual of ONE day (no re-estimation)
    first_day = window_dates[:STEP_D]
    current_d = current_d.append(test_residuals_true.loc[first_day],
                                 exog=exog_all.loc[first_day], refit=False)
    origin_idx += STEP_D
    w += 1

hr_d = pd.DataFrame(rows_d)
hr_d.to_csv(os.path.join(output_dir, f"hybrid_multi_horizon_by_window_daily_{target}.csv"), index=False)

summary_d = hr_d.groupby("horizon").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("pct_error", "mean"),
    persistence_MAE=("persist_abs_error", "mean"),
    persistence_RMSE=("persist_sq_error", lambda s: np.sqrt(s.mean())),
    n=("abs_error", "count"),
).reset_index()
summary_d.to_csv(os.path.join(output_dir, f"hybrid_multi_horizon_summary_daily_{target}.csv"), index=False)
print(f"\n=== Hybrid: rolling multi-horizon, DAILY origin ({w} windows) ===")
print(summary_d)

# Check 1: each horizon covers all weekdays
print("\nTarget weekdays per horizon:")
print(pd.crosstab(hr_d["horizon"], hr_d["target_weekday"]))

# Check 2: h=1 must match the one-step hybrid predictions (diff ~0)
h1 = hr_d[hr_d["horizon"] == 1].set_index("target_date")["hybrid_pred"]
print("Max diff h=1 vs 1-step:", (h1 - y_pred_hybrid.loc[h1.index]).abs().max())

# MAE by horizon and target weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
mae_wd = hr_d.pivot_table(index="horizon", columns="target_weekday",
                          values="abs_error", aggfunc="mean")[weekday_order]
mae_wd.to_csv(os.path.join(output_dir, f"hybrid_multi_horizon_mae_by_weekday_daily_{target}.csv"))
print("\nMAE by horizon and target weekday:")
print(mae_wd.round(3))