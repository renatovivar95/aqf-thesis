# =============================
# 1. IMPORTS
# =============================
import os
import json
import argparse
# import pyodbc   # only needed for the (commented-out) local SQL Server connection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


# =============================
# 2. TARGET VARIABLE
# =============================
# Usage example:
#   python random_forest_model.py --pollutant NO2
#   python random_forest_model.py --pollutant PM10 --data aveiro_2018_2022_final_dataset.csv
parser = argparse.ArgumentParser(description="Random Forest forecasting for Aveiro air quality.")
parser.add_argument("--pollutant", type=str, default="NO2",
                    help="Target pollutant column in the dataset (default: NO2)")
parser.add_argument("--data", type=str, default="aveiro_2018_2022_final_dataset.csv",
                    help="Path to the input CSV dataset (default: aveiro_2018_2022_final_dataset.csv)")
parser.add_argument("--weather", type=str, default="aveiro_weather_2018_2022.csv",
                    help="Path to the weather CSV with Precipitation and Radiation, used only if "
                         "those columns are not already in the main dataset")
args = parser.parse_args()

target = args.pollutant  # e.g. 'NO2', 'PM10', 'O3'

# Single results folder holds everything: plots, params, metrics, importances, CSVs
output_dir = f"random_forest_model_results_{target}_F2"
os.makedirs(output_dir, exist_ok=True)


# =============================
# 3-5. LOAD DATA
# =============================
# --- Original version: local SQL Server data warehouse (not available to other users) ---
# # --- 3. DATABASE CONNECTION ---
# conn = pyodbc.connect(
#     "DRIVER={ODBC Driver 18 for SQL Server};"
#     "SERVER=DESKTOP-FRDCJFV;"
#     "DATABASE=aqf_dw;"
#     "Trusted_Connection=yes;"
#     "TrustServerCertificate=yes;"
# )
#
#
# # --- 4. QUERY ---
# query = f"""
# SELECT *
# FROM (
#     SELECT 
#         CAST(t.time_key AS DATE) AS date,
#         s.district,
#         i.item_name,
#         AVG(f.value) AS avg_value
#     FROM dbo.fact_air_quality_measurement f
#     JOIN dbo.dim_time t     ON f.time_key = t.time_key
#     JOIN dbo.dim_location s ON f.location_key = s.location_key
#     JOIN dbo.dim_item i     ON f.item_key = i.item_key
#
#     WHERE 
#         s.district = 'aveiro'
#         AND CAST(t.time_key AS DATE) BETWEEN '2018-01-01' AND '2022-12-31'
#         AND i.item_name IN ('{target}','Temperature','WindSpeed')
#
#     GROUP BY 
#         CAST(t.time_key AS DATE),
#         s.district,
#         i.item_name
# ) AS source
#
# PIVOT (
#     AVG(avg_value)
#     FOR item_name IN ([{target}],[Temperature],[WindSpeed])
# ) AS pivot_table
#
# ORDER BY date;
# """
#
#
# # --- 5. LOAD DATA ---
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

df = raw[['date'] + required].groupby('date', as_index=False).mean()   # one row per day, as in the SQL AVG/GROUP BY

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')
df = df.asfreq("D")


# =============================
# 6. HANDLE MISSING VALUES
# =============================
sum_nulls = df.isnull().sum()
threshold = len(df) * 0.1
cols_to_drop = sum_nulls[sum_nulls > threshold].index
df = df.drop(columns=cols_to_drop)

if 'district' in df.columns:
    df = df.drop(columns=['district'])

df = df.interpolate(method='time').ffill().bfill()


# =============================
# 6B. ADD WEATHER DATA
# =============================
# Precipitation and Radiation: taken from the main dataset if present, otherwise from the weather CSV
extra_weather = ["Precipitation", "Radiation"]
if all(c in raw.columns for c in extra_weather):
    weather_df = raw[['date'] + extra_weather].groupby('date').mean()
else:
    weather_df = pd.read_csv(args.weather, parse_dates=['date']).set_index('date')
weather_df = weather_df[extra_weather]
weather_df = weather_df.reindex(df.index)
weather_df = weather_df.interpolate(method='time').ffill().bfill()

df = df.join(weather_df)

print("Shape after merge:", df.shape)
print("Nulls after merge:\n", df.isnull().sum())
print("Columns after merge:", df.columns.tolist())



# =============================
# 7. CREATE LAG FEATURES
# =============================
LAGS = [1, 2, 3, 7]

def create_lags(df, lags=LAGS):
    df_lagged = df.copy()
    for col in df.columns:
        for lag in lags:
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df_lagged


df_lagged = create_lags(df, lags=LAGS)
df_lagged = df_lagged.dropna()


# =============================
# 8. TRAIN-TEST SPLIT
# =============================
split_date = '2022-01-01'

train = df_lagged[df_lagged.index < split_date]
test  = df_lagged[df_lagged.index >= split_date]

print(f"\nTrain period: {train.index.min()} → {train.index.max()}")
print(f"Test  period: {test.index.min()} → {test.index.max()}")

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

FEATURE_COLS = X_train.columns.tolist()  # exact column order the model expects


# =============================
# 9A. RANDOMIZED SEARCH (HYPERPARAMETER TUNING)
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

print("\n===== RANDOMIZED SEARCH (HYPERPARAMETER TUNING) =====")

search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_distributions=param_grid,
    n_iter=50,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

print("\nBest parameters found:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best CV MAE: {-search.best_score_:.3f}")


# =============================
# 9B. FINAL MODEL (BEST PARAMETERS)
# =============================
model = search.best_estimator_

# Re-fit on full training set
# (RandomizedSearchCV already does this internally, 
#  but being explicit makes the code easier to read)
model.fit(X_train, y_train)


# =============================
# 10. PREDICTIONS
# =============================
y_pred       = model.predict(X_test)
y_train_pred = model.predict(X_train)


# =============================
# 11. EVALUATION
# =============================
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
r2   = model.score(X_test, y_test)  # R² score

print("\n===== TEST PERFORMANCE =====")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAPE: {mape:.2f}%")
print(f"R²:   {r2:.4f}")



# =============================
# 12. OVERFITTING CHECK
# =============================
train_mae  = mean_absolute_error(y_train, y_train_pred)
test_mae   = mean_absolute_error(y_test, y_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
train_mape = mean_absolute_percentage_error(y_train, y_train_pred) * 100
test_mape  = mean_absolute_percentage_error(y_test, y_pred) * 100
overfit_ratio_mae  = train_mae / test_mae
overfit_ratio_rmse = train_rmse / test_rmse
overfit_ratio_mape = train_mape / test_mape

cv_mae = -search.best_score_                         # nuevo: MAE de validación (out-of-sample dentro del train)


print("\n===== OVERFITTING CHECK =====")
print(f"Train MAE (in-sample): {train_mae:.3f} | CV MAE (validación): {cv_mae:.3f} | Test MAE: {test_mae:.3f}")
print(f"Train RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}")
print(f"Train MAPE: {train_mape:.2f}% | Test MAPE: {test_mape:.2f}%")
print(f"Overfit ratio (MAE):  {overfit_ratio_mae:.2f}x")
print(f"Overfit ratio (RMSE): {overfit_ratio_rmse:.2f}x")
print(f"Overfit ratio (MAPE): {overfit_ratio_mape:.2f}x")



# =============================
# 13. FEATURE IMPORTANCE
# =============================
importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print("\n===== TOP FEATURES =====")
print(importance.head(10))


# =============================
# 13B. SAVE MODEL PARAMETERS, METRICS, OVERFITTING CHECK, TOP FEATURES
# =============================
# 13B.1 Best hyperparameters (JSON, handles the 'None' / numpy types cleanly)
best_params_serializable = {
    k: (v if not isinstance(v, (np.integer, np.floating)) else v.item())
    for k, v in search.best_params_.items()
}
with open(os.path.join(output_dir, f"best_params_{target}.json"), "w") as f:
    json.dump(
        {
            "best_params": best_params_serializable,
            "best_cv_mae": float(-search.best_score_),
        },
        f,
        indent=4,
    )

# 13B.2 Test-set metrics + overfitting check, combined in one small CSV
metrics_df = pd.DataFrame([
    {"metric": "MAE",  "train": train_mae,  "cv": cv_mae, "test": test_mae,  "overfit_ratio": overfit_ratio_mae},
    {"metric": "RMSE", "train": train_rmse, "cv": np.nan, "test": test_rmse, "overfit_ratio": overfit_ratio_rmse},
    {"metric": "MAPE", "train": train_mape, "cv": np.nan, "test": test_mape, "overfit_ratio": overfit_ratio_mape},
])
metrics_df.to_csv(os.path.join(output_dir, f"metrics_overfitting_{target}.csv"), index=False)

# 13B.3 Full feature importance table (not just top 10/15)
importance_df = importance.rename("importance").reset_index().rename(columns={"index": "feature"})
importance_df.to_csv(os.path.join(output_dir, f"feature_importance_{target}.csv"), index=False)


# =============================
# 14. PLOT: OVERFITTING CHECK
# =============================
plt.figure(figsize=(14,5))
plt.plot(y_train.index, y_train,       label='Train Real', alpha=0.5)
plt.plot(y_train.index, y_train_pred,  label='Train Pred', alpha=0.5)
plt.plot(y_test.index,  y_test,        label='Test Real',  linewidth=2)
plt.plot(y_test.index,  y_pred,        label='Test Pred',  linewidth=2)
plt.legend()
plt.title(f"Random Forest - Overfitting Check ({target})")
plt.xlabel("Date")
plt.ylabel(target)
plt.grid()
plt.savefig(f"{output_dir}/overfitting_check_{target}.png", dpi=300)
plt.close()


# =============================
# 15. PLOT: TEST ONLY
# =============================
plt.figure(figsize=(12,5))
plt.plot(y_test.index, y_test,  label='Test Real',      linewidth=2)
plt.plot(y_test.index, y_pred,  label='Test Predicted', linewidth=2)
plt.title(f"Random Forest - Test Set Only ({target})")
plt.xlabel("Date")
plt.ylabel(target)
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/test_only_{target}.png", dpi=300)
plt.close()


# =============================
# 16. PLOT: FEATURE IMPORTANCE
# =============================
plt.figure(figsize=(10,5))
importance.head(5).plot(kind='bar')
plt.title(f"Feature Importance ({target})")
plt.ylabel("Importance")
plt.grid()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance_{target}.png", dpi=300)
plt.close()


# =============================================================
# 17. MULTI-HORIZON ROLLING RECURSIVE FORECAST (1, 3, 7 DAYS)
# =============================================================
# Strategy:
#   - Walk through the test period in 7-day jumps (52 windows).
#   - At each origin, forecast day 1..7 one at a time:
#       * weather features (same-day AND lagged) are ALWAYS taken from
#         the true data -- future weather is assumed known, same
#         assumption as the ARIMAX/exog version.
#       * target lag features are taken from the TRUE observed series
#         whenever the lag date falls BEFORE the origin, and from this
#         window's own PREDICTIONS whenever the lag date falls INSIDE
#         the current 7-day window. That's the actual recursive step --
#         day 3's "NO2_lag1" feature is day 2's prediction, day 2's
#         "NO2_lag1" feature is day 1's prediction, etc.
#   - Accuracy is scored at day 1, day 3 and day 7 of each window.
#   - Every new window restarts the recursion from real observed data
#     only (predictions never leak across windows), the same principle
#     as the state-update step in the SARIMAX version.

HORIZONS = [1, 3, 7]
N_WINDOWS = 52
STEP = 7


def recursive_forecast_rf(model, df, target, feature_cols, lags, origin_idx, horizon=STEP):
    """
    Recursively forecast `horizon` days starting at df.index[origin_idx].
    Returns a dict {date: predicted_value}.
    """
    known_target = df[target]
    preds = {}

    for h in range(horizon):
        current_idx = origin_idx + h
        current_date = df.index[current_idx]

        row = {}
        for col in feature_cols:
            if col.startswith(f"{target}_lag"):
                lag = int(col.split("lag")[-1])
                lag_idx = current_idx - lag
                lag_date = df.index[lag_idx]
                row[col] = preds.get(lag_date, known_target.loc[lag_date])
            elif "_lag" in col:
                base_col, lag_str = col.rsplit("_lag", 1)
                lag = int(lag_str)
                lag_idx = current_idx - lag
                lag_date = df.index[lag_idx]
                row[col] = df.loc[lag_date, base_col]   # weather lag -> always known
            else:
                row[col] = df.loc[current_date, col]     # same-day weather -> always known

        X_row = pd.DataFrame([row], index=[current_date])[feature_cols]
        y_hat = model.predict(X_row)[0]
        preds[current_date] = y_hat

    return preds

model.n_jobs = 1
base_origin_idx = df.index.get_loc(pd.Timestamp(split_date))

rows = []
origin_idx = base_origin_idx

for week in range(N_WINDOWS):
    if origin_idx + STEP > len(df):
        print(f"Stopping at week {week}: not enough data left for a full 7-day window.")
        break

    origin_date = df.index[origin_idx - 1]

    preds = recursive_forecast_rf(model, df, target, FEATURE_COLS, LAGS, origin_idx, horizon=STEP)

    for h in HORIZONS:
        target_date = df.index[origin_idx + h - 1]
        y_hat = preds[target_date]
        y_true = df.loc[target_date, target]
        abs_pct_error = abs((y_true - y_hat) / y_true) if y_true != 0 else np.nan
        rows.append({
            "origin_date": origin_date,
            "target_date": target_date,
            "horizon": h,
            "actual": y_true,
            "predicted": y_hat,
            "abs_error": abs(y_true - y_hat),
            "sq_error": (y_true - y_hat) ** 2,
            "abs_pct_error": abs_pct_error
        })

    origin_idx += STEP

horizon_results = pd.DataFrame(rows)

# =============================
# 18. METRICS PER HORIZON
# =============================
summary = horizon_results.groupby("horizon").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("abs_pct_error", lambda s: s.mean() * 100),
    n=("abs_error", "count")
).reset_index()

print("\n=== Multi-Horizon Rolling Forecast Summary (52 windows, 7-day jumps) ===")
print(summary)

# Per-window (every origin x horizon row) results -> one CSV
horizon_results.to_csv(os.path.join(output_dir, f"multi_horizon_results_by_window_{target}.csv"), index=False)
# Aggregated overall results per horizon -> a separate CSV
summary.to_csv(os.path.join(output_dir, f"multi_horizon_summary_overall_{target}.csv"), index=False)

# =============================
# 19. PLOTS: ERROR BY HORIZON
# =============================
plt.figure(figsize=(8, 5))
sns.boxplot(data=horizon_results, x="horizon", y="abs_error")
plt.title(f"{target} - Absolute Error by Forecast Horizon (52 rolling windows)")
plt.xlabel("Horizon (days ahead)")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_abs_error_boxplot.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="horizon", y="RMSE")
plt.title(f"{target} - RMSE by Forecast Horizon")
plt.xlabel("Horizon (days ahead)")
plt.ylabel("RMSE")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_rmse_bar.png"))
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="horizon", y="MAPE")
plt.title(f"{target} - MAPE by Forecast Horizon")
plt.xlabel("Horizon (days ahead)")
plt.ylabel("MAPE (%)")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_mape_bar.png"))
plt.close()

# =============================
# 20. PLOT: WEEKLY MAE AND RMSE
# =============================
# Calculate MAE and RMSE for each rolling week
# across the 1-day, 3-day and 7-day forecast horizons.

weekly_metrics = (
    horizon_results
    .groupby("origin_date")
    .agg(
        MAE=("abs_error", "mean"),
        RMSE=("sq_error", lambda s: np.sqrt(s.mean()))
    )
    .reset_index()
)

# Assign sequential week numbers: 1, 2, ..., 52
weekly_metrics["week"] = np.arange(1, len(weekly_metrics) + 1)

# Plot MAE and RMSE
plt.figure(figsize=(12, 6))

plt.plot(
    weekly_metrics["week"],
    weekly_metrics["MAE"],
    marker="o",
    label="MAE"
)

plt.plot(
    weekly_metrics["week"],
    weekly_metrics["RMSE"],
    marker="o",
    label="RMSE"
)

plt.xlabel("Week Number")
plt.ylabel("Error Value")
plt.title(f"Random Forest - Weekly MAE and RMSE ({target})")

plt.xticks(
    range(1, len(weekly_metrics) + 1, 2)
)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        f"weekly_mae_rmse_{target}.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Weekly MAE/RMSE plot saved to: "
    f"{os.path.join(output_dir, f'weekly_mae_rmse_{target}.png')}"
)



# =============================================================
# 20B. MULTI-HORIZON ROLLING RECURSIVE FORECAST, DAILY ORIGIN (robustness check)
# =============================================================
# Same recursive procedure as section 17, but the forecast origin advances ONE day
# at a time, so each horizon falls on every day of the week instead of always the
# same weekday. The model is not retrained; only the origin moves.
HORIZONS_D, STEP_D, WINDOW_D = [1, 3, 7], 1, 7

rows_d = []
origin_idx = base_origin_idx
w = 0

while origin_idx + WINDOW_D <= len(df):
    origin_date = df.index[origin_idx - 1]
    persist = df.loc[origin_date, target]          # last observed value before the window

    preds = recursive_forecast_rf(model, df, target, FEATURE_COLS, LAGS, origin_idx, horizon=WINDOW_D)

    for h in HORIZONS_D:
        target_date = df.index[origin_idx + h - 1]
        y_hat = preds[target_date]
        y_true = df.loc[target_date, target]
        rows_d.append({
            "window": w + 1,
            "origin_date": origin_date,
            "target_date": target_date,
            "target_weekday": target_date.day_name(),
            "horizon": h,
            "actual": y_true,
            "predicted": y_hat,
            "abs_error": abs(y_true - y_hat),
            "sq_error": (y_true - y_hat) ** 2,
            "abs_pct_error": abs((y_true - y_hat) / y_true) * 100 if y_true != 0 else np.nan,
            "persist_pred": persist,
            "persist_abs_error": abs(y_true - persist),
            "persist_sq_error": (y_true - persist) ** 2,
        })

    origin_idx += STEP_D
    w += 1

hr_d = pd.DataFrame(rows_d)
hr_d.to_csv(os.path.join(output_dir, f"multi_horizon_results_by_window_daily_{target}.csv"), index=False)

summary_d = hr_d.groupby("horizon").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("abs_pct_error", "mean"),
    persistence_MAE=("persist_abs_error", "mean"),
    persistence_RMSE=("persist_sq_error", lambda s: np.sqrt(s.mean())),
    n=("abs_error", "count"),
).reset_index()
summary_d.to_csv(os.path.join(output_dir, f"multi_horizon_summary_daily_{target}.csv"), index=False)
print(f"\n=== Multi-Horizon Rolling Forecast, DAILY origin ({w} windows) ===")
print(summary_d)

# Check 1: each horizon now covers all weekdays (counts should be ~equal)
print("\nTarget weekdays per horizon:")
print(pd.crosstab(hr_d["horizon"], hr_d["target_weekday"]))

# Check 2: h=1 must match the one-step test predictions (diff ~0)
one_step = pd.Series(y_pred, index=y_test.index)
h1 = hr_d[hr_d["horizon"] == 1].set_index("target_date")["predicted"]
print("Max diff h=1 vs 1-step:", (h1 - one_step.loc[h1.index]).abs().max())

# MAE by horizon and target weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
mae_wd = hr_d.pivot_table(index="horizon", columns="target_weekday",
                          values="abs_error", aggfunc="mean")[weekday_order]
mae_wd.to_csv(os.path.join(output_dir, f"multi_horizon_mae_by_weekday_daily_{target}.csv"))
print("\nMAE by horizon and target weekday:")
print(mae_wd.round(3))

plt.figure(figsize=(8, 5))
sns.boxplot(data=hr_d, x="horizon", y="abs_error")
plt.title(f"{target} - Absolute Error by Horizon (daily origin, {w} windows)")
plt.xlabel("Horizon (days ahead)")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_abs_error_boxplot_daily.png"), bbox_inches="tight")
plt.close()



print(f"\nAll results saved under: {output_dir}/")
print("  - best_params_{t}.json          (final model hyperparameters + best CV MAE)".format(t=target))
print("  - metrics_overfitting_{t}.csv    (train/test MAE, RMSE, MAPE + overfit ratios)".format(t=target))
print("  - feature_importance_{t}.csv     (full feature importance table)".format(t=target))
print("  - multi_horizon_results_by_window_{t}.csv  (per-window horizon results)".format(t=target))
print("  - multi_horizon_summary_overall_{t}.csv    (aggregated results per horizon)".format(t=target))
print("  - *.png                          (all plots)")