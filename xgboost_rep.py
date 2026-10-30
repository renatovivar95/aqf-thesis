# =============================
# 1. IMPORTS
# =============================
import os
import warnings
import argparse
# import pyodbc   # only needed for the (commented-out) local SQL Server connection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import r2_score    

warnings.filterwarnings("ignore", category=UserWarning)


# =============================
# 2. CONFIGURATION
# =============================
# Usage example:
#   python xgboost_model.py --pollutant O3
#   python xgboost_model.py --pollutant NO2 --data aveiro_2018_2022_final_dataset.csv
parser = argparse.ArgumentParser(description="XGBoost forecasting for Aveiro air quality.")
parser.add_argument("--pollutant", type=str, default="O3",
                    help="Target pollutant column in the dataset: 'NO2', 'O3' or 'PM10' (default: O3)")
parser.add_argument("--data", type=str, default="aveiro_2018_2022_final_dataset.csv",
                    help="Path to the input CSV dataset (default: aveiro_2018_2022_final_dataset.csv)")
parser.add_argument("--weather", type=str, default="aveiro_weather_2018_2022.csv",
                    help="Path to the weather CSV with Precipitation and Radiation, used only if "
                         "those columns are not already in the main dataset")
args = parser.parse_args()

target = args.pollutant         # 'NO2', 'O3' or 'PM10'

output_dir = f"xgboost_{target}_F2"
os.makedirs(output_dir, exist_ok=True)

split_date = '2022-01-01'       # start of test period
val_cutoff = '2021-07-01'       # early-stopping validation block (last 6 months of train)

LAGS          = [1, 2, 3, 7]
HORIZON       = 7               # days per rolling window
EVAL_HORIZONS = [1, 3, 7]       # horizons scored in the multi-horizon forecast

RUN_GRID_SEARCH = True          # False -> use DEFAULT_PARAMS directly

# Search space (54 combinations x 5 folds)
PARAM_GRID = {
    'learning_rate':    [0.03, 0.05, 0.1],
    'max_depth':        [3, 4, 6],
    'min_child_weight': [5, 10, 20],
    'subsample':        [0.8],
    'colsample_bytree': [0.8],
    'reg_alpha':        [0.1],
    'reg_lambda':       [1.0, 2.0],
}
GRID_N_ESTIMATORS = 500          # fixed tree count during the grid search

# Used only if RUN_GRID_SEARCH = False
DEFAULT_PARAMS = {
    'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 10,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'reg_alpha': 0.1, 'reg_lambda': 1.0,
}

BASE_KWARGS = dict(random_state=42, n_jobs=-1, verbosity=0)


# =============================
# 2B. METRIC HELPERS
# =============================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


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
#         AND i.item_name IN ('{target}', 'Temperature', 'WindSpeed')
#
#     GROUP BY 
#         CAST(t.time_key AS DATE),
#         s.district,
#         i.item_name
# ) AS source
#
# PIVOT (
#     AVG(avg_value)
#     FOR item_name IN ([{target}], [Temperature], [WindSpeed])
# ) AS pivot_table
#
# ORDER BY date;
# """
#
#
# # --- 5. LOAD DATA ---
# df = pd.read_sql(query, conn)
# conn.close()

# --- Public version: load the same data from a CSV file ---
raw = pd.read_csv(args.data)
raw['date'] = pd.to_datetime(raw['date'])

if 'district' in raw.columns:
    raw = raw[raw['district'].astype(str).str.lower() == 'aveiro']
raw = raw[(raw['date'] >= '2018-01-01') & (raw['date'] <= '2022-12-31')]

required_cols = [target, 'Temperature', 'WindSpeed']
missing_cols = [c for c in required_cols if c not in raw.columns]
if missing_cols:
    raise ValueError(f"Columns {missing_cols} not found in {args.data}. "
                     f"Available columns: {raw.columns.tolist()}")

df = raw[['date'] + required_cols].groupby('date', as_index=False).mean()   # one row per day, as in the SQL AVG/GROUP BY

df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').set_index('date')
df = df.asfreq("D")


# =============================
# 6. HANDLE MISSING VALUES
# =============================
sum_nulls = df.isnull().sum()
threshold = len(df) * 0.1
cols_to_drop = sum_nulls[sum_nulls > threshold].index
if target in cols_to_drop:
    raise ValueError(f"Target {target} has more than 10% missing values.")
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
# 6C. INTERACTION FEATURES (O3 ONLY)
# =============================
if target == 'O3':
    required = {'Temperature', 'WindSpeed', 'Radiation'}
    missing = required - set(df.columns)
    if missing:
        print(f"WARNING: O3 interaction features skipped, missing columns: {missing}")
    else:
        print("Target is O3: building photochemical interaction features.")

        # Core photochemical interaction
        df['Temp_x_Radiation'] = df['Temperature'] * df['Radiation']

        # Wind dispersion under radiation
        df['Wind_x_Radiation'] = df['WindSpeed'] * df['Radiation']

        # High radiation flag (top 25% of radiation days).
        # Threshold computed on the TRAINING period only to avoid test leakage.
        rad_threshold = df.loc[df.index < split_date, 'Radiation'].quantile(0.75)
        df['high_rad_day'] = (df['Radiation'] > rad_threshold).astype(int)

        # Radiation accumulated over past 3 days (shift(1) -> past values only)
        df['Radiation_3day_sum'] = df['Radiation'].shift(1).rolling(3).sum()
else:
    print(f"Target is {target}: O3 interaction features skipped.")

# Calendar features (all targets)
df['month']     = df.index.month
df['dayofyear'] = df.index.dayofyear
df['season']    = df['month'].map({
    12: 0, 1: 0, 2: 0,
     3: 1, 4: 1, 5: 1,
     6: 2, 7: 2, 8: 2,
     9: 3, 10: 3, 11: 3
})

CALENDAR_COLS = ['month', 'dayofyear', 'season']


# =============================
# 7. CREATE LAG FEATURES
# =============================
def create_lags(df, lags, exclude=()):
    """Lag every column except those in `exclude` (calendar lags are redundant)."""
    df_lagged = df.copy()
    for col in df.columns:
        if col in exclude:
            continue
        for lag in lags:
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df_lagged


df_lagged = create_lags(df, lags=LAGS, exclude=CALENDAR_COLS)
df_lagged = df_lagged.dropna()


# =============================
# 8. TRAIN-TEST SPLIT
# =============================
train = df_lagged[df_lagged.index < split_date]
test  = df_lagged[df_lagged.index >= split_date]

print(f"\nTrain period: {train.index.min()} → {train.index.max()}")
print(f"Test  period: {test.index.min()} → {test.index.max()}")

X_train = train.drop(columns=[target])
y_train = train[target]

X_test = test.drop(columns=[target])
y_test = test[target]

print(f"Number of features: {X_train.shape[1]}")


# =============================
# 9A. GRID SEARCH WITH TIME SERIES CROSS-VALIDATION
# =============================
tscv  = TimeSeriesSplit(n_splits=5)
folds = list(tscv.split(X_train))


def cv_evaluate(params, n_estimators=GRID_N_ESTIMATORS):
    fold_rmse, fold_mae = [], []
    for tr_idx, va_idx in folds:
        m = XGBRegressor(n_estimators=n_estimators, **params, **BASE_KWARGS)
        m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        p = m.predict(X_train.iloc[va_idx])
        fold_rmse.append(rmse(y_train.iloc[va_idx], p))
        fold_mae.append(mean_absolute_error(y_train.iloc[va_idx], p))
    return fold_rmse, fold_mae


if RUN_GRID_SEARCH:
    grid = list(ParameterGrid(PARAM_GRID))
    print(f"\n===== GRID SEARCH ({len(grid)} combinations x {len(folds)} folds) =====")

    grid_results = []
    for i, params in enumerate(grid, 1):
        fold_rmse, fold_mae = cv_evaluate(params)
        grid_results.append({
            'params': params,
            'mean_rmse': np.mean(fold_rmse),
            'std_rmse':  np.std(fold_rmse),
            'mean_mae':  np.mean(fold_mae),
            'fold_rmse': fold_rmse,
            'fold_mae':  fold_mae,
        })
        if i % 10 == 0 or i == len(grid):
            best_so_far = min(r['mean_rmse'] for r in grid_results)
            print(f"  {i}/{len(grid)} done - best mean RMSE so far: {best_so_far:.3f}")

    best_result = min(grid_results, key=lambda r: r['mean_mae'])
    best_params = best_result['params']
    cv_rmse_scores = best_result['fold_rmse']
    cv_mae_scores  = best_result['fold_mae']
else:
    best_params = DEFAULT_PARAMS
    cv_rmse_scores, cv_mae_scores = cv_evaluate(best_params)

print("\n===== BEST PARAMETERS =====")
for k, v in best_params.items():
    print(f"  {k}: {v}")

print("\n===== CV SCORES (best parameters) =====")
for fold, (r, m) in enumerate(zip(cv_rmse_scores, cv_mae_scores), 1):
    print(f"Fold {fold}: RMSE={r:.3f}, MAE={m:.3f}")

print("\n===== CV SUMMARY =====")
print(f"Mean RMSE: {np.mean(cv_rmse_scores):.3f}")
print(f"Std  RMSE: {np.std(cv_rmse_scores):.3f}")
print(f"Mean MAE:  {np.mean(cv_mae_scores):.3f}")


# =============================
# 9B. FINAL MODEL: EARLY STOPPING TO FIND TREE COUNT, THEN REFIT
# =============================
X_tr  = X_train[X_train.index <  val_cutoff]
y_tr  = y_train[y_train.index <  val_cutoff]
X_val = X_train[X_train.index >= val_cutoff]
y_val = y_train[y_train.index >= val_cutoff]

es_model = XGBRegressor(
    n_estimators=2000,          # high ceiling - early stopping finds the right number
    early_stopping_rounds=50,
    **best_params,
    **BASE_KWARGS
)
es_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)

best_n_trees = es_model.best_iteration + 1
print(f"\nBest number of trees: {best_n_trees}")

# Refit on the FULL training period (incl. the last 6 months) with that tree count
model = XGBRegressor(n_estimators=best_n_trees, **best_params, **BASE_KWARGS)
model.fit(X_train, y_train)

# Save best parameters
best_params_df = pd.DataFrame(
    list(best_params.items()) + [
        ('n_estimators', best_n_trees),
        ('cv_mean_rmse', np.mean(cv_rmse_scores)),
        ('cv_std_rmse',  np.std(cv_rmse_scores)),
        ('cv_mean_mae',  np.mean(cv_mae_scores)),
    ],
    columns=['parameter', 'value']
)
best_params_df.to_csv(f"{output_dir}/best_params_{target}.csv", index=False)


# =============================
# 10. PREDICTIONS
# =============================
y_pred       = model.predict(X_test)
y_train_pred = model.predict(X_train)


# =============================
# 11. EVALUATION
# =============================
test_mae  = mean_absolute_error(y_test, y_pred)
test_rmse = rmse(y_test, y_pred)
test_mape = mape(y_test, y_pred)
test_r2   = r2_score(y_test, y_pred)

print("\n===== TEST PERFORMANCE (one-step, true lags) =====")
print(f"MAE:  {test_mae:.3f}")
print(f"RMSE: {test_rmse:.3f}")
print(f"MAPE: {test_mape:.2f}%")
print(f"R²:   {test_r2:.3f}")


# =============================
# 12. OVERFITTING CHECK
# =============================
train_mae  = mean_absolute_error(y_train, y_train_pred)
train_rmse = rmse(y_train, y_train_pred)
train_mape = mape(y_train, y_train_pred)
train_r2   = r2_score(y_train, y_train_pred)

print("\n===== OVERFITTING CHECK =====")
print(f"Train MAE:  {train_mae:.3f}")
print(f"Test  MAE:  {test_mae:.3f}")
print(f"Train RMSE: {train_rmse:.3f}")
print(f"Test  RMSE: {test_rmse:.3f}")
print(f"Overfit ratio (MAE): {train_mae / test_mae:.2f}x")
print(f"train R²:   {train_r2:.3f}")

_, cv_mae_final = cv_evaluate(best_params, n_estimators=best_n_trees)
cv_mae = np.mean(cv_mae_final)
print(f"Train MAE (in-sample): {train_mae:.3f} | CV MAE (validación): {cv_mae:.3f} | Test MAE: {test_mae:.3f}")


metrics_df = pd.DataFrame({
    'metric':      ['MAE', 'RMSE', 'MAPE_%', 'R²'],
    'train':       [train_mae, train_rmse, train_mape, train_r2],
    'test':        [test_mae, test_rmse, test_mape, test_r2],
    'cv':          [cv_mae, np.nan, np.nan, np.nan],
})
metrics_df['train_test_ratio'] = metrics_df['train'] / metrics_df['test']
metrics_df.to_csv(f"{output_dir}/metrics_overfitting_{target}.csv", index=False)


# =============================
# 13. FEATURE IMPORTANCE
# =============================
importance = pd.Series(model.feature_importances_, index=X_train.columns)
importance = importance.sort_values(ascending=False)

print("\n===== TOP FEATURES =====")
print(importance.head(10))

importance.rename('importance').rename_axis('feature').reset_index().to_csv(
    f"{output_dir}/feature_importance_{target}.csv", index=False
)


# =============================
# 14. PLOT: OVERFITTING CHECK
# =============================
plt.figure(figsize=(14, 5))
plt.plot(y_train.index, y_train,      label='Train Real', alpha=0.5)
plt.plot(y_train.index, y_train_pred, label='Train Pred', alpha=0.5)
plt.plot(y_test.index,  y_test,       label='Test Real',  linewidth=2)
plt.plot(y_test.index,  y_pred,       label='Test Pred',  linewidth=2)
plt.legend()
plt.title(f"XGBoost - Overfitting Check ({target})")
plt.xlabel("Date")
plt.ylabel(target)
plt.grid()
plt.savefig(f"{output_dir}/overfitting_check_{target}.png", dpi=300)
plt.close()


# =============================
# 15. PLOT: TEST ONLY
# =============================
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, y_test, label='Test Real',      linewidth=2)
plt.plot(y_test.index, y_pred, label='Test Predicted', linewidth=2)
plt.title(f"XGBoost - Test Set Only ({target})")
plt.xlabel("Date")
plt.ylabel(target)
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/test_only_{target}.png", dpi=300)
plt.close()

# Save the 1-step predictions to CSV for further analysis
pd.DataFrame({
    "date": y_test.index,
    "actual": y_test.values,
    "predicted": y_pred,
}).to_csv(f"{output_dir}/1step_predictions_{target}.csv", index=False)


# =============================
# 16. PLOT: FEATURE IMPORTANCE
# =============================
plt.figure(figsize=(10, 5))
importance.head(15).plot(kind='bar')
plt.title(f"Feature Importance ({target})")
plt.ylabel("Importance")
plt.grid()
plt.xticks(rotation=45, ha='right')
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
#         day 3's "{target}_lag1" feature is day 2's prediction, day 2's
#         "{target}_lag1" feature is day 1's prediction, etc.
#   - Accuracy is scored at day 1, day 3 and day 7 of each window.
#   - "origin" = first forecast day; information available up to origin - 1.
# =============================================================
print("\n===== MULTI-HORIZON ROLLING RECURSIVE FORECAST =====")

target_lag_cols = {lag: f"{target}_lag{lag}" for lag in LAGS}
test_dates = X_test.index
n_windows  = len(test_dates) // HORIZON
print(f"Windows: {n_windows} (step = {HORIZON} days)")

horizon_rows = []

for w in range(n_windows):
    origin = test_dates[w * HORIZON]
    window_preds = {}

    for h in range(1, HORIZON + 1):
        day = origin + pd.Timedelta(days=h - 1)
        if day not in X_test.index:
            break

        x_row = X_test.loc[[day]].copy()   # true weather + true lags by default

        # Overwrite target lags that fall inside the current window
        for lag, col in target_lag_cols.items():
            lag_date = day - pd.Timedelta(days=lag)
            if lag_date >= origin:
                x_row[col] = window_preds[lag_date]

        y_hat = float(model.predict(x_row)[0])
        window_preds[day] = y_hat

        if h in EVAL_HORIZONS:
            actual = float(y_test.loc[day])
            horizon_rows.append({
                'window':    w + 1,
                'origin':    origin.date(),
                'horizon':   h,
                'date':      day.date(),
                'actual':    actual,
                'predicted': y_hat,
                'error':     actual - y_hat,
                'abs_error': abs(actual - y_hat),
                'ape_%':     abs(actual - y_hat) / abs(actual) * 100 if actual != 0 else np.nan,
            })

horizon_df = pd.DataFrame(horizon_rows)
horizon_df.to_csv(f"{output_dir}/multi_horizon_results_by_window_{target}.csv", index=False)

# Sanity check: horizon-1 must match the one-step test predictions
h1 = horizon_df[horizon_df['horizon'] == 1]
one_step = pd.Series(y_pred, index=y_test.index).loc[pd.to_datetime(h1['date'])].values
print(f"Horizon-1 vs one-step max diff (should be ~0): {np.max(np.abs(h1['predicted'].values - one_step)):.6f}")

summary_rows = []
for h in EVAL_HORIZONS:
    sub = horizon_df[horizon_df['horizon'] == h]
    summary_rows.append({
        'horizon_days': h,
        'n_windows':    len(sub),
        'MAE':          mean_absolute_error(sub['actual'], sub['predicted']),
        'RMSE':         rmse(sub['actual'], sub['predicted']),
        'MAPE_%':       mape(sub['actual'], sub['predicted']),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{output_dir}/multi_horizon_summary_overall_{target}.csv", index=False)

print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# =============================
# 18. PLOTS: MULTI-HORIZON
# =============================

labels = [f"Day {h}" for h in EVAL_HORIZONS]

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=horizon_df,
    x="horizon",
    y="abs_error"
)

plt.title(f"{target} - XGBoost: Absolute error by horizon (52 windows)")
plt.xlabel("Horizon (days)")
plt.ylabel("Absolute error")
plt.grid(True)

plt.savefig(
    os.path.join(output_dir, "horizon_abs_error_boxplot.png"),
    bbox_inches="tight"
)

plt.close()

# MAPE bar
plt.figure(figsize=(7, 5))
bars = plt.bar(labels, summary_df['MAPE_%'])
plt.bar_label(bars, fmt='%.1f%%')
plt.title(f"XGBoost - MAPE by Horizon ({target})")
plt.ylabel("MAPE (%)")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/horizon_mape_bar.png", dpi=300)
plt.close()

# RMSE bar
plt.figure(figsize=(7, 5))
bars = plt.bar(labels, summary_df['RMSE'])
plt.bar_label(bars, fmt='%.2f')
plt.title(f"XGBoost - RMSE by Horizon ({target})")
plt.ylabel(f"RMSE ({target})")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/horizon_rmse_bar.png", dpi=300)
plt.close()

print(f"\nAll outputs saved to: {output_dir}/")

# =============================
# PLOT: WEEKLY MAE / RMSE ON TEST SET
# =============================
err_df = pd.DataFrame({'actual': y_test, 'pred': y_pred}, index=y_test.index)
err_df['abs_err'] = (err_df['actual'] - err_df['pred']).abs()
err_df['sq_err']  = (err_df['actual'] - err_df['pred']) ** 2

weekly = horizon_df.groupby('window').agg(
    MAE=('abs_error', 'mean'),
    RMSE=('error', lambda e: np.sqrt((e ** 2).mean()))
).reset_index()

plt.figure(figsize=(12, 5))
plt.plot(weekly.index, weekly['MAE'],  marker='o', label='Weekly MAE')
plt.plot(weekly.index, weekly['RMSE'], marker='s', label='Weekly RMSE')
plt.title(f"XGBoost - Weekly Test Error ({target})")
plt.xlabel("Week")
plt.ylabel(f"Error ({target})")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"{output_dir}/weekly_mae_rmse_{target}.png", dpi=300)
plt.close()


# =============================================================
# 18B. MULTI-HORIZON ROLLING RECURSIVE FORECAST, DAILY ORIGIN (robustness check)
# =============================================================
# Same recursive procedure as section 17, but the forecast origin advances ONE day
# at a time, so each horizon falls on every day of the week.
print("\n===== MULTI-HORIZON ROLLING RECURSIVE FORECAST, DAILY ORIGIN =====")

rows_d = []
n_windows_d = len(test_dates) - HORIZON + 1          # every origin with 7 days ahead

for w in range(n_windows_d):
    origin = test_dates[w]
    origin_prev = origin - pd.Timedelta(days=1)
    persist = float(df.loc[origin_prev, target])     # last observed value before the window
    window_preds = {}

    for h in range(1, HORIZON + 1):
        day = origin + pd.Timedelta(days=h - 1)
        x_row = X_test.loc[[day]].copy()             # true weather + true lags by default

        # overwrite target lags that fall inside the current window
        for lag, col in target_lag_cols.items():
            lag_date = day - pd.Timedelta(days=lag)
            if lag_date >= origin:
                x_row[col] = window_preds[lag_date]

        y_hat = float(model.predict(x_row)[0])
        window_preds[day] = y_hat

        if h in EVAL_HORIZONS:
            actual = float(y_test.loc[day])
            rows_d.append({
                'window': w + 1,
                'origin': origin.date(),
                'horizon': h,
                'date': day,
                'target_weekday': day.day_name(),
                'actual': actual,
                'predicted': y_hat,
                'abs_error': abs(actual - y_hat),
                'sq_error': (actual - y_hat) ** 2,
                'ape_%': abs(actual - y_hat) / abs(actual) * 100 if actual != 0 else np.nan,
                'persist_abs_error': abs(actual - persist),
                'persist_sq_error': (actual - persist) ** 2,
            })

hr_d = pd.DataFrame(rows_d)
hr_d.to_csv(f"{output_dir}/multi_horizon_results_by_window_daily_{target}.csv", index=False)

summary_d = hr_d.groupby('horizon').agg(
    MAE=('abs_error', 'mean'),
    RMSE=('sq_error', lambda s: np.sqrt(s.mean())),
    MAPE=('ape_%', 'mean'),
    persistence_MAE=('persist_abs_error', 'mean'),
    persistence_RMSE=('persist_sq_error', lambda s: np.sqrt(s.mean())),
    n=('abs_error', 'count'),
).reset_index()
summary_d.to_csv(f"{output_dir}/multi_horizon_summary_daily_{target}.csv", index=False)
print(f"=== Daily origin ({n_windows_d} windows) ===")
print(summary_d)

# Check 1: each horizon covers all weekdays
print("\nTarget weekdays per horizon:")
print(pd.crosstab(hr_d['horizon'], hr_d['target_weekday']))

# Check 2: h=1 must match the one-step test predictions (diff ~0)
h1 = hr_d[hr_d['horizon'] == 1].set_index('date')['predicted']
one_step = pd.Series(y_pred, index=y_test.index)
print("Max diff h=1 vs 1-step:", (h1 - one_step.loc[h1.index]).abs().max())

# MAE by horizon and target weekday
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
mae_wd = hr_d.pivot_table(index='horizon', columns='target_weekday',
                          values='abs_error', aggfunc='mean')[weekday_order]
mae_wd.to_csv(f"{output_dir}/multi_horizon_mae_by_weekday_daily_{target}.csv")
print("\nMAE by horizon and target weekday:")
print(mae_wd.round(3))