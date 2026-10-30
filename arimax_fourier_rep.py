import os
import argparse
# import pyodbc   # only needed for the (commented-out) local SQL Server connection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
# Usage example:
#   python arimax_model.py --pollutant PM10
#   python arimax_model.py --pollutant NO2 --data aveiro_2018_2022_final_dataset.csv
parser = argparse.ArgumentParser(description="ARIMAX (SARIMAX) forecasting for Aveiro air quality.")
parser.add_argument("--pollutant", type=str, default="PM10",
                    help="Target pollutant column in the dataset (default: PM10)")
parser.add_argument("--data", type=str, default="aveiro_2018_2022_final_dataset.csv",
                    help="Path to the input CSV dataset (default: aveiro_2018_2022_final_dataset.csv)")
parser.add_argument("--weather", type=str, default="aveiro_weather_2018_2022.csv",
                    help="Path to the weather CSV with Precipitation and Radiation, used only if "
                         "those columns are not already in the main dataset")
args = parser.parse_args()

TARGET = args.pollutant
LOG_TARGETS = ["NO2", "PM10", "O3"]
DISTRICT = "aveiro"
START_DATE = "2018-01-01"
END_DATE = "2022-12-31"

# Split BY DATE (not by percentage)
TRAIN_END = "2021-12-31"
TEST_START = "2022-01-01"

BURN_IN = 30          # initial training days excluded from metrics (diffuse initialization)
N_CV_SPLITS = 5       # time-series CV ONLY within 2018-2021
CV_TEST_SIZE = 182    # ~6 months per validation fold

output_dir = f"arimax_model_results_{TARGET}_F2"
os.makedirs(output_dir, exist_ok=True)

# ==========================================================
# 🔌 DATA LOADING
# ==========================================================
# --- Original version: local SQL Server data warehouse (not available to other users) ---
# conn = pyodbc.connect(
#     "DRIVER={ODBC Driver 18 for SQL Server};"
#     "SERVER=DESKTOP-FRDCJFV;"
#     "DATABASE=aqf_dw;"
#     "Trusted_Connection=yes;"
#     "TrustServerCertificate=yes;"
# )
#
# ALL_VARS = [TARGET, "Temperature", "WindSpeed"]
# pivot_cols = ",".join(f"[{c}]" for c in ALL_VARS)
# in_clause = ",".join(f"'{c}'" for c in ALL_VARS)
#
# query = f"""
# SELECT * FROM (
#     SELECT CAST(t.time_key AS DATE) AS date, s.district, i.item_name, AVG(f.value) AS avg_value
#     FROM dbo.fact_air_quality_measurement f
#     JOIN dbo.dim_time t     ON f.time_key = t.time_key
#     JOIN dbo.dim_location s ON f.location_key = s.location_key
#     JOIN dbo.dim_item i     ON f.item_key = i.item_key
#     WHERE s.district = '{DISTRICT}'
#       AND CAST(t.time_key AS DATE) BETWEEN '{START_DATE}' AND '{END_DATE}'
#       AND i.item_name IN ({in_clause})
#     GROUP BY CAST(t.time_key AS DATE), s.district, i.item_name
# ) AS source
# PIVOT (AVG(avg_value) FOR item_name IN ({pivot_cols})) AS pivot_table
# ORDER BY date;
# """
#
# df = pd.read_sql(query, conn)

# --- Public version: load the same data from a CSV file ---
ALL_VARS = [TARGET, "Temperature", "WindSpeed"]
raw = pd.read_csv(args.data)
raw["date"] = pd.to_datetime(raw["date"])

if "district" in raw.columns:
    raw = raw[raw["district"].astype(str).str.lower() == DISTRICT]
raw = raw[(raw["date"] >= START_DATE) & (raw["date"] <= END_DATE)]

missing = [c for c in ALL_VARS if c not in raw.columns]
if missing:
    raise ValueError(f"Columns {missing} not found in {args.data}. "
                     f"Available columns: {raw.columns.tolist()}")

df = raw[["date"] + ALL_VARS].groupby("date", as_index=False).mean()   # one row per day, as in the SQL AVG/GROUP BY
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date").asfreq("D")

nulls = df.isnull().sum()
df = df.drop(columns=nulls[nulls > len(df) * 0.1].index)
if "district" in df.columns:
    df = df.drop(columns=["district"])
df = df.interpolate(method="time").ffill().bfill().dropna()

# Precipitation and Radiation: taken from the main dataset if present, otherwise from the weather CSV
EXTRA_WEATHER = ["Precipitation", "Radiation"]
if all(c in raw.columns for c in EXTRA_WEATHER):
    weather_df = raw[["date"] + EXTRA_WEATHER].groupby("date").mean()
else:
    weather_df = pd.read_csv(args.weather, parse_dates=["date"]).set_index("date")[EXTRA_WEATHER]
weather_df = weather_df.reindex(df.index).interpolate(method="time").ffill().bfill()
df = df.join(weather_df)

EXOG_FEATURES = ["Temperature", "WindSpeed", "Precipitation", "Radiation"]

# ==========================================================
# 🔁 FOURIER (deterministic, no leakage)
# ==========================================================
def create_fourier_terms(index, period, K):
    t = np.arange(len(index))
    out = pd.DataFrame(index=index)
    for k in range(1, K + 1):
        out[f"sin_{k}_{period}"] = np.sin(2 * np.pi * k * t / period)
        out[f"cos_{k}_{period}"] = np.cos(2 * np.pi * k * t / period)
    return out

exog = pd.concat([
    df[EXOG_FEATURES],
    create_fourier_terms(df.index, 365, 3),
    create_fourier_terms(df.index, 7, 1),
], axis=1)

if TARGET in LOG_TARGETS:
    y = np.log1p(df[TARGET]); inv = np.expm1
else:
    y = df[TARGET]; inv = lambda x: x

# ==========================================================
# ✂️ DATE-BASED SPLIT
# ==========================================================
y_train, y_test = y[:TRAIN_END], y[TEST_START:]
exog_train, exog_test = exog.loc[y_train.index], exog.loc[y_test.index]
print(f"Train: {y_train.index.min().date()} → {y_train.index.max().date()} ({len(y_train)} days)")
print(f"Test:  {y_test.index.min().date()} → {y_test.index.max().date()} ({len(y_test)} days)")

def metrics(actual, pred):
    return {
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(mean_squared_error(actual, pred)),
        "MAPE": mean_absolute_percentage_error(actual, pred) * 100,
        "R2":   r2_score(actual, pred),
    }

def fit_sarimax(y_, X_, order):
    return SARIMAX(y_, exog=X_, order=order, seasonal_order=(0, 0, 0, 0),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

def one_step_predictions(fitted, y_new, X_new):
    """1-step-ahead prediction over y_new with frozen parameters.
    The state continues from the end of the fitting period and is updated
    with each ACTUAL observed value (parameters are not re-estimated)."""
    ext = fitted.extend(y_new, exog=X_new)
    return ext.fittedvalues

# ==========================================================
# 🔎 1. SELECTION OF (p,d,q): TIME-SERIES CV ONLY ON 2018-2021
# ==========================================================
# d restricted to {0,1}: d>=2 over-differences a daily log series.
orders = list(product(range(0, 4), range(0, 2), range(0, 4)))
tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS, test_size=CV_TEST_SIZE)

grid_rows = []
print(f"\n=== Grid search with time-series CV on train ({len(orders)} orders × {N_CV_SPLITS} folds) ===")
for order in orders:
    fold_m = []
    try:
        for tr_idx, va_idx in tscv.split(y_train):
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            X_tr, X_va = exog_train.iloc[tr_idx], exog_train.iloc[va_idx]
            res = fit_sarimax(y_tr, X_tr, order)
            pred = np.clip(inv(one_step_predictions(res, y_va, X_va)), 0, None)
            fold_m.append(metrics(inv(y_va), pred))
        m = pd.DataFrame(fold_m)
        grid_rows.append({"order": order, "p": order[0], "d": order[1], "q": order[2],
                          "CV_MAE_mean": m["MAE"].mean(), "CV_MAE_std": m["MAE"].std(),
                          "CV_RMSE_mean": m["RMSE"].mean(), "CV_MAPE_mean": m["MAPE"].mean()})
        print(f"{order}: CV MAE={m['MAE'].mean():.4f} ± {m['MAE'].std():.4f}")
    except Exception as e:
        print(f"{order}: FAILED -> {e}")

grid_df = pd.DataFrame(grid_rows).sort_values("CV_MAE_mean").reset_index(drop=True)
grid_df.to_csv(os.path.join(output_dir, "sarimax_grid_search_cv_train.csv"), index=False)
best_order = grid_df.loc[0, "order"]
print(f"\nBest order (selected WITHOUT seeing 2022): {best_order}")

# ==========================================================
# 🎯 2. FINAL MODEL: refit on all of 2018-2021
# ==========================================================
results = fit_sarimax(y_train, exog_train, best_order)
print(results.summary())
with open(os.path.join(output_dir, "sarimax_model_summary.txt"), "w", encoding="utf-8") as f:
    f.write(results.summary().as_text())

importance_df = pd.DataFrame({
    "Feature": exog.columns,
    "Coefficient": results.params[exog.columns].values,
    "Std_Error": results.bse[exog.columns].values,
    "P_Value": results.pvalues[exog.columns].values,
}).assign(Importance=lambda d: d["Coefficient"].abs()).sort_values("Importance", ascending=False)

importance_df["Importance"] = (importance_df["Coefficient"] / importance_df["Std_Error"]).abs()
importance_df = importance_df.sort_values("Importance", ascending=False)
importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

# ==========================================================
# 📏 3. 1-STEP EVALUATION ON 2022 (main table)
# ==========================================================
y_train_real, y_test_real = inv(y_train), inv(y_test)

train_pred = np.clip(inv(results.fittedvalues), 0, None)           # 1-step in-sample
test_pred = np.clip(inv(one_step_predictions(results, y_test, exog_test)), 0, None)  # 1-step out-of-sample

m_train = metrics(y_train_real.iloc[BURN_IN:], train_pred.iloc[BURN_IN:])
m_test = metrics(y_test_real, test_pred)

final_eval = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "MAPE", "R2"],
    "Train_1step": [m_train[k] for k in ["MAE", "RMSE", "MAPE", "R2"]],
    "Test_1step": [m_test[k] for k in ["MAE", "RMSE", "MAPE", "R2"]],
})
final_eval["Test_minus_Train"] = final_eval["Test_1step"] - final_eval["Train_1step"]
final_eval["Train_over_Test"] = final_eval["Train_1step"] / final_eval["Test_1step"]
final_eval.to_csv(os.path.join(output_dir, "final_evaluation_1step.csv"), index=False)
print("\n=== 1-step evaluation (train vs test, same horizon) ===")
print(final_eval)

pd.DataFrame([{
    "TARGET": TARGET, "order": str(best_order), "AIC": results.aic, "BIC": results.bic,
    "HQIC": results.hqic, "Log_Likelihood": results.llf, "nobs": results.nobs,
    "Converged": results.mle_retvals.get("converged", np.nan),
}]).to_csv(os.path.join(output_dir, "sarimax_key_results.csv"), index=False)

# 1-step figures
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label="Actual")
plt.plot(test_pred, label="1-step prediction", linestyle="--")
plt.title(f"{TARGET} - 1-step prediction vs actual (2022)")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "prediction_vs_actual_1step.png"), bbox_inches="tight"); plt.close()

# Save the 1-step predictions to CSV for further analysis
pd.DataFrame({
    "date": y_test_real.index,
    "actual": y_test_real.values,
    "predicted": test_pred.values,
}).to_csv(os.path.join(output_dir, "1step_predictions_2022.csv"), index=False)


residuals = y_test_real - test_pred
plt.figure(figsize=(8, 5)); sns.histplot(residuals, bins=50, kde=True)
plt.title(f"{TARGET} - Residual distribution (1-step, 2022)")
plt.savefig(os.path.join(output_dir, "residual_dist_1step.png"), bbox_inches="tight"); plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test_real, test_pred, alpha=0.5)
lo, hi = min(y_test_real.min(), test_pred.min()), max(y_test_real.max(), test_pred.max())
plt.plot([lo, hi], [lo, hi], linestyle="--")
plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title(f"{TARGET} - 1-step (2022)"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "scatter_1step.png"), bbox_inches="tight"); plt.close()




# ==========================================================
# 🔁 4. ROLLING MULTI-HORIZON (52 windows, step 7, h=1,3,7)
# ==========================================================
# Frozen parameters; after each window the STATE is updated with the actual week.
# Future exogenous variables are the observed weather (perfect weather forecast assumption).
HORIZONS, N_WINDOWS, STEP = [1, 3, 7], 52, 7
rows, current = [], results
origin = len(y_train)  # index of the first day of 2022 within y/exog

for w in range(N_WINDOWS):
    if origin + STEP > len(y):
        break
    X_fut = exog.iloc[origin:origin + STEP]
    y_fut = y.iloc[origin:origin + STEP]
    fc = current.get_forecast(steps=STEP, exog=X_fut).predicted_mean
    fc.index = y_fut.index
    pred_w, act_w = np.clip(inv(fc), 0, None), inv(y_fut)
    for h in HORIZONS:
        a, p = act_w.iloc[h - 1], pred_w.iloc[h - 1]
        rows.append({"window": w + 1, "origin_date": y.index[origin - 1], "target_date": act_w.index[h - 1],
                     "horizon": h, "actual": a, "predicted": p, "abs_error": abs(a - p),
                     "sq_error": (a - p) ** 2, "abs_pct_error": abs(a - p) / a * 100 if a != 0 else np.nan})
    current = current.append(y_fut, exog=X_fut, refit=False)
    origin += STEP

hr = pd.DataFrame(rows)
hr.to_csv(os.path.join(output_dir, "horizon_window_results.csv"), index=False)
summary = hr.groupby("horizon").agg(
    MAE=("abs_error", "mean"), RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("abs_pct_error", "mean"), n=("abs_error", "count")).reset_index()
summary.to_csv(os.path.join(output_dir, "horizon_summary.csv"), index=False)
print("\n=== Rolling multi-horizon ===")
print(summary)
print("(Check: MAE at h=1 should be similar to the 1-step MAE from section 3)")

plt.figure(figsize=(8, 5))
sns.boxplot(data=hr, x="horizon", y="abs_error")
plt.title(f"{TARGET} - Absolute error by horizon (52 windows)")
plt.xlabel("Horizon (days)"); plt.ylabel("Absolute error"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_abs_error_boxplot.png"), bbox_inches="tight"); plt.close()

weekly = hr.groupby("window").agg(MAE=("abs_error", "mean"),
                                  RMSE=("sq_error", lambda s: np.sqrt(s.mean()))).reset_index()
plt.figure(figsize=(12, 6))
plt.plot(weekly["window"], weekly["MAE"], marker="o", label="MAE")
plt.plot(weekly["window"], weekly["RMSE"], marker="o", label="RMSE")
plt.xlabel("Week"); plt.ylabel("Error"); plt.title(f"{TARGET} - Weekly MAE and RMSE")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "weekly_mae_rmse.png"), dpi=300, bbox_inches="tight"); plt.close()


# ==========================================================
# 🔁 4B. ROLLING MULTI-HORIZON, DAILY ORIGIN (robustness check)
# ==========================================================
# Same procedure as section 4, but the forecast origin advances ONE day at a time,
# so each horizon falls on every day of the week instead of always the same weekday.
# Parameters stay frozen; after each window the STATE is updated with one observed day.
HORIZONS_D, STEP_D, WINDOW_D = [1, 3, 7], 1, 7
rows_d, current_d = [], results
origin = len(y_train)            # first day of 2022 within y/exog
w = 0

while origin + WINDOW_D <= len(y):
    X_fut = exog.iloc[origin:origin + WINDOW_D]
    y_fut = y.iloc[origin:origin + WINDOW_D]
    fc = current_d.get_forecast(steps=WINDOW_D, exog=X_fut).predicted_mean
    fc.index = y_fut.index
    pred_w, act_w = np.clip(inv(fc), 0, None), inv(y_fut)
    origin_date = y.index[origin - 1]
    persist = inv(y.iloc[origin - 1])            # last observed value before the window
    for h in HORIZONS_D:
        a, p = act_w.iloc[h - 1], pred_w.iloc[h - 1]
        rows_d.append({
            "window": w + 1, "origin_date": origin_date,
            "target_date": act_w.index[h - 1], "target_weekday": act_w.index[h - 1].day_name(),
            "horizon": h, "actual": a, "predicted": p,
            "abs_error": abs(a - p), "sq_error": (a - p) ** 2,
            "abs_pct_error": abs(a - p) / a * 100 if a != 0 else np.nan,
            "persist_pred": persist,
            "persist_abs_error": abs(a - persist), "persist_sq_error": (a - persist) ** 2,
        })
    # update the state with ONE observed day (no re-estimation)
    current_d = current_d.append(y.iloc[origin:origin + STEP_D],
                                 exog=exog.iloc[origin:origin + STEP_D], refit=False)
    origin += STEP_D
    w += 1

hr_d = pd.DataFrame(rows_d)
hr_d.to_csv(os.path.join(output_dir, "horizon_window_results_daily.csv"), index=False)

summary_d = hr_d.groupby("horizon").agg(
    MAE=("abs_error", "mean"),
    RMSE=("sq_error", lambda s: np.sqrt(s.mean())),
    MAPE=("abs_pct_error", "mean"),
    persistence_MAE=("persist_abs_error", "mean"),
    persistence_RMSE=("persist_sq_error", lambda s: np.sqrt(s.mean())),
    n=("abs_error", "count"),
).reset_index()
summary_d.to_csv(os.path.join(output_dir, "horizon_summary_daily.csv"), index=False)
print(f"\n=== Rolling multi-horizon, DAILY origin ({w} windows) ===")
print(summary_d)

# Check 1: each horizon now covers all weekdays (counts should be ~equal)
print("\nTarget weekdays per horizon:")
print(pd.crosstab(hr_d["horizon"], hr_d["target_weekday"]))

# Check 2: h=1 must match the 1-step test predictions of section 3 (diff ~0)
h1 = hr_d[hr_d["horizon"] == 1].set_index("target_date")["predicted"]
print("Max diff h=1 vs 1-step:", (h1 - test_pred.loc[h1.index]).abs().max())

# MAE by horizon and target weekday (shows the weekday effect directly)
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
mae_wd = (hr_d.pivot_table(index="horizon", columns="target_weekday",
                           values="abs_error", aggfunc="mean")[weekday_order])
mae_wd.to_csv(os.path.join(output_dir, "horizon_mae_by_weekday_daily.csv"))
print("\nMAE by horizon and target weekday:")
print(mae_wd.round(3))

plt.figure(figsize=(8, 5))
sns.boxplot(data=hr_d, x="horizon", y="abs_error")
plt.title(f"{TARGET} - Absolute error by horizon (daily origin, {w} windows)")
plt.xlabel("Horizon (days)"); plt.ylabel("Absolute error"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "horizon_abs_error_boxplot_daily.png"), bbox_inches="tight"); plt.close()



plt.figure(figsize=(10, 6))
colors = ["tab:blue" if p < 0.05 else "lightgray" for p in importance_df["P_Value"]]
sns.barplot(data=importance_df, x="Importance", y="Feature", palette=colors)
plt.title(f"{TARGET} - Feature Importance")
plt.xlabel("Importance (|coef| / std_error)")
plt.ylabel("Variable")
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches="tight")
plt.close()