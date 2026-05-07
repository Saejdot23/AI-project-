"""
06_arima_model.py — ARIMA/SARIMA Model for AQI Forecasting
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
import pmdarima as pm
from pmdarima import auto_arima
import joblib
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODELS_DIR, MODEL_PLOTS_DIR, RESULTS_DIR,
    TRAIN_TEST_SPLIT, ARIMA_MAX_P, ARIMA_MAX_D, ARIMA_MAX_Q,
    ARIMA_SEASONAL, ARIMA_SEASONAL_PERIOD, RANDOM_SEED
)
from src.utils import evaluate_model, plot_forecast, setup_plot_style

warnings.filterwarnings("ignore")


def fit_auto_arima(train_series, seasonal=ARIMA_SEASONAL, seasonal_period=ARIMA_SEASONAL_PERIOD):
    print("\n  Running Auto-ARIMA (stepwise search)...")
    model = auto_arima(
        train_series, start_p=0, max_p=ARIMA_MAX_P, start_q=0, max_q=ARIMA_MAX_Q,
        max_d=ARIMA_MAX_D, seasonal=seasonal, m=seasonal_period if seasonal else 1,
        start_P=0, max_P=2, start_Q=0, max_Q=2, max_D=1,
        trace=True, error_action="ignore", suppress_warnings=True,
        stepwise=True, information_criterion="aic", n_jobs=-1,
    )
    print(f"\n  Best ARIMA order: {model.order}")
    if seasonal:
        print(f"  Seasonal order: {model.seasonal_order}")
    print(f"  AIC: {model.aic():.2f}, BIC: {model.bic():.2f}")
    return model


def plot_arima_diagnostics(model, save_dir=MODEL_PLOTS_DIR):
    setup_plot_style()
    residuals = model.resid()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(residuals, color="#4ECDC4", linewidth=0.7, alpha=0.8)
    axes[0, 0].axhline(0, color="#FF6B6B", linewidth=0.8, linestyle="--")
    axes[0, 0].set_title("Residuals Over Time", fontsize=12, fontweight="bold")

    axes[0, 1].hist(residuals, bins=50, color="#FF6B6B", alpha=0.7, edgecolor="#30363D", density=True)
    from scipy import stats
    x = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), color="#FFEAA7", linewidth=2)
    axes[0, 1].set_title("Residual Distribution", fontsize=12, fontweight="bold")

    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].get_lines()[0].set_color("#4ECDC4")
    axes[1, 0].get_lines()[1].set_color("#FF6B6B")
    axes[1, 0].set_title("Q-Q Plot", fontsize=12, fontweight="bold")

    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=40, ax=axes[1, 1], color="#45B7D1", vlines_kwargs={"colors": "#45B7D1"})
    axes[1, 1].set_title("Residual ACF", fontsize=12, fontweight="bold")

    fig.suptitle("ARIMA Diagnostics", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "arima_diagnostics.png"))
    plt.close()


def train_arima(df=None, zone=None):
    print("\n" + "=" * 60)
    print("  Phase 6: ARIMA MODEL TRAINING")
    print("=" * 60)

    if df is None:
        df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"))
    df["Date"] = pd.to_datetime(df["Date"])

    if zone is None:
        zone = df["Zone"].unique()[0]
    print(f"\n  Training zone: {zone}")

    zone_df = df[df["Zone"] == zone].set_index("Date").sort_index()
    aqi_series = zone_df["AQI"].dropna()

    split_idx = int(len(aqi_series) * TRAIN_TEST_SPLIT)
    train = aqi_series.iloc[:split_idx]
    test = aqi_series.iloc[split_idx:]
    print(f"  Train: {len(train)}, Test: {len(test)}")

    model = fit_auto_arima(train)

    forecasts, conf_int = model.predict(n_periods=len(test), return_conf_int=True)
    forecast_series = pd.Series(forecasts, index=test.index)

    train_fitted = model.predict_in_sample()
    train_residuals = train.values - train_fitted
    test_residuals = test.values - forecasts

    metrics = evaluate_model(test.values, forecasts, "ARIMA")

    plot_forecast(test.values, forecasts, dates=test.index,
                  title=f"ARIMA Forecast — {zone.replace('_', ' ')}",
                  save_path=os.path.join(MODEL_PLOTS_DIR, "arima_forecast.png"), model_name="ARIMA")

    # Forecast with confidence interval plot
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))
    context = aqi_series.iloc[max(0, split_idx - 60):split_idx]
    ax.plot(context.index, context.values, color="#8B949E", linewidth=0.8, alpha=0.7, label="Training")
    ax.plot(test.index, test.values, color="#4ECDC4", linewidth=1.0, label="Actual")
    ax.plot(test.index, forecasts, color="#FF6B6B", linewidth=1.2, linestyle="--", label="ARIMA")
    ax.fill_between(test.index, conf_int[:, 0], conf_int[:, 1], alpha=0.15, color="#FF6B6B", label="95% CI")
    ax.axvline(train.index[-1], color="#FFEAA7", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title(f"ARIMA Forecast — {zone.replace('_', ' ')}", fontsize=14, fontweight="bold")
    ax.set_ylabel("AQI"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PLOTS_DIR, "arima_forecast_ci.png"))
    plt.close()

    plot_arima_diagnostics(model)

    joblib.dump(model, os.path.join(MODELS_DIR, "arima_model.pkl"))

    residuals_df = pd.DataFrame({
        "Date": list(train.index) + list(test.index),
        "Actual_AQI": list(train.values) + list(test.values),
        "ARIMA_Predicted": list(train_fitted) + list(forecasts),
        "Residual": list(train_residuals) + list(test_residuals),
        "Split": ["train"] * len(train) + ["test"] * len(test),
    })
    residuals_df.to_csv(os.path.join(DATA_DIR, "arima_residuals.csv"), index=False)

    print(f"\n  Phase 6 complete: ARIMA{model.order} trained")
    return {
        "model": model, "train": train, "test": test, "forecasts": forecast_series,
        "train_residuals": train_residuals, "test_residuals": test_residuals,
        "metrics": metrics, "zone": zone, "order": model.order,
    }

if __name__ == "__main__":
    train_arima()
