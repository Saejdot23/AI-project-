"""
08_hybrid_forecast.py — Hybrid ARIMA-LSTM Forecast & Evaluation
================================================================
Combines ARIMA + LSTM predictions and compares against standalone models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODELS_DIR, MODEL_PLOTS_DIR, RESULTS_DIR, TRAIN_TEST_SPLIT
)
from src.utils import evaluate_model, plot_comparison, setup_plot_style


def combine_hybrid_forecast(arima_results, lstm_results):
    """Combine ARIMA predictions + LSTM residual predictions."""
    print("\n" + "=" * 60)
    print("  Phase 7b: HYBRID ARIMA-LSTM COMBINATION")
    print("=" * 60)

    # ARIMA test predictions
    arima_test = arima_results["test"]
    arima_forecasts = arima_results["forecasts"]

    # LSTM residual predictions (shorter due to windowing)
    lstm_pred_residuals = lstm_results["y_pred_residuals"]
    lstm_dates = lstm_results["test_dates"]

    # Align ARIMA predictions with LSTM dates
    arima_aligned = arima_forecasts.loc[lstm_dates].values
    actual_aligned = arima_test.loc[lstm_dates].values

    # Hybrid = ARIMA + LSTM residual
    hybrid_predictions = arima_aligned + lstm_pred_residuals

    print(f"\n  Aligned test samples: {len(actual_aligned)}")

    # Evaluate all models
    print("\n  ── Model Comparison ──")
    arima_metrics = evaluate_model(actual_aligned, arima_aligned, "ARIMA (standalone)")
    lstm_direct_preds = arima_aligned + lstm_results["y_actual_residuals"]
    hybrid_metrics = evaluate_model(actual_aligned, hybrid_predictions, "Hybrid ARIMA-LSTM")

    # Standalone LSTM (trained directly on AQI, approximate via residuals)
    lstm_resid_metrics = evaluate_model(
        lstm_results["y_actual_residuals"], lstm_pred_residuals, "LSTM (residuals only)"
    )

    # Comparison table
    metrics_df = pd.DataFrame([arima_metrics, lstm_resid_metrics, hybrid_metrics])
    print("\n  ╔════════════════════════════════════════════════════╗")
    print("  ║         MODEL COMPARISON RESULTS                  ║")
    print("  ╠════════════════════════════════════════════════════╣")
    print(metrics_df.to_string(index=False))
    print("  ╚════════════════════════════════════════════════════╝")

    metrics_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)

    # Comparison plot
    plot_comparison(
        actual_aligned,
        {"ARIMA": arima_aligned, "Hybrid ARIMA-LSTM": hybrid_predictions},
        dates=lstm_dates,
        title="Model Comparison — Actual vs Predicted AQI",
        save_path=os.path.join(MODEL_PLOTS_DIR, "model_comparison.png"),
    )

    # Detailed hybrid forecast plot
    setup_plot_style()
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    # Panel 1: ARIMA component
    axes[0].plot(lstm_dates, actual_aligned, color="#FFFFFF", linewidth=1, alpha=0.8, label="Actual")
    axes[0].plot(lstm_dates, arima_aligned, color="#FF6B6B", linewidth=1, linestyle="--", label="ARIMA", alpha=0.8)
    axes[0].set_title("ARIMA Component", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("AQI"); axes[0].legend()

    # Panel 2: LSTM residual component
    axes[1].plot(lstm_dates, lstm_results["y_actual_residuals"], color="#FFFFFF",
                 linewidth=1, alpha=0.8, label="Actual Residual")
    axes[1].plot(lstm_dates, lstm_pred_residuals, color="#4ECDC4",
                 linewidth=1, linestyle="--", label="LSTM Predicted", alpha=0.8)
    axes[1].axhline(0, color="#8B949E", linewidth=0.5, linestyle=":")
    axes[1].set_title("LSTM Residual Component", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("Residual"); axes[1].legend()

    # Panel 3: Hybrid combined
    axes[2].plot(lstm_dates, actual_aligned, color="#FFFFFF", linewidth=1.2, alpha=0.9, label="Actual")
    axes[2].plot(lstm_dates, hybrid_predictions, color="#FFEAA7", linewidth=1.2,
                 linestyle="--", label="Hybrid ARIMA-LSTM", alpha=0.9)
    axes[2].fill_between(lstm_dates, actual_aligned, hybrid_predictions, alpha=0.1, color="#FFEAA7")
    axes[2].set_title("Hybrid ARIMA-LSTM Combined Forecast", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("AQI"); axes[2].set_xlabel("Date"); axes[2].legend()

    fig.suptitle("Hybrid ARIMA-LSTM Decomposition", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PLOTS_DIR, "hybrid_decomposition.png"))
    plt.close()

    # Improvement summary
    arima_rmse = arima_metrics["RMSE"]
    hybrid_rmse = hybrid_metrics["RMSE"]
    improvement = ((arima_rmse - hybrid_rmse) / arima_rmse) * 100

    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  HYBRID MODEL IMPROVEMENT                ║")
    print(f"  ╠══════════════════════════════════════════╣")
    print(f"  ║  ARIMA RMSE:  {arima_rmse:>8.4f}                  ║")
    print(f"  ║  Hybrid RMSE: {hybrid_rmse:>8.4f}                  ║")
    print(f"  ║  Improvement: {improvement:>7.2f}%                  ║")
    print(f"  ╚══════════════════════════════════════════╝")

    return {
        "hybrid_predictions": hybrid_predictions,
        "actual": actual_aligned,
        "arima_preds": arima_aligned,
        "dates": lstm_dates,
        "metrics": {"arima": arima_metrics, "hybrid": hybrid_metrics, "lstm_resid": lstm_resid_metrics},
        "improvement_pct": improvement,
    }


def forecast_future(arima_model, n_days=30):
    """Forecast future AQI values using the ARIMA model."""
    print(f"\n  Forecasting {n_days} days ahead...")
    forecasts, conf_int = arima_model.predict(n_periods=n_days, return_conf_int=True)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    days = range(1, n_days + 1)
    ax.plot(days, forecasts, color="#FF6B6B", linewidth=2, label="ARIMA Forecast", marker="o", markersize=3)
    ax.fill_between(days, conf_int[:, 0], conf_int[:, 1], alpha=0.2, color="#FF6B6B", label="95% CI")
    ax.set_title(f"Future {n_days}-Day AQI Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Days Ahead"); ax.set_ylabel("Predicted AQI"); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PLOTS_DIR, "future_forecast.png"))
    plt.close()

    forecast_df = pd.DataFrame({
        "Day_Ahead": range(1, n_days + 1),
        "Predicted_AQI": np.round(forecasts, 1),
        "CI_Lower": np.round(conf_int[:, 0], 1),
        "CI_Upper": np.round(conf_int[:, 1], 1),
    })
    forecast_df.to_csv(os.path.join(RESULTS_DIR, "future_forecast.csv"), index=False)
    print(f"  Future forecast saved to results/future_forecast.csv")
    return forecast_df


if __name__ == "__main__":
    print("Run via main.py for full pipeline.")
