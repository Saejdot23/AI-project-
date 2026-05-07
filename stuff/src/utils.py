"""
utils.py — Shared Utilities for AQI Forecasting Pipeline
=========================================================
Contains: AQI calculation (CPCB India), evaluation metrics,
plotting helpers, and LSTM sequence preparation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AQI_BREAKPOINTS, FIGURE_DPI, COLOR_PALETTE


# ──────────────────────────────────────────────────────────────
# AQI CALCULATION (CPCB India Formula)
# ──────────────────────────────────────────────────────────────

def calculate_sub_index(pollutant: str, concentration: float) -> float:
    """
    Calculate AQI sub-index for a single pollutant using CPCB breakpoints.
    
    Formula: Ip = ((IHi - ILo) / (BPHi - BPLo)) * (Cp - BPLo) + ILo
    
    Parameters
    ----------
    pollutant : str
        Name of the pollutant (must be in AQI_BREAKPOINTS).
    concentration : float
        Measured concentration value.
    
    Returns
    -------
    float
        Sub-index value, or NaN if concentration is missing/invalid.
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    
    if pollutant not in AQI_BREAKPOINTS:
        return np.nan
    
    breakpoints = AQI_BREAKPOINTS[pollutant]
    
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= concentration <= bp_hi:
            sub_index = ((i_hi - i_lo) / (bp_hi - bp_lo)) * (concentration - bp_lo) + i_lo
            return round(sub_index, 1)
    
    # If concentration exceeds the highest breakpoint, cap at 500
    if concentration > breakpoints[-1][1]:
        return 500.0
    
    return np.nan


def calculate_aqi(row: pd.Series, pollutant_cols: list = None) -> float:
    """
    Calculate overall AQI from a row of pollutant concentrations.
    
    Rules (CPCB India):
    - AQI = max of all sub-indices
    - Minimum 3 pollutants must have valid data
    - At least one of PM2.5 or PM10 must be present
    
    Parameters
    ----------
    row : pd.Series
        Row containing pollutant concentration values.
    pollutant_cols : list, optional
        List of pollutant column names. Defaults to all AQI_BREAKPOINTS keys.
    
    Returns
    -------
    float
        Calculated AQI value, or NaN if insufficient data.
    """
    if pollutant_cols is None:
        pollutant_cols = list(AQI_BREAKPOINTS.keys())
    
    sub_indices = {}
    for pollutant in pollutant_cols:
        if pollutant in row.index and not pd.isna(row[pollutant]):
            si = calculate_sub_index(pollutant, row[pollutant])
            if not pd.isna(si):
                sub_indices[pollutant] = si
    
    # Check minimum data requirements
    if len(sub_indices) < 3:
        return np.nan
    
    # At least one of PM2.5 or PM10 must be present
    if "PM2.5" not in sub_indices and "PM10" not in sub_indices:
        return np.nan
    
    return max(sub_indices.values())


def get_aqi_category(aqi_value: float) -> str:
    """Return AQI category string based on value."""
    if pd.isna(aqi_value):
        return "Unknown"
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Satisfactory"
    elif aqi_value <= 200:
        return "Moderate"
    elif aqi_value <= 300:
        return "Poor"
    elif aqi_value <= 400:
        return "Very Poor"
    else:
        return "Severe"


AQI_CATEGORY_COLORS = {
    "Good": "#009966",
    "Satisfactory": "#58BC50",
    "Moderate": "#FFDD44",
    "Poor": "#FF8C00",
    "Very Poor": "#CC0033",
    "Severe": "#800020",
    "Unknown": "#999999",
}


# ──────────────────────────────────────────────────────────────
# EVALUATION METRICS
# ──────────────────────────────────────────────────────────────

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute regression evaluation metrics.
    
    Returns dict with MAE, RMSE, R², MAPE and prints a summary.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # Remove NaN pairs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan
    
    metrics = {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R²": round(r2, 4),
        "MAPE (%)": round(mape, 2),
    }
    
    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Metrics")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k:<12}: {v}")
    print(f"{'='*50}\n")
    
    return metrics


# ──────────────────────────────────────────────────────────────
# LSTM SEQUENCE PREPARATION
# ──────────────────────────────────────────────────────────────

def create_sequences(data: np.ndarray, target: np.ndarray, window_size: int):
    """
    Create sliding window sequences for LSTM input.
    
    Parameters
    ----------
    data : np.ndarray
        Feature array of shape (n_samples, n_features).
    target : np.ndarray
        Target array of shape (n_samples,).
    window_size : int
        Number of past time steps to use as input.
    
    Returns
    -------
    X : np.ndarray of shape (n_sequences, window_size, n_features)
    y : np.ndarray of shape (n_sequences,)
    """
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ──────────────────────────────────────────────────────────────

def setup_plot_style():
    """Set up consistent matplotlib styling for all plots."""
    plt.rcParams.update({
        "figure.dpi": FIGURE_DPI,
        "figure.facecolor": "#0D1117",
        "axes.facecolor": "#161B22",
        "axes.edgecolor": "#30363D",
        "axes.labelcolor": "#C9D1D9",
        "axes.grid": True,
        "grid.color": "#21262D",
        "grid.alpha": 0.6,
        "text.color": "#C9D1D9",
        "xtick.color": "#8B949E",
        "ytick.color": "#8B949E",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": "#161B22",
        "legend.edgecolor": "#30363D",
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#0D1117",
        "savefig.pad_inches": 0.3,
    })


def plot_forecast(y_true, y_pred, dates=None, title="Forecast vs Actual",
                  save_path=None, model_name="Model"):
    """
    Plot actual vs predicted values with residual distribution.
    
    Creates a 2-panel figure: time series comparison + residual histogram.
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                              gridspec_kw={"hspace": 0.3})
    
    # Panel 1: Actual vs Predicted
    ax1 = axes[0]
    x_axis = dates if dates is not None else range(len(y_true))
    ax1.plot(x_axis, y_true, color="#4ECDC4", linewidth=1.2, alpha=0.9, label="Actual AQI")
    ax1.plot(x_axis, y_pred, color="#FF6B6B", linewidth=1.2, alpha=0.8,
             linestyle="--", label=f"{model_name} Predicted")
    ax1.fill_between(x_axis, y_true, y_pred, alpha=0.15, color="#FFEAA7")
    ax1.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax1.set_ylabel("AQI Value")
    ax1.legend(loc="upper right")
    
    # Panel 2: Residuals
    ax2 = axes[1]
    residuals = np.array(y_true) - np.array(y_pred)
    ax2.bar(x_axis, residuals, color="#45B7D1", alpha=0.6, width=1)
    ax2.axhline(0, color="#FF6B6B", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Residual")
    ax2.set_xlabel("Date" if dates is not None else "Time Step")
    ax2.set_title("Prediction Residuals", fontsize=12, pad=8)
    
    if save_path:
        plt.savefig(save_path)
        print(f"  📊 Plot saved: {save_path}")
    plt.close()
    
    return fig


def plot_comparison(y_true, predictions_dict, dates=None,
                    title="Model Comparison", save_path=None):
    """
    Plot actual values against multiple model predictions.
    
    Parameters
    ----------
    predictions_dict : dict
        {model_name: y_pred_array}
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x_axis = dates if dates is not None else range(len(y_true))
    ax.plot(x_axis, y_true, color="#FFFFFF", linewidth=1.8, alpha=0.9,
            label="Actual AQI", zorder=5)
    
    colors = ["#FF6B6B", "#4ECDC4", "#FFEAA7", "#DDA0DD", "#45B7D1"]
    for idx, (name, y_pred) in enumerate(predictions_dict.items()):
        ax.plot(x_axis, y_pred, color=colors[idx % len(colors)],
                linewidth=1.2, alpha=0.8, linestyle="--", label=name)
    
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel("AQI Value")
    ax.set_xlabel("Date" if dates is not None else "Time Step")
    ax.legend(loc="upper right")
    
    if save_path:
        plt.savefig(save_path)
        print(f"  📊 Plot saved: {save_path}")
    plt.close()
    
    return fig
