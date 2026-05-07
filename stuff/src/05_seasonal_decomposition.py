"""
05_seasonal_decomposition.py — Seasonal Decomposition & Stationarity
======================================================================
Performs:
  - STL decomposition (Seasonal-Trend via LOESS)
  - Classical additive/multiplicative decomposition
  - Stationarity tests (ADF, KPSS)
  - ACF/PACF plots for ARIMA order guidance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, DECOMP_PLOTS_DIR, ARIMA_SEASONAL_PERIOD
from src.utils import setup_plot_style

warnings.filterwarnings("ignore")


def test_stationarity(series: pd.Series, name: str = "Series") -> dict:
    """
    Run ADF and KPSS stationarity tests.
    
    Returns dict with test statistics and conclusions.
    """
    results = {"name": name}
    
    # ADF Test (H0: unit root exists → non-stationary)
    adf_result = adfuller(series.dropna(), autolag="AIC")
    results["ADF_Statistic"] = round(adf_result[0], 4)
    results["ADF_p_value"] = round(adf_result[1], 6)
    results["ADF_Stationary"] = adf_result[1] < 0.05
    
    # KPSS Test (H0: series is stationary)
    try:
        kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
        results["KPSS_Statistic"] = round(kpss_result[0], 4)
        results["KPSS_p_value"] = round(kpss_result[1], 6)
        results["KPSS_Stationary"] = kpss_result[1] > 0.05
    except Exception:
        results["KPSS_Statistic"] = np.nan
        results["KPSS_p_value"] = np.nan
        results["KPSS_Stationary"] = None
    
    # Overall conclusion
    adf_stat = results["ADF_Stationary"]
    kpss_stat = results.get("KPSS_Stationary")
    
    if adf_stat and kpss_stat:
        results["Conclusion"] = "Stationary"
    elif not adf_stat and not kpss_stat:
        results["Conclusion"] = "Non-Stationary"
    elif adf_stat and not kpss_stat:
        results["Conclusion"] = "Trend-Stationary"
    else:
        results["Conclusion"] = "Difference-Stationary"
    
    return results


def perform_stl_decomposition(series: pd.Series, period: int = ARIMA_SEASONAL_PERIOD,
                               name: str = "AQI"):
    """
    Perform STL decomposition and create visualization.
    
    Returns trend, seasonal, and residual components.
    """
    setup_plot_style()
    
    # STL decomposition
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    
    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    components = [
        (series, "Original", "#FFFFFF"),
        (result.trend, "Trend", "#4ECDC4"),
        (result.seasonal, "Seasonal", "#FF6B6B"),
        (result.resid, "Residual", "#FFEAA7"),
    ]
    
    for ax, (data, title, color) in zip(axes, components):
        ax.plot(data.index, data.values, color=color, linewidth=0.8, alpha=0.85)
        ax.set_ylabel(title, fontsize=11, fontweight="bold")
        if title == "Residual":
            ax.axhline(0, color="#FF6B6B", linewidth=0.5, linestyle="--", alpha=0.5)
    
    axes[0].set_title(f"STL Decomposition — {name} (period={period})",
                      fontsize=14, fontweight="bold", pad=12)
    axes[-1].set_xlabel("Date")
    
    plt.tight_layout()
    path = os.path.join(DECOMP_PLOTS_DIR, f"stl_decomposition_{name.lower()}.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 STL decomposition: {path}")
    
    return result


def plot_acf_pacf(series: pd.Series, name: str = "AQI", lags: int = 50):
    """Plot ACF and PACF for ARIMA order selection."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    
    plot_acf(series.dropna(), lags=lags, ax=axes[0],
             color="#4ECDC4", vlines_kwargs={"colors": "#4ECDC4"})
    axes[0].set_title(f"Autocorrelation Function (ACF) — {name}",
                      fontsize=13, fontweight="bold")
    
    plot_pacf(series.dropna(), lags=lags, ax=axes[1],
              color="#FF6B6B", vlines_kwargs={"colors": "#FF6B6B"})
    axes[1].set_title(f"Partial Autocorrelation (PACF) — {name}",
                      fontsize=13, fontweight="bold")
    
    for ax in axes:
        ax.axhline(0, color="#C9D1D9", linewidth=0.5)
    
    plt.tight_layout()
    path = os.path.join(DECOMP_PLOTS_DIR, f"acf_pacf_{name.lower()}.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 ACF/PACF plots: {path}")


def run_seasonal_decomposition(df: pd.DataFrame = None, zone: str = None) -> dict:
    """
    Run complete seasonal decomposition analysis.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Processed data. If None, loads from CSV.
    zone : str, optional
        Specific zone to analyze. If None, uses first zone.
    
    Returns
    -------
    dict with decomposition results and stationarity info.
    """
    print("\n" + "=" * 60)
    print("  Phase 5: SEASONAL DECOMPOSITION")
    print("=" * 60)
    
    if df is None:
        processed_path = os.path.join(DATA_DIR, "processed_data.csv")
        df = pd.read_csv(processed_path)
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Select zone
    if zone is None:
        zone = df["Zone"].unique()[0]
    
    print(f"\n  🏭 Analyzing zone: {zone}")
    
    zone_df = df[df["Zone"] == zone].copy()
    zone_df = zone_df.set_index("Date").sort_index()
    aqi_series = zone_df["AQI"].dropna()
    
    print(f"  📊 AQI series length: {len(aqi_series)} observations")
    
    # ─── Stationarity Tests ───
    print("\n  🧪 Stationarity Tests:")
    
    # Original series
    stat_original = test_stationarity(aqi_series, "AQI (Original)")
    print(f"\n     Original AQI:")
    print(f"       ADF: statistic={stat_original['ADF_Statistic']}, "
          f"p={stat_original['ADF_p_value']} → {'Stationary' if stat_original['ADF_Stationary'] else 'Non-Stationary'}")
    print(f"       KPSS: statistic={stat_original['KPSS_Statistic']}, "
          f"p={stat_original['KPSS_p_value']} → {'Stationary' if stat_original['KPSS_Stationary'] else 'Non-Stationary'}")
    print(f"       Conclusion: {stat_original['Conclusion']}")
    
    # First difference
    aqi_diff = aqi_series.diff().dropna()
    stat_diff = test_stationarity(aqi_diff, "AQI (1st Difference)")
    print(f"\n     First Difference:")
    print(f"       ADF: statistic={stat_diff['ADF_Statistic']}, "
          f"p={stat_diff['ADF_p_value']} → {'Stationary' if stat_diff['ADF_Stationary'] else 'Non-Stationary'}")
    print(f"       Conclusion: {stat_diff['Conclusion']}")
    
    # ─── STL Decomposition ───
    print("\n  📈 STL Decomposition:")
    stl_result = perform_stl_decomposition(aqi_series, period=ARIMA_SEASONAL_PERIOD, name="AQI")
    
    # Stationarity of residual component
    stat_resid = test_stationarity(stl_result.resid.dropna(), "STL Residual")
    print(f"\n     STL Residual stationarity: {stat_resid['Conclusion']}")
    
    # ─── ACF / PACF ───
    print("\n  📊 ACF/PACF Analysis:")
    plot_acf_pacf(aqi_series, "AQI_Original")
    plot_acf_pacf(aqi_diff, "AQI_Differenced")
    
    # ─── Summary ───
    d_suggested = 0 if stat_original["ADF_Stationary"] else 1
    print(f"\n  💡 Suggested ARIMA differencing order (d): {d_suggested}")
    print(f"     (Based on ADF test — {'original is stationary' if d_suggested == 0 else 'first difference needed'})")
    
    results = {
        "zone": zone,
        "stationarity_original": stat_original,
        "stationarity_differenced": stat_diff,
        "stationarity_residual": stat_resid,
        "stl_result": stl_result,
        "suggested_d": d_suggested,
    }
    
    print(f"\n  ✅ Phase 5 complete: Decomposition and stationarity analysis done")
    
    return results


if __name__ == "__main__":
    results = run_seasonal_decomposition()
