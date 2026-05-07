"""
02_data_preprocessing.py — Data Preprocessing & AQI Calculation
================================================================
Handles:
  - Missing value analysis & visualization
  - Multi-strategy imputation (interpolation, KNN, forward fill)
  - Outlier detection & winsorization
  - AQI computation using CPCB India formula
  - Feature engineering (rolling averages, lags, rate-of-change)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, EDA_PLOTS_DIR, AQI_POLLUTANTS, METEO_FEATURES,
    ALL_FEATURES, RANDOM_SEED
)
from src.utils import calculate_aqi, get_aqi_category, setup_plot_style


def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze and visualize missing value patterns."""
    feature_cols = [c for c in df.columns if c in ALL_FEATURES]
    
    missing_summary = pd.DataFrame({
        "Column": feature_cols,
        "Missing_Count": [df[c].isna().sum() for c in feature_cols],
        "Missing_Pct": [df[c].isna().mean() * 100 for c in feature_cols],
        "Total_Count": [len(df) for _ in feature_cols],
    })
    missing_summary = missing_summary.sort_values("Missing_Pct", ascending=False)
    
    print("\n  📋 Missing Value Summary:")
    print(missing_summary.to_string(index=False))
    
    # Missing value heatmap
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sample for visualization (full data too dense)
    sample_df = df[feature_cols].iloc[::3]  # Every 3rd row
    sns.heatmap(sample_df.isna().T, cbar=False, cmap=["#161B22", "#FF6B6B"],
                ax=ax, yticklabels=True)
    ax.set_title("Missing Data Pattern (white = present, red = missing)",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Observation Index (sampled)")
    
    path = os.path.join(EDA_PLOTS_DIR, "missing_data_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 Missing data heatmap saved: {path}")
    
    return missing_summary


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-strategy imputation:
    1. Time-series interpolation for small gaps (≤ 3 consecutive NaNs)
    2. KNN imputation using correlated features for larger gaps
    3. Forward/backward fill as final fallback
    """
    feature_cols = [c for c in df.columns if c in ALL_FEATURES]
    df = df.copy()
    
    print("\n  🔧 Imputation Strategy:")
    
    # Step 1: Time-series interpolation (handles small gaps)
    print("     Step 1/3: Time-series interpolation (limit=3)...")
    for col in feature_cols:
        df[col] = df[col].interpolate(method="linear", limit=3, limit_direction="both")
    
    remaining_before = df[feature_cols].isna().sum().sum()
    print(f"     → Remaining missing values: {remaining_before}")
    
    # Step 2: KNN imputation for larger gaps
    if remaining_before > 0:
        print("     Step 2/3: KNN imputation (k=5)...")
        knn_imputer = KNNImputer(n_neighbors=5, weights="distance")
        df[feature_cols] = knn_imputer.fit_transform(df[feature_cols])
        
        remaining_after = df[feature_cols].isna().sum().sum()
        print(f"     → Remaining missing values: {remaining_after}")
    
    # Step 3: Forward/backward fill (final fallback)
    remaining = df[feature_cols].isna().sum().sum()
    if remaining > 0:
        print("     Step 3/3: Forward/backward fill (fallback)...")
        df[feature_cols] = df[feature_cols].ffill().bfill()
    
    final_missing = df[feature_cols].isna().sum().sum()
    print(f"\n  ✅ Imputation complete. Remaining missing: {final_missing}")
    
    return df


def detect_and_handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect outliers using IQR method and apply winsorization.
    Pollutant values are clipped to [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    df = df.copy()
    pollutant_cols = [c for c in df.columns if c in AQI_POLLUTANTS]
    
    print("\n  🔍 Outlier Detection & Handling (IQR method):")
    total_outliers = 0
    
    for col in pollutant_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outlier_mask = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask.sum()
        total_outliers += n_outliers
        
        if n_outliers > 0:
            # Winsorize: clip to bounds
            df[col] = df[col].clip(lower=max(lower, 0), upper=upper)
            print(f"     {col:<14}: {n_outliers:>4} outliers clipped to [{max(lower,0):.1f}, {upper:.1f}]")
    
    print(f"\n  ✅ Total outliers handled: {total_outliers}")
    return df


def compute_aqi(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AQI and category for each row using CPCB India formula."""
    df = df.copy()
    
    print("\n  🧮 Computing AQI using CPCB India formula...")
    df["AQI"] = df.apply(lambda row: calculate_aqi(row, AQI_POLLUTANTS), axis=1)
    df["AQI_Category"] = df["AQI"].apply(get_aqi_category)
    
    valid_aqi = df["AQI"].notna().sum()
    print(f"     Valid AQI values: {valid_aqi}/{len(df)} ({valid_aqi/len(df)*100:.1f}%)")
    
    # AQI category distribution
    cat_dist = df["AQI_Category"].value_counts()
    print(f"\n  📊 AQI Category Distribution:")
    for cat, count in cat_dist.items():
        pct = count / len(df) * 100
        print(f"     {cat:<15}: {count:>5} ({pct:.1f}%)")
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features:
    - Rolling averages (7-day, 30-day) for AQI and key pollutants
    - Lag features (1-day, 7-day)
    - Rate of change (day-over-day)
    - Day-of-week, month, season indicators
    """
    df = df.copy()
    
    print("\n  ⚙️  Engineering features...")
    
    # Ensure Date is datetime
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Rolling averages (per zone)
    for col in ["AQI", "PM2.5", "PM10"]:
        if col in df.columns:
            df[f"{col}_MA7"] = df.groupby("Zone")[col].transform(
                lambda x: x.rolling(7, min_periods=1).mean()
            )
            df[f"{col}_MA30"] = df.groupby("Zone")[col].transform(
                lambda x: x.rolling(30, min_periods=1).mean()
            )
    
    # Lag features
    if "AQI" in df.columns:
        df["AQI_lag1"] = df.groupby("Zone")["AQI"].shift(1)
        df["AQI_lag7"] = df.groupby("Zone")["AQI"].shift(7)
        
        # Rate of change
        df["AQI_roc"] = df.groupby("Zone")["AQI"].pct_change()
    
    # Temporal features
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
        9: "Post-Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon",
    })
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    
    # Fill any NaN in new features
    df = df.bfill().ffill()
    
    n_new = len([c for c in df.columns if c not in ALL_FEATURES + ["Date", "Zone", "AQI", "AQI_Category"]])
    print(f"     ✓ {n_new} new features engineered")
    print(f"     ✓ Final shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def preprocess_data(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame, optional
        Raw data. If None, loads from CSV.
    
    Returns
    -------
    pd.DataFrame
        Fully preprocessed data with AQI and engineered features.
    """
    print("\n" + "=" * 60)
    print("  Phase 2: DATA PREPROCESSING")
    print("=" * 60)
    
    if df is None:
        raw_path = os.path.join(DATA_DIR, "raw_pollutant_data.csv")
        print(f"\n  📂 Loading raw data from: {raw_path}")
        df = pd.read_csv(raw_path)
    
    print(f"  📊 Input shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 1: Analyze missing values
    analyze_missing_values(df)
    
    # Step 2: Impute missing values
    df = impute_missing_values(df)
    
    # Step 3: Handle outliers
    df = detect_and_handle_outliers(df)
    
    # Step 4: Compute AQI
    df = compute_aqi(df)
    
    # Step 5: Feature engineering
    df = engineer_features(df)
    
    # Save processed data
    processed_path = os.path.join(DATA_DIR, "processed_data.csv")
    df.to_csv(processed_path, index=False)
    print(f"\n  💾 Processed data saved: {processed_path}")
    print(f"  ✅ Phase 2 complete: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


if __name__ == "__main__":
    df = preprocess_data()
    print(f"\nProcessed data sample:\n{df.head()}")
    print(f"\nColumns: {list(df.columns)}")
