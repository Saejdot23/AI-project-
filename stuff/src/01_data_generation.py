"""
01_data_generation.py — Synthetic Refinery Pollutant Data Generator
====================================================================
Generates realistic daily pollutant data for 3 Indian Oil Corporation
refinery zones (Mathura, Panipat, Gujarat) with:
  - Seasonal patterns (higher pollution in winter)
  - Weekly cycles (lower on weekends)
  - Long-term trend (gradual improvement)
  - Correlated pollutant spikes
  - ~15% random missingness (MCAR + MAR)
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    REFINERY_ZONES, DATE_RANGE_START, DATE_RANGE_END,
    MISSING_FRACTION, DATA_DIR, RANDOM_SEED, ALL_FEATURES
)


def generate_seasonal_component(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate seasonal multiplier: higher in winter (Oct-Feb), lower in monsoon (Jul-Sep).
    Returns array of multipliers centered around 1.0.
    """
    day_of_year = dates.dayofyear.values.astype(float)
    # Peak around day 15 (Jan 15) — winter peak
    seasonal = 1.0 + 0.35 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    return np.array(seasonal)


def generate_weekly_component(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate weekly cycle: lower on weekends (reduced industrial activity).
    Returns array of multipliers centered around 1.0.
    """
    day_of_week = dates.dayofweek  # 0=Mon, 6=Sun
    weekly = np.where(day_of_week >= 5, 0.85, 1.03)  # Weekends lower
    return weekly


def generate_trend_component(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate gradual downward trend (emission controls improving over years).
    Returns array of multipliers starting at ~1.05 and ending at ~0.92.
    """
    total_days = (dates[-1] - dates[0]).days
    days_elapsed = (dates - dates[0]).days.values.astype(float)
    trend = 1.05 - 0.13 * (days_elapsed / total_days)
    return np.array(trend)


def inject_correlated_spikes(df: pd.DataFrame, zone: str, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject correlated pollution spikes — when one pollutant spikes,
    related ones spike too (e.g., SO2 ↑ → PM2.5 ↑, PM10 ↑).
    """
    n = len(df)
    n_spikes = int(n * 0.03)  # ~3% of days have spikes
    spike_days = rng.choice(n, size=n_spikes, replace=False)
    
    spike_groups = [
        (["PM2.5", "PM10"], 1.8),          # Particulate spike
        (["SO2", "PM2.5", "NO2"], 1.6),     # Refinery emission spike
        (["CO", "NO2"], 1.5),               # Combustion spike
        (["O3"], 1.7),                       # Photochemical spike (summer)
    ]
    
    for day_idx in spike_days:
        group = spike_groups[rng.integers(0, len(spike_groups))]
        pollutants, multiplier = group
        for p in pollutants:
            if p in df.columns:
                spike_val = multiplier + rng.normal(0, 0.15)
                df.iloc[day_idx, df.columns.get_loc(p)] *= max(spike_val, 1.0)
    
    return df


def inject_missingness(df: pd.DataFrame, fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    """
    Inject ~fraction of missing values using a mix of:
    - MCAR (Missing Completely At Random): ~60% of missing
    - MAR (Missing At Random): ~40% — more missing on high-wind days
    """
    feature_cols = [c for c in df.columns if c not in ["Date", "Zone"]]
    n_total_cells = len(df) * len(feature_cols)
    
    # MCAR: Random missing values
    n_mcar = int(n_total_cells * fraction * 0.6)
    for _ in range(n_mcar):
        row = rng.integers(0, len(df))
        col = feature_cols[rng.integers(0, len(feature_cols))]
        df.at[df.index[row], col] = np.nan
    
    # MAR: More missing on high-wind days (instrument failure)
    if "Wind_Speed" in df.columns:
        high_wind_mask = df["Wind_Speed"] > df["Wind_Speed"].quantile(0.8)
        high_wind_indices = df.index[high_wind_mask]
        n_mar = int(n_total_cells * fraction * 0.4)
        pollutant_cols = [c for c in feature_cols if c not in ["Wind_Speed", "Temperature", "Humidity"]]
        for _ in range(min(n_mar, len(high_wind_indices) * len(pollutant_cols))):
            row = rng.choice(high_wind_indices)
            col = pollutant_cols[rng.integers(0, len(pollutant_cols))]
            df.at[row, col] = np.nan
    
    return df


def generate_zone_data(zone_name: str, zone_params: dict,
                       dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic pollutant data for a single refinery zone."""
    n = len(dates)
    
    # Temporal components
    seasonal = generate_seasonal_component(dates)
    weekly = generate_weekly_component(dates)
    trend = generate_trend_component(dates)
    
    data = {"Date": dates, "Zone": zone_name}
    
    for feature, (mean, std) in zone_params.items():
        # Base signal: random noise around mean
        base = rng.normal(mean, std, size=n)
        
        if feature in ["PM2.5", "PM10", "SO2", "NO2", "CO", "NH3", "Pb"]:
            # Apply seasonal, weekly, trend modulation
            signal = np.array(base * seasonal * weekly * trend, dtype=float)
        elif feature == "O3":
            # O3 is inversely seasonal (higher in summer)
            inverse_seasonal = 2.0 - seasonal
            signal = np.array(base * inverse_seasonal * weekly * trend, dtype=float)
        elif feature == "Temperature":
            # Temperature has its own seasonal pattern
            doy = dates.dayofyear.values.astype(float)
            temp_seasonal = mean + (std * 1.2) * np.sin(2 * np.pi * (doy - 100) / 365)
            signal = np.array(temp_seasonal + rng.normal(0, std * 0.3, size=n), dtype=float)
        elif feature == "Humidity":
            # Higher humidity in monsoon (Jul-Sep)
            monsoon = np.where((dates.month >= 7) & (dates.month <= 9), 1.3, 0.9)
            signal = np.array(base * monsoon, dtype=float)
        elif feature == "Wind_Speed":
            signal = np.abs(base).astype(float)  # Wind speed is always positive
        else:
            signal = np.array(base, dtype=float)
        
        # Ensure non-negative values for pollutants
        if feature not in ["Temperature"]:
            signal = np.maximum(signal, 0.01)
        
        # Add autocorrelation (today depends on yesterday)
        for i in range(1, n):
            signal[i] = 0.6 * signal[i] + 0.4 * signal[i - 1]
        
        data[feature] = np.round(signal, 3)
    
    df = pd.DataFrame(data)
    
    # Inject correlated spikes
    df = inject_correlated_spikes(df, zone_name, rng)
    
    return df


def generate_all_data() -> pd.DataFrame:
    """
    Generate synthetic pollutant data for all refinery zones.
    
    Returns
    -------
    pd.DataFrame
        Combined dataframe with columns: Date, Zone, PM2.5, PM10, ..., Wind_Speed
        Includes ~15% missing values.
    """
    print("\n" + "=" * 60)
    print("  Phase 1: SYNTHETIC DATA GENERATION")
    print("=" * 60)
    
    rng = np.random.default_rng(RANDOM_SEED)
    dates = pd.date_range(start=DATE_RANGE_START, end=DATE_RANGE_END, freq="D")
    
    all_zones = []
    for zone_name, zone_params in REFINERY_ZONES.items():
        print(f"\n  🏭 Generating data for {zone_name}...")
        df = generate_zone_data(zone_name, zone_params, dates, rng)
        all_zones.append(df)
        print(f"     ✓ {len(df)} daily records generated")
    
    # Combine all zones
    combined = pd.concat(all_zones, ignore_index=True)
    print(f"\n  📊 Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns")
    
    # Save BEFORE injecting missingness (for reference)
    complete_path = os.path.join(DATA_DIR, "complete_data_reference.csv")
    combined.to_csv(complete_path, index=False)
    print(f"  💾 Complete data saved: {complete_path}")
    
    # Inject missingness
    print(f"\n  🔧 Injecting ~{MISSING_FRACTION*100:.0f}% missing values...")
    combined = inject_missingness(combined, MISSING_FRACTION, rng)
    
    # Report missingness
    feature_cols = [c for c in combined.columns if c not in ["Date", "Zone"]]
    missing_pct = combined[feature_cols].isna().mean() * 100
    print(f"\n  Missing value percentages:")
    for col, pct in missing_pct.items():
        bar = "█" * int(pct / 2) + "░" * (25 - int(pct / 2))
        print(f"     {col:<14} {bar} {pct:.1f}%")
    
    # Save raw data with missingness
    raw_path = os.path.join(DATA_DIR, "raw_pollutant_data.csv")
    combined.to_csv(raw_path, index=False)
    print(f"\n  💾 Raw data (with missing values) saved: {raw_path}")
    print(f"\n  ✅ Phase 1 complete: {combined.shape[0]} records across {len(REFINERY_ZONES)} zones")
    
    return combined


if __name__ == "__main__":
    df = generate_all_data()
    print(f"\nSample data:\n{df.head(10)}")
