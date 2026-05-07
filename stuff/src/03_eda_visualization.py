"""
03_eda_visualization.py — Exploratory Data Analysis & Visualization
====================================================================
Generates comprehensive EDA plots:
  - Time series for AQI and pollutants by zone
  - Correlation heatmap
  - Distribution plots (histograms + KDE)
  - Seasonal box plots (monthly)
  - AQI category distribution
  - Zone-wise comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, EDA_PLOTS_DIR, AQI_POLLUTANTS, METEO_FEATURES,
    COLOR_PALETTE, FIGURE_DPI
)
from src.utils import setup_plot_style, AQI_CATEGORY_COLORS


def plot_aqi_time_series(df: pd.DataFrame):
    """Plot AQI time series for all refinery zones."""
    setup_plot_style()
    zones = df["Zone"].unique()
    fig, axes = plt.subplots(len(zones), 1, figsize=(16, 4 * len(zones)),
                              sharex=True)
    if len(zones) == 1:
        axes = [axes]
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    
    for idx, (zone, ax) in enumerate(zip(zones, axes)):
        zone_df = df[df["Zone"] == zone].copy()
        zone_df["Date"] = pd.to_datetime(zone_df["Date"])
        
        ax.plot(zone_df["Date"], zone_df["AQI"], color=colors[idx],
                linewidth=0.8, alpha=0.7)
        
        # Add 30-day moving average
        if "AQI_MA30" in zone_df.columns:
            ax.plot(zone_df["Date"], zone_df["AQI_MA30"],
                    color="#FFFFFF", linewidth=1.5, alpha=0.9,
                    label="30-day MA")
        
        # Color bands for AQI categories
        ax.axhspan(0, 50, alpha=0.05, color="#009966")
        ax.axhspan(50, 100, alpha=0.05, color="#58BC50")
        ax.axhspan(100, 200, alpha=0.05, color="#FFDD44")
        ax.axhspan(200, 300, alpha=0.05, color="#FF8C00")
        ax.axhspan(300, 500, alpha=0.05, color="#CC0033")
        
        ax.set_ylabel("AQI")
        ax.set_title(f"🏭 {zone.replace('_', ' ')}", fontsize=13, fontweight="bold")
        ax.legend(loc="upper right")
        ax.set_ylim(0, df["AQI"].quantile(0.99) * 1.1)
    
    axes[-1].set_xlabel("Date")
    fig.suptitle("Air Quality Index — Indian Oil Refinery Zones",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = os.path.join(EDA_PLOTS_DIR, "aqi_time_series.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 AQI time series: {path}")


def plot_pollutant_distributions(df: pd.DataFrame):
    """Plot distribution of each pollutant across zones."""
    setup_plot_style()
    pollutants = [p for p in AQI_POLLUTANTS if p in df.columns]
    n_cols = 4
    n_rows = (len(pollutants) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    zone_colors = {"Mathura_Refinery": "#FF6B6B", "Panipat_Refinery": "#4ECDC4",
                   "Gujarat_Refinery": "#45B7D1"}
    
    for idx, pollutant in enumerate(pollutants):
        ax = axes[idx]
        for zone, color in zone_colors.items():
            zone_data = df[df["Zone"] == zone][pollutant].dropna()
            ax.hist(zone_data, bins=40, alpha=0.5, color=color, density=True,
                    label=zone.split("_")[0])
            zone_data.plot.kde(ax=ax, color=color, linewidth=1.5)
        
        ax.set_title(pollutant, fontsize=12, fontweight="bold")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("Density")
        if idx == 0:
            ax.legend(fontsize=8)
    
    # Hide unused axes
    for idx in range(len(pollutants), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Pollutant Distributions by Refinery Zone",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = os.path.join(EDA_PLOTS_DIR, "pollutant_distributions.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 Pollutant distributions: {path}")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Plot correlation matrix heatmap of all features."""
    setup_plot_style()
    feature_cols = [c for c in AQI_POLLUTANTS + METEO_FEATURES + ["AQI"] if c in df.columns]
    
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 15, s=85, l=40, n=9, center="dark", as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
                linecolor="#30363D", ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"})
    
    ax.set_title("Feature Correlation Matrix",
                 fontsize=15, fontweight="bold", pad=15)
    
    plt.tight_layout()
    path = os.path.join(EDA_PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 Correlation heatmap: {path}")


def plot_monthly_boxplots(df: pd.DataFrame):
    """Plot monthly AQI variation as box plots."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_plot = df.copy()
    df_plot["Month"] = pd.to_datetime(df_plot["Date"]).dt.month
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    df_plot["Month_Name"] = df_plot["Month"].map(month_names)
    
    # Order months
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    box_palette = {m: COLOR_PALETTE[i % len(COLOR_PALETTE)] for i, m in enumerate(month_order)}
    
    sns.boxplot(data=df_plot, x="Month_Name", y="AQI", order=month_order,
                palette=box_palette, ax=ax, flierprops={"marker": ".", "alpha": 0.3},
                linewidth=0.8)
    
    ax.set_title("Monthly AQI Variation Across All Refinery Zones",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("AQI Value")
    
    # Add category reference lines
    for val, label, color in [(50, "Good", "#009966"), (100, "Satisfactory", "#58BC50"),
                               (200, "Moderate", "#FFDD44"), (300, "Poor", "#FF8C00")]:
        ax.axhline(val, color=color, linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(11.5, val + 5, label, fontsize=7, color=color, ha="right")
    
    plt.tight_layout()
    path = os.path.join(EDA_PLOTS_DIR, "monthly_aqi_boxplots.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 Monthly boxplots: {path}")


def plot_aqi_category_pie(df: pd.DataFrame):
    """Plot AQI category distribution as a donut chart."""
    setup_plot_style()
    fig, axes = plt.subplots(1, len(df["Zone"].unique()) + 1,
                              figsize=(5 * (len(df["Zone"].unique()) + 1), 5))
    
    # Overall
    cat_counts = df["AQI_Category"].value_counts()
    colors = [AQI_CATEGORY_COLORS.get(c, "#999") for c in cat_counts.index]
    
    axes[0].pie(cat_counts.values, labels=cat_counts.index, colors=colors,
                autopct="%1.1f%%", startangle=90, pctdistance=0.8,
                textprops={"fontsize": 8, "color": "#C9D1D9"})
    centre_circle = plt.Circle((0, 0), 0.55, fc="#0D1117")
    axes[0].add_artist(centre_circle)
    axes[0].set_title("All Zones", fontsize=12, fontweight="bold")
    
    # Per zone
    for idx, zone in enumerate(sorted(df["Zone"].unique()), 1):
        zone_df = df[df["Zone"] == zone]
        cat_counts = zone_df["AQI_Category"].value_counts()
        colors = [AQI_CATEGORY_COLORS.get(c, "#999") for c in cat_counts.index]
        
        axes[idx].pie(cat_counts.values, labels=cat_counts.index, colors=colors,
                      autopct="%1.1f%%", startangle=90, pctdistance=0.8,
                      textprops={"fontsize": 8, "color": "#C9D1D9"})
        centre_circle = plt.Circle((0, 0), 0.55, fc="#0D1117")
        axes[idx].add_artist(centre_circle)
        axes[idx].set_title(zone.replace("_", " "), fontsize=11, fontweight="bold")
    
    fig.suptitle("AQI Category Distribution",
                 fontsize=15, fontweight="bold", y=1.05)
    plt.tight_layout()
    
    path = os.path.join(EDA_PLOTS_DIR, "aqi_category_distribution.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 AQI category chart: {path}")


def plot_zone_comparison(df: pd.DataFrame):
    """Compare AQI statistics across refinery zones."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    zone_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    zones = sorted(df["Zone"].unique())
    
    # Bar chart: Mean AQI by zone
    mean_aqi = df.groupby("Zone")["AQI"].mean()
    bars = axes[0].bar(range(len(zones)), [mean_aqi[z] for z in zones],
                       color=zone_colors, alpha=0.85, edgecolor="#30363D")
    axes[0].set_xticks(range(len(zones)))
    axes[0].set_xticklabels([z.replace("_", "\n") for z in zones], fontsize=9)
    axes[0].set_ylabel("Mean AQI")
    axes[0].set_title("Average AQI by Refinery Zone", fontsize=13, fontweight="bold")
    
    for bar, zone in zip(bars, zones):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                     f"{mean_aqi[zone]:.0f}", ha="center", fontsize=10,
                     fontweight="bold", color="#C9D1D9")
    
    # Violin plot: AQI distribution by zone
    parts = axes[1].violinplot([df[df["Zone"] == z]["AQI"].dropna().values for z in zones],
                                showmeans=True, showmedians=True)
    
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(zone_colors[idx])
        pc.set_alpha(0.7)
    
    axes[1].set_xticks(range(1, len(zones) + 1))
    axes[1].set_xticklabels([z.replace("_", "\n") for z in zones], fontsize=9)
    axes[1].set_ylabel("AQI")
    axes[1].set_title("AQI Distribution by Refinery Zone", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    path = os.path.join(EDA_PLOTS_DIR, "zone_comparison.png")
    plt.savefig(path)
    plt.close()
    print(f"  📊 Zone comparison: {path}")


def run_eda(df: pd.DataFrame = None):
    """Run complete EDA pipeline."""
    print("\n" + "=" * 60)
    print("  Phase 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    if df is None:
        processed_path = os.path.join(DATA_DIR, "processed_data.csv")
        print(f"\n  📂 Loading processed data from: {processed_path}")
        df = pd.read_csv(processed_path)
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"\n  📊 Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  📊 Zones: {df['Zone'].unique().tolist()}")
    print(f"  📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Summary statistics
    print(f"\n  📋 AQI Statistics:")
    aqi_stats = df.groupby("Zone")["AQI"].describe().round(1)
    print(aqi_stats.to_string())
    
    # Generate plots
    print(f"\n  🎨 Generating visualizations...")
    plot_aqi_time_series(df)
    plot_pollutant_distributions(df)
    plot_correlation_heatmap(df)
    plot_monthly_boxplots(df)
    plot_aqi_category_pie(df)
    plot_zone_comparison(df)
    
    print(f"\n  ✅ Phase 3 complete: 6 visualization sets generated in {EDA_PLOTS_DIR}")
    return df


if __name__ == "__main__":
    run_eda()
