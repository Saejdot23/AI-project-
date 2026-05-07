"""
04_feature_selection.py — Feature Selection for AQI Prediction
===============================================================
Three-stage feature selection:
  1. Correlation analysis (Pearson & Spearman)
  2. Mutual Information scoring
  3. Recursive Feature Elimination (RFE) with Random Forest
  4. Consensus ranking — features selected by ≥ 2 methods
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_regression, RFE, SelectKBest, f_regression
)
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, RESULTS_DIR, EDA_PLOTS_DIR, AQI_POLLUTANTS, METEO_FEATURES,
    CORRELATION_THRESHOLD, RFE_N_FEATURES, FEATURE_CONSENSUS_MIN,
    RANDOM_SEED
)
from src.utils import setup_plot_style


def get_candidate_features(df: pd.DataFrame) -> list:
    """Get list of candidate feature columns for selection."""
    exclude = ["Date", "Zone", "AQI", "AQI_Category", "Season"]
    candidates = [c for c in df.columns if c not in exclude and df[c].dtype in ["float64", "int64", "int32", "float32"]]
    return candidates


def correlation_selection(df: pd.DataFrame, target: str = "AQI",
                          threshold: float = CORRELATION_THRESHOLD) -> dict:
    """
    Stage 1: Correlation-based feature selection.
    Computes Pearson and Spearman correlation with AQI.
    """
    candidates = get_candidate_features(df)
    
    pearson_corr = df[candidates].corrwith(df[target], method="pearson").abs()
    spearman_corr = df[candidates].corrwith(df[target], method="spearman").abs()
    
    # Average of both
    avg_corr = (pearson_corr + spearman_corr) / 2
    
    selected = avg_corr[avg_corr >= threshold].index.tolist()
    
    results = pd.DataFrame({
        "Feature": candidates,
        "Pearson_r": [pearson_corr.get(c, 0) for c in candidates],
        "Spearman_r": [spearman_corr.get(c, 0) for c in candidates],
        "Avg_Correlation": [avg_corr.get(c, 0) for c in candidates],
        "Selected_Corr": [c in selected for c in candidates],
    }).sort_values("Avg_Correlation", ascending=False)
    
    print(f"\n  📊 Correlation Analysis (threshold = {threshold}):")
    print(f"     Selected {len(selected)}/{len(candidates)} features")
    
    return {"results": results, "selected": selected}


def mutual_info_selection(df: pd.DataFrame, target: str = "AQI",
                          top_k: int = RFE_N_FEATURES) -> dict:
    """
    Stage 2: Mutual Information-based feature selection.
    Captures non-linear dependencies.
    """
    candidates = get_candidate_features(df)
    
    X = df[candidates].fillna(0).values
    y = df[target].fillna(df[target].median()).values
    
    mi_scores = mutual_info_regression(X, y, random_state=RANDOM_SEED, n_neighbors=5)
    
    mi_df = pd.DataFrame({
        "Feature": candidates,
        "MI_Score": mi_scores,
    }).sort_values("MI_Score", ascending=False)
    
    selected = mi_df.head(top_k)["Feature"].tolist()
    mi_df["Selected_MI"] = mi_df["Feature"].isin(selected)
    
    print(f"\n  📊 Mutual Information Analysis (top {top_k}):")
    print(f"     Selected: {selected}")
    
    return {"results": mi_df, "selected": selected}


def rfe_selection(df: pd.DataFrame, target: str = "AQI",
                  n_features: int = RFE_N_FEATURES) -> dict:
    """
    Stage 3: Recursive Feature Elimination with Random Forest.
    """
    candidates = get_candidate_features(df)
    
    X = df[candidates].fillna(0)
    y = df[target].fillna(df[target].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Random Forest estimator
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=10,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    
    rfe = RFE(estimator=rf, n_features_to_select=n_features, step=1)
    rfe.fit(X_scaled, y)
    
    rfe_df = pd.DataFrame({
        "Feature": candidates,
        "RFE_Rank": rfe.ranking_,
        "Selected_RFE": rfe.support_,
    }).sort_values("RFE_Rank")
    
    selected = rfe_df[rfe_df["Selected_RFE"]]["Feature"].tolist()
    
    # Also get feature importances from the fitted RF
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    rfe_df["RF_Importance"] = importances
    
    print(f"\n  📊 RFE Analysis (Random Forest, select {n_features}):")
    print(f"     Selected: {selected}")
    
    return {"results": rfe_df, "selected": selected}


def consensus_selection(corr_selected: list, mi_selected: list,
                        rfe_selected: list, min_votes: int = FEATURE_CONSENSUS_MIN) -> list:
    """
    Consensus ranking: Keep features selected by ≥ min_votes methods.
    """
    all_features = set(corr_selected + mi_selected + rfe_selected)
    
    consensus = {}
    for feat in all_features:
        votes = sum([
            feat in corr_selected,
            feat in mi_selected,
            feat in rfe_selected,
        ])
        consensus[feat] = votes
    
    selected = [f for f, v in consensus.items() if v >= min_votes]
    
    print(f"\n  📊 Consensus Selection (≥ {min_votes} votes):")
    print(f"     Selected {len(selected)} features:")
    for feat in sorted(selected):
        votes = consensus[feat]
        methods = []
        if feat in corr_selected:
            methods.append("Corr")
        if feat in mi_selected:
            methods.append("MI")
        if feat in rfe_selected:
            methods.append("RFE")
        print(f"       {feat:<20} — {votes}/3 votes ({', '.join(methods)})")
    
    return selected


def plot_feature_importance(corr_results: pd.DataFrame, mi_results: pd.DataFrame,
                            rfe_results: pd.DataFrame, final_selected: list):
    """Plot comprehensive feature importance visualization."""
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Correlation scores
    ax = axes[0, 0]
    data = corr_results.sort_values("Avg_Correlation", ascending=True).tail(15)
    colors = ["#4ECDC4" if f in final_selected else "#30363D" for f in data["Feature"]]
    ax.barh(data["Feature"], data["Avg_Correlation"], color=colors, edgecolor="#21262D")
    ax.set_title("Avg Correlation with AQI", fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation (|r|)")
    
    # Plot 2: Mutual Information scores
    ax = axes[0, 1]
    data = mi_results.sort_values("MI_Score", ascending=True).tail(15)
    colors = ["#FF6B6B" if f in final_selected else "#30363D" for f in data["Feature"]]
    ax.barh(data["Feature"], data["MI_Score"], color=colors, edgecolor="#21262D")
    ax.set_title("Mutual Information Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("MI Score")
    
    # Plot 3: RF Feature Importance
    ax = axes[1, 0]
    data = rfe_results.sort_values("RF_Importance", ascending=True).tail(15)
    colors = ["#FFEAA7" if f in final_selected else "#30363D" for f in data["Feature"]]
    ax.barh(data["Feature"], data["RF_Importance"], color=colors, edgecolor="#21262D")
    ax.set_title("Random Forest Importance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance")
    
    # Plot 4: Consensus summary
    ax = axes[1, 1]
    all_features = list(set(
        corr_results["Feature"].tolist() +
        mi_results["Feature"].tolist()
    ))
    
    consensus_data = []
    for f in all_features:
        votes = 0
        if f in corr_results[corr_results["Selected_Corr"]]["Feature"].values:
            votes += 1
        if f in mi_results[mi_results["Selected_MI"]]["Feature"].values:
            votes += 1
        if f in rfe_results[rfe_results["Selected_RFE"]]["Feature"].values:
            votes += 1
        consensus_data.append({"Feature": f, "Votes": votes})
    
    cons_df = pd.DataFrame(consensus_data).sort_values("Votes", ascending=True).tail(15)
    colors = ["#45B7D1" if f in final_selected else "#30363D" for f in cons_df["Feature"]]
    ax.barh(cons_df["Feature"], cons_df["Votes"], color=colors, edgecolor="#21262D")
    ax.set_title("Consensus Votes (3 methods)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Methods Selecting Feature")
    ax.set_xticks([0, 1, 2, 3])
    
    fig.suptitle("Feature Selection Analysis — Highlighted = Final Selected",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    path = os.path.join(EDA_PLOTS_DIR, "feature_importance.png")
    plt.savefig(path)
    plt.close()
    print(f"\n  📊 Feature importance plot: {path}")


def run_feature_selection(df: pd.DataFrame = None) -> list:
    """
    Run complete feature selection pipeline.
    
    Returns
    -------
    list
        Final selected feature names.
    """
    print("\n" + "=" * 60)
    print("  Phase 4: FEATURE SELECTION")
    print("=" * 60)
    
    if df is None:
        processed_path = os.path.join(DATA_DIR, "processed_data.csv")
        df = pd.read_csv(processed_path)
    
    # Filter to numeric columns with valid AQI
    df = df[df["AQI"].notna()].copy()
    
    print(f"\n  📊 Working with {len(get_candidate_features(df))} candidate features")
    
    # Stage 1: Correlation
    corr_result = correlation_selection(df)
    
    # Stage 2: Mutual Information
    mi_result = mutual_info_selection(df)
    
    # Stage 3: RFE
    rfe_result = rfe_selection(df)
    
    # Consensus
    final_selected = consensus_selection(
        corr_result["selected"],
        mi_result["selected"],
        rfe_result["selected"]
    )
    
    # Visualization
    plot_feature_importance(
        corr_result["results"],
        mi_result["results"],
        rfe_result["results"],
        final_selected
    )
    
    # Save results
    results_path = os.path.join(RESULTS_DIR, "feature_selection.csv")
    
    # Merge all results
    merged = corr_result["results"].merge(
        mi_result["results"], on="Feature", how="outer"
    ).merge(
        rfe_result["results"], on="Feature", how="outer"
    )
    merged["Final_Selected"] = merged["Feature"].isin(final_selected)
    merged.to_csv(results_path, index=False)
    print(f"\n  💾 Feature selection results: {results_path}")
    
    print(f"\n  ✅ Phase 4 complete: {len(final_selected)} features selected")
    print(f"     Final features: {final_selected}")
    
    return final_selected


if __name__ == "__main__":
    selected = run_feature_selection()
