"""
main.py — End-to-End AQI Forecasting Pipeline
===============================================
Hybrid ARIMA-LSTM model for Indian Oil Corporation refinery zones.

Usage:
    python main.py                  # Full pipeline
    python main.py --skip-tuning    # Skip LSTM hyperparameter tuning
    python main.py --quick          # Quick mode (fewer epochs)
    python main.py --zone Mathura_Refinery   # Specific zone
"""

import argparse
import time
import sys
import os
import warnings

warnings.filterwarnings("ignore")

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import REFINERY_ZONES


def main():
    parser = argparse.ArgumentParser(description="AQI Forecasting — Hybrid ARIMA-LSTM Pipeline")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip LSTM hyperparameter tuning")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 epochs, skip tuning)")
    parser.add_argument("--zone", type=str, default=None, help="Specific refinery zone to analyze")
    args = parser.parse_args()

    if args.quick:
        args.skip_tuning = True
        import config
        config.LSTM_EPOCHS = 10
        config.LSTM_PATIENCE = 5

    start_time = time.time()

    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  HYBRID ARIMA-LSTM AQI FORECASTING MODEL                     ║")
    print("║  Indian Oil Corporation — Refinery Zone Air Quality           ║")
    print("╚" + "═" * 62 + "╝")
    print(f"\n  Mode: {'Quick' if args.quick else 'Full'}")
    print(f"  Tuning: {'Skipped' if args.skip_tuning else 'Enabled'}")
    print(f"  Zone: {args.zone or 'Auto (first zone)'}")

    # ══════════════════════════════════════════
    # Phase 1: Data Generation
    # ══════════════════════════════════════════
    import importlib
    mod_gen = importlib.import_module("src.01_data_generation")
    raw_df = mod_gen.generate_all_data()

    # ══════════════════════════════════════════
    # Phase 2: Preprocessing
    # ══════════════════════════════════════════
    mod_prep = importlib.import_module("src.02_data_preprocessing")
    processed_df = mod_prep.preprocess_data(raw_df)

    # ══════════════════════════════════════════
    # Phase 3: EDA
    # ══════════════════════════════════════════
    mod_eda = importlib.import_module("src.03_eda_visualization")
    mod_eda.run_eda(processed_df)

    # ══════════════════════════════════════════
    # Phase 4: Feature Selection
    # ══════════════════════════════════════════
    mod_feat = importlib.import_module("src.04_feature_selection")
    selected_features = mod_feat.run_feature_selection(processed_df)

    # ══════════════════════════════════════════
    # Phase 5: Seasonal Decomposition
    # ══════════════════════════════════════════
    mod_decomp = importlib.import_module("src.05_seasonal_decomposition")
    decomp_results = mod_decomp.run_seasonal_decomposition(processed_df, zone=args.zone)

    # ══════════════════════════════════════════
    # Phase 6: ARIMA Model
    # ══════════════════════════════════════════
    mod_arima = importlib.import_module("src.06_arima_model")
    arima_results = mod_arima.train_arima(processed_df, zone=args.zone)

    # ══════════════════════════════════════════
    # Phase 7a: LSTM Model
    # ══════════════════════════════════════════
    mod_lstm = importlib.import_module("src.07_lstm_model")
    lstm_results = mod_lstm.train_lstm(processed_df, selected_features, skip_tuning=args.skip_tuning)

    # ══════════════════════════════════════════
    # Phase 7b: Hybrid Combination
    # ══════════════════════════════════════════
    mod_hybrid = importlib.import_module("src.08_hybrid_forecast")
    hybrid_results = mod_hybrid.combine_hybrid_forecast(arima_results, lstm_results)

    # ══════════════════════════════════════════
    # Phase 8: Future Forecasting
    # ══════════════════════════════════════════
    forecast_df = mod_hybrid.forecast_future(arima_results["model"], n_days=30)

    # ══════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════
    elapsed = time.time() - start_time

    print("\n\n" + "╔" + "═" * 62 + "╗")
    print("║  PIPELINE COMPLETE — SUMMARY                                 ║")
    print("╠" + "═" * 62 + "╣")
    print(f"║  Zone Analyzed : {arima_results['zone']:<44}║")
    print(f"║  ARIMA Order   : {str(arima_results['order']):<44}║")
    print(f"║  ARIMA RMSE    : {arima_results['metrics']['RMSE']:<44}║")
    print(f"║  Hybrid RMSE   : {hybrid_results['metrics']['hybrid']['RMSE']:<44}║")
    print(f"║  Improvement   : {hybrid_results['improvement_pct']:.2f}%{' ' * 37}║")
    print(f"║  Runtime       : {elapsed:.1f}s{' ' * 40}║")
    print("╠" + "═" * 62 + "╣")
    print("║  Output Files:                                               ║")
    print("║    data/processed_data.csv         — Preprocessed dataset    ║")
    print("║    data/arima_residuals.csv         — ARIMA residuals        ║")
    print("║    results/model_comparison.csv     — Metrics comparison     ║")
    print("║    results/feature_selection.csv    — Feature rankings       ║")
    print("║    results/future_forecast.csv      — 30-day forecast        ║")
    print("║    plots/                           — All visualizations     ║")
    print("║    models/                          — Saved models           ║")
    print("╚" + "═" * 62 + "╝")

    print(f"\n  Next 5-day AQI forecast:")
    print(forecast_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
