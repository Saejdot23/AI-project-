"""
07_lstm_model.py — LSTM Model for ARIMA Residual Prediction
=============================================================
Trains LSTM on ARIMA residuals + multivariate features.
Supports hyperparameter tuning via Keras Tuner.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODELS_DIR, MODEL_PLOTS_DIR, RESULTS_DIR,
    LSTM_WINDOW_SIZE, LSTM_EPOCHS, LSTM_PATIENCE, RANDOM_SEED,
    LSTM_DEFAULT_UNITS_1, LSTM_DEFAULT_UNITS_2, LSTM_DEFAULT_DROPOUT,
    LSTM_DEFAULT_LR, LSTM_DEFAULT_BATCH_SIZE, TRAIN_TEST_SPLIT,
    AQI_POLLUTANTS, METEO_FEATURES
)
from src.utils import evaluate_model, create_sequences, setup_plot_style

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def build_lstm_model(input_shape, units_1=LSTM_DEFAULT_UNITS_1, units_2=LSTM_DEFAULT_UNITS_2,
                     dropout=LSTM_DEFAULT_DROPOUT, lr=LSTM_DEFAULT_LR):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units_1, return_sequences=True),
        Dropout(dropout),
        LSTM(units_2, return_sequences=False),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model


def prepare_lstm_data(df, selected_features, window_size=LSTM_WINDOW_SIZE):
    """Prepare data for LSTM: scale features, create sequences."""
    residuals_path = os.path.join(DATA_DIR, "arima_residuals.csv")
    residuals_df = pd.read_csv(residuals_path)
    residuals_df["Date"] = pd.to_datetime(residuals_df["Date"])

    df["Date"] = pd.to_datetime(df["Date"])
    zone = df["Zone"].unique()[0]
    zone_df = df[df["Zone"] == zone].copy().set_index("Date").sort_index()

    # Align features with residuals
    available_feats = [f for f in selected_features if f in zone_df.columns and f != "AQI"]
    if not available_feats:
        available_feats = [f for f in AQI_POLLUTANTS + METEO_FEATURES if f in zone_df.columns][:5]

    merged = residuals_df.set_index("Date").join(zone_df[available_feats], how="inner")
    merged = merged.dropna()

    # Separate train/test
    train_mask = merged["Split"] == "train"
    test_mask = merged["Split"] == "test"
    train_data = merged[train_mask]
    test_data = merged[test_mask]

    feature_cols = available_feats + ["Residual"]

    # Scale
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data[feature_cols])
    test_scaled = scaler.transform(test_data[feature_cols])

    # Target scaler for residuals
    target_scaler = MinMaxScaler()
    train_target = target_scaler.fit_transform(train_data[["Residual"]])
    test_target = target_scaler.transform(test_data[["Residual"]])

    # Create sequences
    X_train, y_train = create_sequences(train_scaled, train_target.flatten(), window_size)
    X_test, y_test = create_sequences(test_scaled, test_target.flatten(), window_size)

    print(f"  LSTM Input shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"  Features used: {available_feats}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "target_scaler": target_scaler,
        "feature_cols": feature_cols, "available_feats": available_feats,
        "test_dates": test_data.index[window_size:],
        "test_actual_residuals": test_data["Residual"].values[window_size:],
    }


def run_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    """Hyperparameter tuning via Keras Tuner (RandomSearch)."""
    try:
        import keras_tuner as kt
    except ImportError:
        print("  keras-tuner not installed, using defaults")
        return None

    print("\n  Running hyperparameter tuning...")

    def build_model(hp):
        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(hp.Choice("units_1", [32, 64, 128]), return_sequences=True),
            Dropout(hp.Choice("dropout", [0.1, 0.2, 0.3])),
            LSTM(hp.Choice("units_2", [16, 32, 64]), return_sequences=False),
            Dropout(hp.Choice("dropout_2", [0.1, 0.2, 0.3])),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        lr = hp.Choice("lr", [0.001, 0.0005, 0.0001])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
        return model

    tuner = kt.RandomSearch(
        build_model, objective="val_loss", max_trials=10, seed=RANDOM_SEED,
        directory=os.path.join(MODELS_DIR, "tuner"), project_name="lstm_aqi",
        overwrite=True,
    )

    tuner.search(X_train, y_train, epochs=30, batch_size=32,
                 validation_data=(X_val, y_val), verbose=0,
                 callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"  Best HP: units_1={best_hp.get('units_1')}, units_2={best_hp.get('units_2')}, "
          f"dropout={best_hp.get('dropout')}, lr={best_hp.get('lr')}")
    return best_hp


def train_lstm(df=None, selected_features=None, skip_tuning=False):
    print("\n" + "=" * 60)
    print("  Phase 7a: LSTM MODEL TRAINING")
    print("=" * 60)

    if df is None:
        df = pd.read_csv(os.path.join(DATA_DIR, "processed_data.csv"))
    if selected_features is None:
        sel_path = os.path.join(RESULTS_DIR, "feature_selection.csv")
        if os.path.exists(sel_path):
            sel_df = pd.read_csv(sel_path)
            selected_features = sel_df[sel_df["Final_Selected"] == True]["Feature"].tolist()
        else:
            selected_features = AQI_POLLUTANTS + METEO_FEATURES

    data = prepare_lstm_data(df, selected_features)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Hyperparameter tuning or defaults
    best_hp = None
    if not skip_tuning:
        best_hp = run_hyperparameter_tuning(X_train, y_train, X_test, y_test)

    if best_hp:
        model = build_lstm_model(
            (X_train.shape[1], X_train.shape[2]),
            units_1=best_hp.get("units_1"), units_2=best_hp.get("units_2"),
            dropout=best_hp.get("dropout"), lr=best_hp.get("lr"),
        )
        batch_size = 32
    else:
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
        batch_size = LSTM_DEFAULT_BATCH_SIZE

    print("\n  Model architecture:")
    model.summary(print_fn=lambda x: print(f"     {x}"))

    callbacks = [
        EarlyStopping(patience=LSTM_PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-6, monitor="val_loss"),
    ]

    print(f"\n  Training LSTM (epochs={LSTM_EPOCHS}, batch={batch_size})...")
    history = model.fit(
        X_train, y_train, epochs=LSTM_EPOCHS, batch_size=batch_size,
        validation_data=(X_test, y_test), callbacks=callbacks, verbose=1,
    )

    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    target_scaler = data["target_scaler"]
    y_pred_residuals = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual_residuals = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = evaluate_model(y_actual_residuals, y_pred_residuals, "LSTM (Residuals)")

    # Training history plot
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], color="#FF6B6B", label="Train Loss")
    axes[0].plot(history.history["val_loss"], color="#4ECDC4", label="Val Loss")
    axes[0].set_title("Training Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()

    axes[1].plot(history.history["mae"], color="#FF6B6B", label="Train MAE")
    axes[1].plot(history.history["val_mae"], color="#4ECDC4", label="Val MAE")
    axes[1].set_title("Training MAE", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_PLOTS_DIR, "lstm_training_history.png"))
    plt.close()

    model.save(os.path.join(MODELS_DIR, "lstm_model.keras"))
    joblib.dump(data["scaler"], os.path.join(MODELS_DIR, "lstm_scaler.pkl"))
    joblib.dump(data["target_scaler"], os.path.join(MODELS_DIR, "lstm_target_scaler.pkl"))

    print(f"\n  Phase 7a complete: LSTM trained ({len(history.history['loss'])} epochs)")
    return {
        "model": model, "history": history, "metrics": metrics,
        "y_pred_residuals": y_pred_residuals, "y_actual_residuals": y_actual_residuals,
        "test_dates": data["test_dates"], "data": data,
    }

if __name__ == "__main__":
    train_lstm(skip_tuning=True)
