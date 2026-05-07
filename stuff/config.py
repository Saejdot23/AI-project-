"""
config.py — Global Configuration for AQI Forecasting Pipeline
==============================================================
Central configuration for the Hybrid ARIMA-LSTM AQI Forecasting Model.
All tunable parameters, file paths, and constants are defined here.
"""

import os

# ──────────────────────────────────────────────────────────────
# DIRECTORY PATHS
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Sub-directories for plots
EDA_PLOTS_DIR = os.path.join(PLOTS_DIR, "eda")
DECOMP_PLOTS_DIR = os.path.join(PLOTS_DIR, "decomposition")
MODEL_PLOTS_DIR = os.path.join(PLOTS_DIR, "models")

# Create all directories
for d in [DATA_DIR, EDA_PLOTS_DIR, DECOMP_PLOTS_DIR, MODEL_PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# DATA GENERATION SETTINGS
# ──────────────────────────────────────────────────────────────
DATE_RANGE_START = "2019-01-01"
DATE_RANGE_END = "2024-12-31"
MISSING_FRACTION = 0.15  # ~15% random missingness

# Refinery zones with baseline pollutant profiles (µg/m³ unless noted)
# Format: {pollutant: (mean, std_dev)}
REFINERY_ZONES = {
    "Mathura_Refinery": {
        "PM2.5": (85, 35),
        "PM10": (160, 55),
        "SO2": (45, 18),
        "NO2": (55, 22),
        "CO": (2.5, 1.0),       # mg/m³
        "O3": (35, 15),
        "NH3": (30, 12),
        "Pb": (0.3, 0.15),
        "Temperature": (28, 8),  # °C
        "Humidity": (60, 18),    # %
        "Wind_Speed": (8, 3.5),  # km/h
    },
    "Panipat_Refinery": {
        "PM2.5": (75, 30),
        "PM10": (145, 50),
        "SO2": (50, 20),
        "NO2": (48, 20),
        "CO": (2.2, 0.9),
        "O3": (40, 16),
        "NH3": (25, 10),
        "Pb": (0.25, 0.12),
        "Temperature": (26, 9),
        "Humidity": (55, 20),
        "Wind_Speed": (9, 4),
    },
    "Gujarat_Refinery": {
        "PM2.5": (65, 25),
        "PM10": (130, 45),
        "SO2": (55, 22),
        "NO2": (42, 18),
        "CO": (1.8, 0.8),
        "O3": (45, 18),
        "NH3": (22, 9),
        "Pb": (0.2, 0.1),
        "Temperature": (32, 6),
        "Humidity": (65, 15),
        "Wind_Speed": (12, 5),
    },
}

# Pollutants used for AQI sub-index calculation
AQI_POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "NH3", "Pb"]

# Meteorological features (not used for AQI calc, but as predictive features)
METEO_FEATURES = ["Temperature", "Humidity", "Wind_Speed"]

# All feature columns
ALL_FEATURES = AQI_POLLUTANTS + METEO_FEATURES

# ──────────────────────────────────────────────────────────────
# AQI BREAKPOINT TABLE (CPCB India Standard)
# Format: list of (BPLo, BPHi, ILo, IHi)
# ──────────────────────────────────────────────────────────────
AQI_BREAKPOINTS = {
    "PM2.5": [  # 24-hr avg, µg/m³
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ],
    "PM10": [  # 24-hr avg, µg/m³
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 600, 401, 500),
    ],
    "SO2": [  # 24-hr avg, µg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2400, 401, 500),
    ],
    "NO2": [  # 24-hr avg, µg/m³
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 600, 401, 500),
    ],
    "CO": [  # 8-hr avg, mg/m³
        (0, 1.0, 0, 50),
        (1.1, 2.0, 51, 100),
        (2.1, 10.0, 101, 200),
        (10.1, 17.0, 201, 300),
        (17.1, 34.0, 301, 400),
        (34.1, 50.0, 401, 500),
    ],
    "O3": [  # 8-hr avg, µg/m³
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500),
    ],
    "NH3": [  # 24-hr avg, µg/m³
        (0, 200, 0, 50),
        (201, 400, 51, 100),
        (401, 800, 101, 200),
        (801, 1200, 201, 300),
        (1201, 1800, 301, 400),
        (1801, 2400, 401, 500),
    ],
    "Pb": [  # 24-hr avg, µg/m³
        (0, 0.5, 0, 50),
        (0.51, 1.0, 51, 100),
        (1.1, 2.0, 101, 200),
        (2.1, 3.0, 201, 300),
        (3.1, 3.5, 301, 400),
        (3.51, 5.0, 401, 500),
    ],
}

# ──────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ──────────────────────────────────────────────────────────────
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test

# ──────────────────────────────────────────────────────────────
# ARIMA CONFIGURATION
# ──────────────────────────────────────────────────────────────
ARIMA_MAX_P = 5
ARIMA_MAX_D = 2
ARIMA_MAX_Q = 5
ARIMA_SEASONAL = True
ARIMA_SEASONAL_PERIOD = 7  # Weekly seasonality (more tractable than 365)

# ──────────────────────────────────────────────────────────────
# LSTM CONFIGURATION
# ──────────────────────────────────────────────────────────────
LSTM_WINDOW_SIZE = 30         # Lookback window in days
LSTM_UNITS_OPTIONS = [32, 64, 128]
LSTM_DROPOUT_OPTIONS = [0.1, 0.2, 0.3]
LSTM_LEARNING_RATE_OPTIONS = [0.001, 0.0005, 0.0001]
LSTM_BATCH_SIZE_OPTIONS = [16, 32, 64]
LSTM_EPOCHS = 100
LSTM_PATIENCE = 15            # EarlyStopping patience

# Default LSTM hyperparameters (used when --skip-tuning)
LSTM_DEFAULT_UNITS_1 = 64
LSTM_DEFAULT_UNITS_2 = 32
LSTM_DEFAULT_DROPOUT = 0.2
LSTM_DEFAULT_LR = 0.001
LSTM_DEFAULT_BATCH_SIZE = 32

# ──────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ──────────────────────────────────────────────────────────────
CORRELATION_THRESHOLD = 0.1   # Min |r| to keep a feature
RFE_N_FEATURES = 8            # Number of features for RFE to select
FEATURE_CONSENSUS_MIN = 2     # Feature must be selected by ≥ N methods

# ──────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 150
COLOR_PALETTE = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

# ──────────────────────────────────────────────────────────────
# RANDOM SEED
# ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
