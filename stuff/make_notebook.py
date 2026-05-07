import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Hybrid ARIMA-LSTM AQI Forecasting — Interactive Walkthrough\n",
            "\n",
            "This notebook runs through the entire Air Quality Index (AQI) forecasting pipeline step-by-step. It uses the modular scripts found in the `src/` directory to perform the heavy lifting, allowing you to interactively explore the data, view the generated visualizations inline, and understand each phase of the process."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%matplotlib inline\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "import sys\n",
            "import os\n",
            "import pandas as pd\n",
            "import importlib\n",
            "from IPython.display import Image, display\n",
            "\n",
            "# Ensure the local modules are discoverable\n",
            "sys.path.insert(0, os.path.abspath('.'))\n",
            "\n",
            "# Define the zone we want to analyze later in the models\n",
            "ZONE_TO_ANALYZE = \"Mathura_Refinery\""
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 1: Synthetic Data Generation\n",
            "\n",
            "First, we generate realistic daily pollutant data for 3 Indian Oil Corporation refinery zones (Mathura, Panipat, Gujarat). This includes seasonal patterns, weekly cycles, and injects ~15% random missingness to simulate real-world sensor issues."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_gen = importlib.import_module(\"src.01_data_generation\")\n",
            "raw_df = mod_gen.generate_all_data()\n",
            "\n",
            "print(f\"\\nGenerated {len(raw_df)} records.\")\n",
            "raw_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 2: Data Preprocessing & AQI Calculation\n",
            "\n",
            "Here we handle the missing values using time-series interpolation and KNN imputation. We also cap outliers, compute the official AQI using the CPCB India formula, and engineer new temporal features (like rolling averages)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_prep = importlib.import_module(\"src.02_data_preprocessing\")\n",
            "processed_df = mod_prep.preprocess_data(raw_df)\n",
            "\n",
            "processed_df.head()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 3: Exploratory Data Analysis (EDA)\n",
            "\n",
            "Let's run the EDA pipeline and view some of the generated plots inline."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_eda = importlib.import_module(\"src.03_eda_visualization\")\n",
            "mod_eda.run_eda(processed_df)\n",
            "\n",
            "print(\"\\n--- Visualizations ---\")\n",
            "display(Image(filename='plots/eda/aqi_time_series.png'))\n",
            "display(Image(filename='plots/eda/monthly_aqi_boxplots.png'))\n",
            "display(Image(filename='plots/eda/zone_comparison.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 4: Feature Selection\n",
            "\n",
            "We run a 3-stage feature selection process: Correlation analysis, Mutual Information, and Recursive Feature Elimination (RFE) via Random Forest. Features that get 2 or more \"votes\" are kept for the LSTM model."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_feat = importlib.import_module(\"src.04_feature_selection\")\n",
            "selected_features = mod_feat.run_feature_selection(processed_df)\n",
            "\n",
            "display(Image(filename='plots/eda/feature_importance.png'))\n",
            "print(\"\\nFinal Selected Features:\", selected_features)"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 5: Seasonal Decomposition & Stationarity\n",
            "\n",
            "Before building the ARIMA model, we analyze the time series for trend and seasonality, and check for stationarity using ADF and KPSS tests."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_decomp = importlib.import_module(\"src.05_seasonal_decomposition\")\n",
            "decomp_results = mod_decomp.run_seasonal_decomposition(processed_df, zone=ZONE_TO_ANALYZE)\n",
            "\n",
            "display(Image(filename='plots/decomposition/stl_decomposition_aqi.png'))\n",
            "display(Image(filename='plots/decomposition/acf_pacf_aqi_original.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 6: ARIMA Model Training\n",
            "\n",
            "We use `pmdarima` to automatically find the best ARIMA order. The model predicts the linear/seasonal part of the time series, and we save its residuals (the errors) for the LSTM to learn from."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_arima = importlib.import_module(\"src.06_arima_model\")\n",
            "arima_results = mod_arima.train_arima(processed_df, zone=ZONE_TO_ANALYZE)\n",
            "\n",
            "display(Image(filename='plots/models/arima_forecast_ci.png'))\n",
            "display(Image(filename='plots/models/arima_diagnostics.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 7: LSTM Model on ARIMA Residuals\n",
            "\n",
            "Now we train an LSTM neural network to predict the *residuals* of the ARIMA model. The LSTM takes in the multivariate features we selected earlier, allowing it to capture non-linear patterns (like sudden pollution spikes from other correlated pollutants)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import config\n",
            "# For a faster notebook run, we use 15 epochs and skip Keras Tuner hyperparameter search.\n",
            "# You can change these to run a full search!\n",
            "config.LSTM_EPOCHS = 15\n",
            "config.LSTM_PATIENCE = 5\n",
            "\n",
            "mod_lstm = importlib.import_module(\"src.07_lstm_model\")\n",
            "lstm_results = mod_lstm.train_lstm(processed_df, selected_features, skip_tuning=True)\n",
            "\n",
            "display(Image(filename='plots/models/lstm_training_history.png'))"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Phase 8: Hybrid Combination & Future Forecasting\n",
            "\n",
            "Finally, we combine the predictions: `Hybrid Forecast = ARIMA Prediction + LSTM Residual Prediction`.\n",
            "We evaluate how much the hybrid model improves over standalone ARIMA, and then forecast 30 days into the future."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "mod_hybrid = importlib.import_module(\"src.08_hybrid_forecast\")\n",
            "hybrid_results = mod_hybrid.combine_hybrid_forecast(arima_results, lstm_results)\n",
            "\n",
            "display(Image(filename='plots/models/hybrid_decomposition.png'))\n",
            "display(Image(filename='plots/models/model_comparison.png'))\n",
            "\n",
            "print(\"\\n--- 30-Day Future Forecast ---\")\n",
            "forecast_df = mod_hybrid.forecast_future(arima_results['model'], n_days=30)\n",
            "display(Image(filename='plots/models/future_forecast.png'))\n",
            "forecast_df.head(10)"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('Interactive_Walkthrough.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print("Notebook generated successfully.")
