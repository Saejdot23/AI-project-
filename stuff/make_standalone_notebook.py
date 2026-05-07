import json
import os

def create_standalone_notebook():
    cells = []
    
    # 1. Introduction
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Hybrid ARIMA-LSTM AQI Forecasting Model\n",
            "\n",
            "This is a standalone, all-in-one Jupyter Notebook that contains the complete pipeline for forecasting Air Quality Index (AQI) around Indian Oil Corporation refinery zones using a hybrid ARIMA-LSTM approach.\n",
            "\n",
            "### Pipeline Phases:\n",
            "1. **Configuration & Setup**\n",
            "2. **Synthetic Data Generation**\n",
            "3. **Data Preprocessing**\n",
            "4. **Exploratory Data Analysis (EDA)**\n",
            "5. **Feature Selection**\n",
            "6. **Seasonal Decomposition**\n",
            "7. **ARIMA Modeling**\n",
            "8. **LSTM Modeling**\n",
            "9. **Hybrid Forecast Combination**"
        ]
    })
    
    # 2. Base Imports & Setup
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "%matplotlib inline\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "import os\n",
            "import sys\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n",
            "from IPython.display import display\n"
        ]
    })
    
    # Function to read a file and extract code (removing relative imports and sys.path)
    def process_file(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        filtered_lines = []
        skip_mode = False
        for line in lines:
            if "sys.path.insert" in line or "import sys" in line and not skip_mode:
                continue
            if "from config import" in line or "from src.utils import" in line:
                continue
            filtered_lines.append(line)
        return "".join(filtered_lines)
    
    # 3. Config & Utils
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 1. Configuration & Utility Functions"]
    })
    
    with open("config.py", 'r', encoding='utf-8') as f:
        config_code = f.read()
    
    utils_code = process_file("src/utils.py")
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": config_code + "\n\n" + utils_code
    })
    
    # 4. Data Generation
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 2. Synthetic Data Generation"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/01_data_generation.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "raw_df = generate_all_data()\ndisplay(raw_df.head())"
    })
    
    # 5. Preprocessing
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 3. Data Preprocessing & AQI Calculation"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/02_data_preprocessing.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "processed_df = preprocess_data(raw_df)\ndisplay(processed_df.head())"
    })
    
    # 6. EDA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 4. Exploratory Data Analysis (EDA)"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/03_eda_visualization.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "run_eda(processed_df)"
    })
    
    # 7. Feature Selection
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 5. Feature Selection"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/04_feature_selection.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "selected_features = run_feature_selection(processed_df)\nprint(\"Selected Features:\", selected_features)"
    })
    
    # 8. Seasonal Decomposition
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 6. Seasonal Decomposition & Stationarity"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/05_seasonal_decomposition.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "ZONE_TO_ANALYZE = \"Mathura_Refinery\"\ndecomp_results = run_seasonal_decomposition(processed_df, zone=ZONE_TO_ANALYZE)"
    })
    
    # 9. ARIMA
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 7. ARIMA Modeling"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/06_arima_model.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "arima_results = train_arima(processed_df, zone=ZONE_TO_ANALYZE)"
    })
    
    # 10. LSTM
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 8. LSTM Modeling (on ARIMA Residuals)"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/07_lstm_model.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "# For faster execution in the notebook, we'll set epochs to 15 and skip Keras Tuner\nLSTM_EPOCHS = 15\nLSTM_PATIENCE = 5\nlstm_results = train_lstm(processed_df, selected_features, skip_tuning=True)"
    })
    
    # 11. Hybrid Combine
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 9. Hybrid Combination & Future Forecast"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": process_file("src/08_hybrid_forecast.py")
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": "hybrid_results = combine_hybrid_forecast(arima_results, lstm_results)\n\nprint(\"\\n--- 30-Day Future Forecast ---\")\nforecast_df = forecast_future(arima_results['model'], n_days=30)\ndisplay(forecast_df.head(10))"
    })
    
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
    
    with open('Standalone_AQI_Forecasting.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    create_standalone_notebook()
    print("Standalone notebook generated.")
