# Bitcoin Hourly Price Prediction Dashboard

A real-time Bitcoin price forecasting application built with a Random Forest Regressor model, deployed as an interactive web dashboard using Streamlit. The model ingests the last 24 hours of live market data from Binance and outputs hourly price predictions up to 24 hours ahead.

**Live App:** [btc-priceprediction.streamlit.app](https://btc-priceprediction.streamlit.app)

---

## Overview

This project demonstrates an end-to-end machine learning pipeline for time series forecasting, covering data acquisition, model training, validation, and production deployment. The dashboard provides real-time price metrics, an interactive candlestick chart, and a multi-hour prediction panel, all updated automatically on a configurable refresh interval.

The model was trained on historical hourly OHLCV data sourced from the Binance public API. A conservative validation layer is applied at inference time to prevent unrealistic predictions caused by market anomalies or data noise.

---

## Features

- Live BTC/USDT price ticker with 24-hour high, low, and percentage change
- Next-hour price prediction with directional signal relative to current price
- Multi-hour forecast panel (configurable from 1 to 24 hours ahead)
- 12-hour market outlook with Buy, Sell, or Hold signal and expected move percentage
- Interactive candlestick chart with volume bars (configurable from 1 to 7 days of history)
- Prediction trend chart overlaying historical prices and forecasted values
- Prediction locking: forecasts are cached per hour to prevent per-second fluctuation
- Auto-refresh every 60 seconds with a manual refresh option

---

## Model Performance

| Metric | Value |
|---|---|
| Model Type | Random Forest Regressor |
| Input Features | Last 24 closed hourly close prices (normalized) |
| Prediction Target | Next hour close price |
| Model Accuracy | 99.53% |
| Mean Absolute Percentage Error (MAPE) | 0.47% |
| Training Data | Binance BTCUSDT historical hourly OHLCV |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Dashboard | Streamlit |
| Model Training | Scikit-learn (Random Forest Regressor) |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Model Serialization | Joblib, Pickle |
| Data Source | Binance Vision Public API |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
btc_price_prediction/
|
|-- Random Forest Regressor/
|   |-- rf_btc_prediction_model.pkl   # Trained Random Forest model
|
|-- scaler.pkl                        # MinMaxScaler fitted on training data
|-- config.pkl                        # Model configuration and metadata
|-- dashboard.py                      # Main Streamlit application
|-- requirements.txt                  # Python dependencies
|-- runtime.txt                       # Python version specification
|-- btc_price_prediction_model.ipynb  # Model training notebook
```

---
## Requirements

```
streamlit
pandas
numpy
plotly
requests
joblib
scikit-learn
tensorflow-cpu==2.15.0
```

Python version: 3.11 (specified in `runtime.txt`)

---

## How It Works

**Data Pipeline**

On each refresh cycle, the app fetches the last 25 completed hourly candles from the Binance Vision public API. The most recent (incomplete) candle is discarded to avoid feeding the model a partially formed price point that would otherwise cause the prediction to fluctuate in real time. Only finalized, closed hourly candles are passed to the model.

**Prediction Logic**

The 24 most recent close prices are normalized using the same MinMaxScaler fitted during training, then passed to the Random Forest model as a flat feature vector of shape (1, 24). The model outputs a scaled prediction that is inverse-transformed back to USD.

A validation step checks whether the raw prediction implies a price change greater than 2% relative to the last known close. If so, a conservative fallback is applied using the recent 5-hour price trend at 50% weight. This prevents outlier predictions from being shown to the user during periods of data irregularity.

**Prediction Caching**

Multi-hour forecasts are cached for one hour using Streamlit's `@st.cache_data` decorator with a TTL of 3600 seconds. This ensures predictions remain stable within a given hour and only recalculate when a new hourly candle becomes available, which is consistent with the model's intended temporal resolution.

**Signal Generation**

The 12-hour outlook aggregates the next 12 predicted hourly prices, calculates the average expected percentage move from the current live price, and classifies the market as Bullish (Buy), Bearish (Sell), or Choppy (Hold) using a 0.5% threshold to account for typical fees and slippage.

---

## Deployment Notes

This application is deployed on Streamlit Community Cloud targeting Python 3.11. The Binance Vision mirror endpoint (`data-api.binance.vision`) is used in place of the standard `api.binance.com` endpoint, which is geo-restricted from US-based cloud server infrastructure.

---

## Certifications

Certified in Artificial Intelligence and Data Science — Boston Institute of Analytics, Lahore
(Top Distinguished Performer)

---

## Author

**Muhammad Umair Bilal**
AI/ML Python Engineer

- LinkedIn: [linkedin.com/in/mumairbilal](https://www.linkedin.com/in/mumairbilal)
- GitHub: [github.com/mumairbilal](https://github.com/mumairbilal)
- YouTube: [youtube.com/@xplexis_ai](https://www.youtube.com/@xplexis_ai)
- Email: mrumair157@gmail.com

---

## Disclaimer

This application is intended for educational and informational purposes only. Predictions generated by this model are probabilistic estimates and do not constitute financial advice. Cryptocurrency markets are inherently volatile and unpredictable. Past model performance does not guarantee future accuracy. Do not make investment decisions based solely on this tool.
