# 📈 Stock Market Prediction using LSTM & Attention (MSFT, TSLA, AAL)

This project uses a deep learning model to predict future stock prices based on historical data and technical indicators.

---

## Overview

- Predicts **next 4 days** of stock prices using past **100 days of technical data**
- Built with **LSTM + Self-Attention** for capturing long-term dependencies
- Integrates **RSI, MACD, Bollinger Bands** as input features
- Visualizes actual vs. predicted prices using **mplfinance** and **matplotlib**

---

## Features

- Supports MSFT, TSLA, and AAL stocks
- Uses `yfinance` to fetch historical OHLCV data
- Real-time prediction & visualization
- Forecasts generated using rolling window approach
- Metrics: MAE, RMSE, test loss

---

## Model Architecture

- 3-layer LSTM with 50 units each
- Additive Self-Attention mechanism
- Dropout & BatchNormalization for regularization
- Final Dense layer for price output

---

## Tech Stack

- Python, TensorFlow, Keras
- scikit-learn, pandas, numpy
- yfinance (data), ta (indicators), matplotlib, mplfinance



