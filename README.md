# 📈 Stock Price Prediction with LSTM & Attention: For American Airlines (AAL), Tesla (TSLA), and Microsoft (MSFT)

This project applies **Long Short-Term Memory (LSTM)** neural networks, enhanced with **Self-Attention**, to predict the stock prices of **American Airlines (AAL), Tesla (TSLA), and Microsoft (MSFT)** using the last 60 days of historical data.

The goal is to forecast the next 4 days of stock prices, emphasizing model flexibility, sequential learning, and real-time usability.

You can change the ticker and input any stock you want. For the sake of this project, though, we are only focusing on those three.

(NOTE: The last code segment in the file is for you to input any date you desire to see what the next 4 days will be like.)
(NOTE: Key for file names: LSAMP is the company I am working with to conduct this project. FP stands for Finance Project.)

---

## 🔍 Project Summary

- **Stock**: American Airlines (Ticker: AAL), Tesla (Ticker: TSLA), and Microsoft (Ticker: MSFT)
- **Model Type**: Deep Learning (LSTM + Additive Attention)
- **Timeframe**: Historical data from 2015 to 2025
- **Forecast**: Predict the next 4 days of closing prices
- **Key Features**:
  - Scaled input data
  - 2-Layer LSTM with self-attention mechanism
  - Interactive date-based prediction
  - Real-time stock fetching via Yahoo Finance API
  - Evaluation: MAE, RMSE, and Loss
  - Visual output using Matplotlib & mplfinance

---

## 🚀 Features

- 📊 Predict next 4-day closing prices from past 60 days
- 📌 User-input interface to simulate predictions from past points
- 🧠 Self-attention mechanism built into LSTM model
- 📈 Visuals: candlestick chart + prediction overlay
- 🧪 Evaluation metrics: MAE, RMSE, MSE
- 📦 Extendable for other stocks like MSFT and TSLA

---

🧠 Tech Stack

 - Python 3.9+
 - TensorFlow / Keras
 - NumPy, Pandas, Matplotlib
 - yfinance (data fetching)
 - mplfinance (candlestick charting)

⚙️ Model Architecture

 - Input: Previous 60 closing prices
 - 2 LSTM layers (50 units each, return sequences)
 - Self-Attention (Keras AdditiveAttention)
 - Dense output layer (predicts next day's price)
 - Loss Function: Mean Squared Error
 - Optimizer: Adam
 - Evaluation Metrics: MAE, RMSE

