# Stock Market Prediction using LSTM & Attention (MSFT, TSLA, AAL)

This project was funded by the LSAMP Program's "Research Experience" project - SP25. It was for students to conduct research in an area of their interest. 
I was allowed to research the stock market using the model of my choice. It was to be presented at Molloy University in front of other researchers on 4/28/2025.
Link to presentation slot (proof): https://digitalcommons.molloy.edu/murc/2025/all/26/
This project uses a deep learning model to predict future stock prices based on historical data and technical indicators.

---

## Overview

- Predicts **next 4 days** of stock prices using past **100 days of technical data**
- Built with **LSTM + Self-Attention** for capturing long-term dependencies
- Integrates **RSI, MACD, Bollinger Bands** as input features
- Visualizes actual vs. predicted prices using **mplfinance** and **matplotlib**

---

## Academic Research Paper

This project was officially documented in a research paper titled:

"An Efficient Deep Learning Model for Stock Market Prediction:
LSTM & Technical Indicators for Short-Term Forecasting"
Ayobami Makinde & Arturo Franco
Department of Computer Science and Mathematics
St Joseph’s University & Molloy University
Advisor: Dr. Helen Dang
April 28, 2025.

The paper explains the methodology, experiments, results, and future work in greater detail.

---

## Features

- Supports MSFT, TSLA, and AAL stocks
- Uses `yfinance` to fetch historical OHLCV data
- Real-time prediction & visualization
- Forecasts generated using a rolling window approach
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

---

## References & Inspiration

This project was inspired by:

- [Advanced Stock Pattern Prediction Using LSTM with Attention (Dr. Lee)](https://drlee.io/advanced-stock-pattern-prediction-using-lstm-with-the-attention-mechanism-in-tensorflow-a-step-by-143a2e8b0e95)
- Goodfellow, I., Bengio, Y., & Courville, A. — Deep Learning (MIT Press). http://www.deeplearningbook.org/

Their work helped shape the attention-based forecasting architecture implemented here.

---

## Acknowledgments

LSAMP Research Program for supporting undergraduate research in STEM.

Dr. Helen Dang for advising and guiding this project.

St. Joseph’s University & Molloy University for providing academic resources.

---

## Last Updated: April 28, 2025

