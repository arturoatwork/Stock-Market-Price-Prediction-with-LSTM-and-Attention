# Market-Direction-Prediction-with-LSTM-Global-Indices-and-News-Sentiment
This project builds an LSTM-based deep learning model to predict the next-day direction (up or down) of major U.S. stock market indices, including the S&amp;P 500, NASDAQ, Dow Jones, Russell 2000, and NYSE Composite. The model combines historical price data with technical indicators, global market trends, and news sentiment analysis.

Features:
LSTM (Long Short-Term Memory) model for time series classification

Technical indicators: rolling moving averages and momentum signals

Global market influence: includes Nikkei (Japan), FTSE (UK), and DAX (Germany)

News sentiment scoring using NLP (TextBlob) and the NewsAPI

Walk-forward backtesting for realistic evaluation

Precision-based thresholding to avoid low-confidence trades

Tools & Libraries:
Python, Pandas, Scikit-learn

TensorFlow / Keras (LSTM)

yfinance (market data)

NewsAPI + TextBlob (sentiment analysis)

Matplotlib (visualization)

Goal:
Improve prediction accuracy and trading signal quality by fusing structured data (price, volume, international markets) and unstructured data (financial news sentiment).

This project demonstrates the power of multi-modal data integration and serves as a foundation for building advanced quantitative trading strategies.
