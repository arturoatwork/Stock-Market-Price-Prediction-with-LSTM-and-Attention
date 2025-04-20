
#install the required libraries
!pip install tensorflow -qqq
!pip install keras -qqq
!pip install yfinance -qqq
!pip install ta -qqq
!pip install pandas -qqq
!pip install numpy -qqq
!pip install matplotlib -qqq
!pip install scikit-learn -qqq
!pip install mplfinance -qqq

# %%
# Importing necessary libraries after install
import tensorflow as tf
import yfinance as yf
import numpy as np
import datetime 
import pandas as pd
import ta
import matplotlib.pyplot as plt
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, BatchNormalization, AdditiveAttention, Permute, Reshape, Multiply
from keras.callbacks import EarlyStopping

# Global constants
WINDOW_SIZE = 100
FEATURE_COUNT = 5


# %%
#User selects a ticker and fetches data for multiple stocks

print("Available tickers: MSFT, TSLA, AAL")

# Allow up to 3 attempts to enter a valid ticker
attempts = 3
selected_ticker = None

for attempt in range(attempts):
    selected_ticker = input("Please enter the ticker you want to analyze: ").upper()
    
    # Validate the input
    if selected_ticker in ['MSFT', 'TSLA', 'AAL']:
        print(f"You selected: {selected_ticker}")
        break
    else:
        print(f"Invalid ticker. You have {attempts - attempt - 1} attempts remaining.")

# If all attempts fail, display a message and exit
if selected_ticker not in ['MSFT', 'TSLA', 'AAL']:
    print("Please take your time typing the ticker, restart the program now.")
    raise SystemExit

# Fetch data for multiple stocks
tickers = ['MSFT', 'TSLA', 'AAL']
stock_data = {}
today = datetime.datetime.today().strftime('%Y-%m-%d')

for ticker in tickers:
    stock_data[ticker] = yf.download(ticker, start='2020-01-01', end=today)


    
# Display the first few rows of the selected ticker's dataframe
print(f"Displaying data for {selected_ticker}:")
print(stock_data[selected_ticker].head())


# %%
# Fetch extended stock data for the selected ticker
today = datetime.datetime.today().strftime('%Y-%m-%d')
data = yf.download(selected_ticker, start='2020-01-01', end=today)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

# Technical Indicators
close_prices = data['Close'].squeeze()
data['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()
data['MACD'] = ta.trend.MACD(close=close_prices).macd()
bollinger = ta.volatility.BollingerBands(close=close_prices)
data['Bollinger_High'] = bollinger.bollinger_hband()
data['Bollinger_Low'] = bollinger.bollinger_lband()

# Handle NaNs (due to indicators)
data.dropna(inplace=True)

# Display the first few rows with indicators
print(data.head())

# %%
# Small data cleaning
# Checking for missing values
missing_values = stock_data[ticker].isnull().sum().sum()  # Total number of missing values in the DataFrame

if missing_values > 0:
    print(f"Missing values detected: {missing_values}. Handling missing values...")
    
    # Filling missing values, if any
    stock_data[ticker].fillna(method='ffill', inplace=True)  # Forward fill
    stock_data[ticker].fillna(method='bfill', inplace=True)  # Backward fill

    # Recalculate technical indicators after filling missing values
    close_prices = stock_data[ticker]['Close'].squeeze()

    # Calculate RSI
    stock_data[ticker]['RSI'] = ta.momentum.RSIIndicator(close=close_prices).rsi()

    # Calculate MACD
    stock_data[ticker]['MACD'] = ta.trend.MACD(close=close_prices).macd()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=close_prices)
    stock_data[ticker]['Bollinger_High'] = bollinger.bollinger_hband()
    stock_data[ticker]['Bollinger_Low'] = bollinger.bollinger_lband()

    # Drop any remaining NaN values caused by indicator calculations
    stock_data[ticker].dropna(inplace=True)

    print("Missing values handled and technical indicators recalculated.")
else:
    print("No missing values detected. Skipping missing value handling.")

# Display the first few rows with indicators placed
print("Data after processing:")
print(stock_data[ticker].head())

# %%
# Ensure the required columns exist
required_columns = ['Close', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"'{col}' column not found in the data. Please check the data source.")

# Drop NaNs before fitting
data.dropna(inplace=True)

# Fit the scaler once on all required features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data[required_columns])
print(f"Data for {selected_ticker} has been scaled with {len(required_columns)} features.")

# Save the last 60 rows for future prediction
latest_data = data[required_columns].values[-60:]
scaled_latest_data = scaler.transform(latest_data)
X_latest = np.reshape(scaled_latest_data, (1, 60, len(required_columns)))  # Shape: (1, 60, 5)


# %%
# Preparing the data for LSTM
X, y = [], []

# Use scaled data for the selected ticker
for i in range(WINDOW_SIZE, len(scaled_data)):
    X.append(scaled_data[i - WINDOW_SIZE:i])
    y.append(scaled_data[i, 0])  # Predict Close

X, y = np.array(X), np.array(y)
print(f"Prepared LSTM data for {selected_ticker} stock.")
print(f"X shape: {X.shape}, y shape: {y.shape}")


# %%
# Splitting the data into training and testing sets
# 80% for training and 20% for testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Data split completed for {selected_ticker}.")
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")


# %%
# Reshaping the training data for LSTM
X_train = np.array(X_train)
y_train = np.array(y_train)

# Ensure the shape is (num_samples, timesteps, num_features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
print(f"Reshaped X_train: {X_train.shape}")

# %%
# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, FEATURE_COUNT)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print(f"LSTM model built for {selected_ticker}.")

# %%
# Adding self-attention mechanism
# The attention mechanism
attention = AdditiveAttention(name='attention_weight')

# Define the input explicitly
input_layer = tf.keras.layers.Input(shape=(WINDOW_SIZE, FEATURE_COUNT))  # Use global constants for timesteps and features
x = LSTM(units=50, return_sequences=True)(input_layer)
x = LSTM(units=50, return_sequences=True)(x)

# Permute and reshape for compatibility
x = Permute((2, 1))(x)
x = Reshape((-1, WINDOW_SIZE))(x)  # Use global constant for timesteps

# Apply attention
attention_result = attention([x, x])
x = Multiply()([x, attention_result])

# Return to original shape
x = Permute((2, 1))(x)
x = Reshape((-1, 50))(x)

# Adding a Flatten layer before the final Dense layer
x = tf.keras.layers.Flatten()(x)
output_layer = Dense(1)(x)

# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

print(f"Self-attention LSTM model built for {selected_ticker}.")

# %%
# Adding Dropout and Batch Normalization
x = Dropout(0.2)(x)
x = BatchNormalization()(x)

print(f"Dropout and Batch Normalization layers added for {selected_ticker}.")

# %%
model.summary()

# %%
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")

# %%
# Adding Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# Compile the model so its called properly to work
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])

# %%
# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Verify the shape of X_test before reshaping
print(f"Shape of X_test before reshaping: {X_test.shape}")

# Reshape X_test to match the shape of X_train
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_train.shape[2]))

# Verify the shape of X_test after reshaping
print(f"Shape of X_test after reshaping: {X_test.shape}")

# Evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")

# %%
# Making predictions
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Mean Absolute Error for {selected_ticker}: ", mae)
print(f"Root Mean Square Error for {selected_ticker}: ", rmse)

# %%
# Set global constants
required_columns = ['Close', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']

# Re-fetch latest stock data for inference
data = yf.download(selected_ticker, period='150d', interval='1d')
close_series = data['Close'].squeeze()

# Add indicators
data['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
data['MACD'] = ta.trend.MACD(close=close_series).macd()
bb = ta.volatility.BollingerBands(close=close_series)
data['Bollinger_High'] = bb.bollinger_hband()
data['Bollinger_Low'] = bb.bollinger_lband()

# Clean NaNs
data.dropna(inplace=True)

# Check enough data
if len(data) < WINDOW_SIZE:
    raise ValueError(f"Not enough rows. Need at least {WINDOW_SIZE}, got {len(data)}.")

# To fit new scaler on just the recent WINDOW_SIZE rows
recent_data = data[required_columns].iloc[-WINDOW_SIZE:]
scaler = MinMaxScaler()
scaled_recent = scaler.fit_transform(recent_data)

# Prepare model input
X_latest = scaled_recent.reshape(1, WINDOW_SIZE, FEATURE_COUNT)
print(f"X_latest shape for model: {X_latest.shape}")


# %%
# Run rolling prediction for 4 days
predicted_scaled = []
current_batch = X_latest.copy()

for _ in range(4):
    next_scaled = model.predict(current_batch)
    predicted_scaled.append(next_scaled[0, 0])  # Save only 'Close'

    # Pad with dummy 0s for RSI, MACD, etc.
    next_input = np.array([[next_scaled[0, 0], 0, 0, 0, 0]])
    current_batch = np.append(current_batch[:, 1:, :], [next_input], axis=1)

# Pad predictions with dummy 0s to match scaler shape (4, 5)
preds_with_zeros = np.concatenate([
    np.array(predicted_scaled).reshape(-1, 1),
    np.zeros((4, FEATURE_COUNT - 1))  # pad other features with 0s
], axis=1)

# Now safely inverse transform and extract just the 'Close' (column 0)
predicted_prices = scaler.inverse_transform(preds_with_zeros)[:, 0]
print(f"Predicted Prices for {selected_ticker}: {predicted_prices}")


# %%
# Plotting the predictions
# Fetch full OHLCV data for charting
plot_data = yf.download(selected_ticker, period=f'{WINDOW_SIZE + 20}d', interval='1d')

# Flatten MultiIndex if any
if isinstance(plot_data.columns, pd.MultiIndex):
    plot_data.columns = [col[0] for col in plot_data.columns]

# Clean OHLCV types
required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
missing_cols = [col for col in required_ohlcv if col not in plot_data.columns]
if missing_cols:
    raise KeyError(f"Missing columns in data: {missing_cols}")

plot_data[required_ohlcv] = plot_data[required_ohlcv].apply(pd.to_numeric, errors='coerce')
plot_data.dropna(subset=required_ohlcv, inplace=True)

# Confirm enough rows to plot
if len(plot_data) < WINDOW_SIZE:
    raise ValueError(f"Not enough rows to plot. Only {len(plot_data)} rows available.")

#Generate prediction dates
last_date = plot_data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

#Build predictions DataFrame
predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

#Plot candlestick chart (last 100 days)
mpf.plot(plot_data[-WINDOW_SIZE:], type='candle', style='charles', volume=True, title=f"{selected_ticker} Stock Price")

#Overlay prediction points
plt.figure(figsize=(10, 6))
plt.plot(predictions_df.index, predictions_df['Close'], linestyle='dashed', marker='o', color='red')
plt.title(f"{selected_ticker} Predicted Prices (Next 4 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()


# %%

# Fetch recent stock data
forecast_data = yf.download(selected_ticker, period='150d', interval='1d')
close_series = forecast_data['Close'].squeeze()

# Add indicators
forecast_data['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
forecast_data['MACD'] = ta.trend.MACD(close=close_series).macd()
bb = ta.volatility.BollingerBands(close=close_series)
forecast_data['Bollinger_High'] = bb.bollinger_hband()
forecast_data['Bollinger_Low'] = bb.bollinger_lband()

# Drop NaNs from indicators
forecast_data.dropna(inplace=True)

# Check enough data available
if len(forecast_data) < WINDOW_SIZE:
    raise ValueError(f"Need at least {WINDOW_SIZE} rows for prediction. Got {len(forecast_data)}.")

# Extract the latest WINDOW_SIZE rows and scale
latest_data = forecast_data[required_columns].values[-WINDOW_SIZE:]
scaled_latest = scaler.transform(latest_data)
current_batch = scaled_latest.reshape(1, WINDOW_SIZE, FEATURE_COUNT)

# Predict next 4 days
predicted_prices = []
for _ in range(4):
    next_scaled = model.predict(current_batch)
    
    # Combine predicted close with dummy zeros for other features
    padded = np.concatenate([next_scaled, np.zeros((1, FEATURE_COUNT - 1))], axis=1)
    predicted_price = scaler.inverse_transform(padded)[0, 0]  # Only care about 'Close'
    predicted_prices.append(predicted_price)
    
    # Prepare input for next day
    next_input = np.concatenate([next_scaled, np.zeros((1, FEATURE_COUNT - 1))], axis=1)
    current_batch = np.append(current_batch[:, 1:, :], [next_input], axis=1)

# Create prediction date index
last_date = forecast_data.index[-1]
prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=4)
predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(forecast_data.index[-WINDOW_SIZE:], forecast_data['Close'][-WINDOW_SIZE:], label="Actual Price", color='blue')
plt.plot(predictions_df.index, predictions_df['Close'], linestyle='--', marker='o', color='red', label="Predicted Price")
plt.title(f"{selected_ticker} Closing Price — Last {WINDOW_SIZE} Days + Next 4-Day Forecast")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()



