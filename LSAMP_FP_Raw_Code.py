# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
!pip install tensorflow -qqq
!pip install keras -qqq
!pip install yfinance -qqq

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Fetch any three of the stock data (AAL, TSLA, and MSFT) using yfinance
stock_data = yf.download('AAL', start='2015-01-01', end='2025-01-01') #change the ticker symbol as needed

# Display the first few rows of the dataframe to make sure the data is loaded correctly
stock_data.head()

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# small data cleaning now

# Checking for missing values
stock_data.isnull().sum()

# Filling missing values, if any
stock_data.fillna(method='ffill', inplace=True)

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
# Scaling the data to a range of 0 to 1
scaler = MinMaxScaler(feature_range=(0,1))
stock_data_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1,1))


# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Preparing the data for LSTM
X = []
y = []

for i in range(60, len(stock_data_scaled)):
    X.append(stock_data_scaled[i-60:i, 0])
    y.append(stock_data_scaled[i, 0])

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply

model = Sequential()

# Adding LSTM layers with return_sequences=True
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))


# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adding self-attention mechanism
# The attention mechanism
attention = AdditiveAttention(name='attention_weight')

# Define the input explicitly
input_layer = tf.keras.layers.Input(shape=(X_train.shape[1], 1))
x = LSTM(units=50, return_sequences=True)(input_layer)
x = LSTM(units=50, return_sequences=True)(x)

# Permute and reshape for compatibility
x = Permute((2, 1))(x)
x = Reshape((-1, X_train.shape[1]))(x)

# Apply attention
attention_result = attention([x, x])
x = Multiply()([x, attention_result])

# Return to original shape
x = Permute((2, 1))(x)
x = Reshape((-1, 50))(x)

# Adding a Flatten layer before the final Dense layer
x = tf.keras.layers.Flatten()(x)

# Final Dense layer
output_layer = Dense(1)(x)

# the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
from keras.layers import BatchNormalization, Dropout

# Adding Dropout and Batch Normalization
x = Dropout(0.2)(x)
x = BatchNormalization()(x)


# %%-------------------------------------------------------------------------------------------------
model.compile(optimizer='adam', loss='mean_squared_error')

# %%-------------------------------------------------------------------------------------------------
model.summary()

# %%-------------------------------------------------------------------------------------------------
# Assuming X_train and y_train are already defined and preprocessed
history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2)

# %%-------------------------------------------------------------------------------------------------
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=25, validation_split=0.2, callbacks=[early_stopping])


# %%-------------------------------------------------------------------------------------------------
# Convert X_test and y_test to Numpy arrays if they are not already
X_test = np.array(X_test)
y_test = np.array(y_test)

# Ensure X_test is reshaped similarly to how X_train was reshaped
# This depends on how you preprocessed the training data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Now evaluate the model on the test data
test_loss = model.evaluate(X_test, y_test)
print("Test Loss: ", test_loss)

# %%-------------------------------------------------------------------------------------------------
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Making predictions
y_pred = model.predict(X_test)

# Calculating MAE and RMSE
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)

# %%-------------------------------------------------------------------------------------------------
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetching the latest 60 days of stock stock data
data = yf.download('AAL', period='60d', interval='1d') #change the ticker symbol as needed

# Selecting the 'Close' price and converting to numpy array
closing_prices = data['Close'].values

# Scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(closing_prices.reshape(-1,1))

# Since we need the last 60 days to predict the next day, we reshape the data accordingly
X_latest = np.array([scaled_data[-60:].reshape(60)])

# Reshaping the data for the model (adding batch dimension)
X_latest = np.reshape(X_latest, (X_latest.shape[0], X_latest.shape[1], 1))

# Making predictions for the next 4 candles
predicted_stock_price = model.predict(X_latest)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

print("Predicted Stock Prices for the next 4 days: ", predicted_stock_price)


# %%-------------------------------------------------------------------------------------------------
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fetch the latest 60 days of desired stock data
data = yf.download('AAL', period='60d', interval='1d') #change the ticker symbol as needed

# Select 'Close' price and scale it
closing_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Predict the next 4 days iteratively
predicted_prices = []
current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

for i in range(4):  # Predicting 4 days
    # Get the prediction (next day)
    next_prediction = model.predict(current_batch)
    
    # Reshape the prediction to fit the batch dimension
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
    
    # Append the prediction to the batch used for predicting
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
    
    # Inverse transform the prediction to the original price scale
    predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

print("Predicted Stock Prices for the next 4 days: ", predicted_prices)


# %%-------------------------------------------------------------------------------------------------
!pip install mplfinance -qqq
import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame with the fetched stock data
# Flatten multi-index columns if necessary
if isinstance(data.columns, pd.MultiIndex):
	data.columns = [col[0] for col in data.columns]

# Ensure all required columns are of type float or int
data = data.astype({'Open': 'float', 'High': 'float', 'Low': 'float', 'Close': 'float', 'Volume': 'int'})

# Creating a list of dates for the predictions
last_date = data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Assuming 'predicted_prices' is your list of predicted prices for the next 4 days
predictions_df = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

# Plotting the actual data with mplfinance
mpf.plot(data, type='candle', style='charles', volume=True)

# Overlaying the predicted data
plt.figure(figsize=(10,6))
plt.plot(predictions_df.index, predictions_df['Close'], linestyle='dashed', marker='o', color='red')

plt.title(" 'AAL' Stock Price with Predicted Next 4 Days ") #change the ticker symbol as needed
plt.show()

# %%-------------------------------------------------------------------------------------------------
import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt

# Fetch the latest 60 days of stock data
data = yf.download('AAL', period='64d', interval='1d') # Fetch 64 days to display last 60 days in the chart # change the ticker symbol as needed

# Select 'Close' price and scale it
closing_prices = data['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Predict the next 4 days 
predicted_prices = []
current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

for i in range(4):  # Predicting 4 days
    next_prediction = model.predict(current_batch)
    next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
    current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
    predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

# Creating a list of dates for the predictions
last_date = data.index[-1]
next_day = last_date + pd.Timedelta(days=1)
prediction_dates = pd.date_range(start=next_day, periods=4)

# Adding preds to the df
predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

# Combining actual and predicted data
combined_data = pd.concat([data['Close'], predicted_data['Close']])
combined_data = combined_data[-64:] # Last 60 days of actual data + 4 days of predictions

# Plotting the real data
plt.figure(figsize=(10,6))
plt.plot(data.index[-60:], data['Close'][-60:], linestyle='-', marker='o', color='blue', label='Actual Data')

# Plotting the predicted data
plt.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Data')

plt.title(" 'AAL' Stock Price: Last 60 Days and Next 4 Days Predicted") # change the ticker symbol as needed
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# %%-------------------------------------------------------------------------------------------------
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


#Model to predict next 4 days of any stock price depending on the day user inputs

# Load the trained model
def predict_stock_price(input_date):
    # Check if the data is in valid date format
    try:
        input_date = pd.to_datetime(input_date)
    except ValueError:
        print("Invalid Date Format. Please enter date in YYYY-MM-DD format.")
        return

    # Get data from yfinance
    end_date = input_date
    start_date = input_date - timedelta(days=90)  # Get more days to ensure we have 60 days
    data = yf.download('AAL', start=start_date, end=end_date)

    if len(data) < 60:
        print("Not enough historical data to make a prediction. Try an earlier date.")
        return

    # Prepare the data
    closing_prices = data['Close'].values[-60:]  # Last 60 days
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(closing_prices.reshape(-1, 1))

    # Make predictions
    predicted_prices = []
    current_batch = scaled_data.reshape(1, 60, 1)

    for i in range(4):  # Predicting 4 days
        next_prediction = model.predict(current_batch)
        next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
        current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
        predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

    # Output the predictions
    for i, price in enumerate(predicted_prices, 1):
        print(f"Day {i} prediction: {price}")

# Output predictions for the next 4 days
user_input = input("Enter a date (YYYY-MM-DD) to predict 'AAL' stock for the next 4 days: ") # change the ticker symbol as needed
predict_stock_price(user_input)


