from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

def predict_stock_price(historical_data):
    # Extract features and target variable
    features = historical_data.dropna()[['Open', 'High', 'Low', 'Close', 'Volume']]  # Example features
    target = historical_data.dropna()['Close']  # Target variable

    # Train a simple linear regression model
    model = LinearRegression()
    model.fit(features, target)

    # Use the latest data for prediction (assuming the last row is the latest data)
    latest_data = features.iloc[[-1]]

    # Predict the next day's stock price
    predicted_price = model.predict(latest_data)

    return predicted_price[0]  # Return the predicted price as a single value

#arima

def predict_stock_price_lstm(historical_data):
    try:
        # Extract 'Close' prices
        close_prices = historical_data['Close'].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_close_prices = scaler.fit_transform(close_prices)

        # Prepare data for LSTM (use a time window of 10 days to predict the next day)
        def create_dataset(data, time_steps=1):
            X, y = [], []
            for i in range(len(data) - time_steps - 1):  # Adjusted loop range
                X.append(data[i:(i + time_steps), 0])
                y.append(data[i + time_steps, 0])
            return np.array(X), np.array(y)

        time_steps = 10  # Define the number of time steps (days) to consider
        X, y = create_dataset(scaled_close_prices, time_steps)

        # Reshape input data for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(units=1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the model
        model.fit(X, y, epochs=100, batch_size=32)

        # Make predictions
        test_input = scaled_close_prices[-time_steps - 1:-1, 0]  # Input for prediction
        test_input = test_input.reshape(1, -1)
        test_input = np.reshape(test_input, (1, time_steps, 1))

        predicted_price = model.predict(test_input)
        
        # Inverse transform the predictions to original scale
        predicted_price = scaler.inverse_transform(predicted_price)

        return predicted_price[0][0]
    except Exception as e:
        print("Exception:", e)
        return None  # Return None if an error occurs

#random forest
def predict_stock_price_random_forest(historical_data):
    # Prepare features and target variable
    features = historical_data.dropna()[['Open', 'High', 'Low', 'Volume']]  # Example features
    target = historical_data.dropna()['Close']  # Target variable

    # Initialize and train the Random Forest Regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features, target)

    # Use the latest available data for prediction
    latest_data = features.iloc[[-1]]

    # Predict the next day's stock price
    predicted_price = model.predict(latest_data)

    return predicted_price[0]  # Return the predicted price as a single value