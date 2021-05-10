"""
Stock prediction using LSTM

This projects showcases performance of an LSTM model for stock prediction on Berkshire Hathaway and Microsoft share data.

There are various standard trading indicators used by investors to make decisions. Some standard trading indicators are:
* closing price
* hlc average (mean of high, low and closing share price)
* ohlc average (mean of open, high, low and closing share price)
etc...

In this project, I have chosen OHLC average as our trading indicator.
"""

import argparse
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def get_ohlc_avg(symbol: str, period='3y'):
    """
    Fetches historical data for the given stock and calculates OHLC_avg.
    """
    data = yf.Ticker(symbol)
    historical_data = data.history(period=period)
    ohlc_avg = pd.DataFrame(historical_data.iloc[:, 0:4].mean(axis=1), columns=['OHLC_avg'])
    return ohlc_avg

def split_data(data: pd.DataFrame, split_parameter='90days'):
    """
    Splitting the data into testing and training datasets
    Testing dataset: Last 90 days (~60 trading days)
    Training dataset: Last 3 years - Last 90 days
    """
    testing_dataframe = ohlc_avg[ohlc_avg.index > datetime.utcnow() - pd.to_timedelta('90days')]
    training_dataframe = ohlc_avg[ohlc_avg.index < testing_dataframe.index[0]]
    print('training dataframe:', training_dataframe.shape)
    print('testing dataframe:', testing_dataframe.shape)
    return training_dataframe, testing_dataframe

def plot_training_testing_distribution(symbol, training_df, testing_df):
    """
    Visualize training and testing split.
    """
    plt.plot(testing_df, color = 'red', label = 'Testing dataset')
    plt.plot(training_df, color = 'blue', label = 'Training dataset')
    plt.title('{} Stock Dataset'.format(symbol))
    plt.xlabel('Time')
    plt.ylabel('{} Stock Price'.format(symbol))
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.show()

def fit_data_scalar(data, scaling_window=700):
    """
    Create MinMaxScalar and fit the training data.
    """
    transformer = MinMaxScaler(feature_range = (0, 1))
    for idx in range(0,len(data),scaling_window):
        transformer.fit(data[idx:idx+scaling_window,:])
        data[idx:idx+scaling_window,:] = transformer.transform(data[idx:idx+scaling_window,:])
    return transformer

def prepare_data_with_timesteps(data, lookback_timesteps: int):
    """
    Here we are preparing the input for LSTM model such that
    we will be curating data with timesteps.
    """
    X_data = []
    y_data = []

    for i in range(lookback_timesteps, len(data)):
        X_data.append(data[i-lookback_timesteps:i, :])
        y_data.append(data[i, 0])

    return np.array(X_data), np.array(y_data)

def create_and_fit_LSTM(X_train_data, y_train_data, epochs=100, batch_size=32):
    """
    Creating and fitting the LSTM model
    """
    stock_prediction_model = Sequential()
    stock_prediction_model.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train_data.shape[1], 1)))
    stock_prediction_model.add(Dropout(0.2))
    stock_prediction_model.add(LSTM(units = 50,return_sequences = True))
    stock_prediction_model.add(Dropout(0.2))
    stock_prediction_model.add(LSTM(units = 50,return_sequences = True))
    stock_prediction_model.add(Dropout(0.2))
    stock_prediction_model.add(LSTM(units = 50))
    stock_prediction_model.add(Dropout(0.2))
    stock_prediction_model.add(Dense(units = 1))
    stock_prediction_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # fit the model
    stock_prediction_model.fit(X_train_data, y_train_data, epochs = epochs, batch_size = batch_size)

    return stock_prediction_model


# Stock prediction using LSTM model and trading suggestion

def suggest_favourable_action(historical_ohlc_avg, next_day_prediction):
    """
    This is a simplistic trading suggestion login which takes into consideration the average share price to
    suggest whether it is favourable to buy, sell or hold the stock. In our code we have taken the average
    share price of last 30 trading days to suggest a favourable action.
    """
    mean_of_historical_ohlc_avg = historical_ohlc_avg.mean()[0]
    if next_day_prediction < mean_of_historical_ohlc_avg:
        return 'buy'
    elif next_day_prediction > mean_of_historical_ohlc_avg:
        return 'sell'
    else:
        return 'hold'

# Stock prediction
def predict_stock_prices(ohlc_avg_data, testing_dataframe_data, model, scaler_transformer, trading_window=30):
    """
    We will use the LSTM model on our test dataset to predict the
    share price and suggest a favourable action.

    Args:
        ohlc_avg_data: complete dataset with OHLC average values
        testing_dataframe_data: testing dataset
        model: trained LSTM model
        scaler_transformer: Scaler transformer for normalizing and denormalizing the data
        trading_window: Number of days you want to consider to calcuate
                        average ohlc data to compare with the next prediction

    Returns:
        predicted_stock_price (list): predicted ohlc stock price
    """
    real_stock_price = testing_dataframe_data.iloc[:, 0:1].values
    predicted_stock_price = []
    predicted_stock_price_scaled = []

    inputs = ohlc_avg_data[len(ohlc_avg_data) - len(testing_dataframe_data) - lookback:].values
    inputs = scaler_transformer.transform(inputs)

    correct = wrong = total = 0
    for i in range(lookback, len(inputs)):
        next_day_input = inputs[i-lookback:i, :]
        next_day_input = next_day_input.reshape(1, -1, 1)

        next_day_predicted_stock_price = model.predict(next_day_input)

        predicted_stock_price_scaled.append(next_day_predicted_stock_price[0][0])

        next_day_predicted_stock_price = scaler_transformer.inverse_transform(next_day_predicted_stock_price)
        predicted_stock_price.append(next_day_predicted_stock_price[0][0])

        trading_window_start_index = len(training_dataframe) + i - lookback - trading_window
        trading_window_end_index = len(training_dataframe) + i - lookback
        tradin_window_data = ohlc_avg_data[trading_window_start_index:trading_window_end_index]

        prediction = suggest_favourable_action(tradin_window_data, next_day_predicted_stock_price[0][0])
        print('Prediction: Tomorrow (utc:{}) it is more favourable to {}'.format(str(testing_dataframe_data.index[i-60].date()), prediction))

        real = suggest_favourable_action(tradin_window_data, real_stock_price[i-lookback])
        if prediction != real:
            wrong += 1
        else:
            correct += 1
        total += 1

    print("\n\nCorrect predictions:", correct)
    print("Wrong predictions:", wrong)
    print("Correct predictions percentage: {}%\n".format(round(correct*100/total, 2)))

    print("RMSE (range: 0 to 1):", mean_squared_error(inputs[60:], predicted_stock_price_scaled, squared=False))

    return predicted_stock_price
    print("\n\nCorrect predictions:", correct)
    print("Wrong predictions:", wrong)
    print("Correct predictions percentage: {}%\n".format(round(correct*100/total, 2)))

    print("RMSE (range: 0 to 1):", mean_squared_error(inputs[60:], predicted_stock_price_scaled, squared=False))

    return predicted_stock_price

def prediction_vs_real(symbol: str, real_values, predicted_values):
    """
    Plots the prediction vs real share values
    """
    plt.figure(figsize = (18,7))
    plt.plot(real_values, color = 'black', label = '{} Stock Price'.format(stock_symbol))
    plt.plot(predicted_values, color = 'green', label = 'Predicted {} Stock Price'.format((stock_symbol)))
    plt.xticks(predicted_values.index.to_pydatetime())
    plt.title('{} Stock Price Prediction'.format(stock_symbol))
    plt.xlabel('Time')
    plt.ylabel('{} Stock Price'.format(stock_symbol))
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.savefig('prediction_vs_real.png')

def plot_full_graph(symbol: str, real_values, predicted_values, train_dataframe):
    """
    Visulaize training data, testing data and predictions together
    """
    plt.figure(figsize = (18,7))
    plt.plot(real_values, color = 'black', label = '{} Stock Price'.format(stock_symbol))
    plt.plot(predicted_values, color = 'green', label = 'Predicted {} Stock Price'.format((stock_symbol)))
    plt.plot(train_dataframe, color = 'blue', label = '{} Stock Price'.format(stock_symbol))
    # plt.xticks(predicted_values.index.to_pydatetime())
    plt.title('{} Stock Price Prediction'.format(stock_symbol))
    plt.xlabel('Time')
    plt.ylabel('{} Stock Price'.format(stock_symbol))
    plt.legend()
    plt.xticks(rotation='vertical')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stock predictor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('stock_symbol', type=str, help='Stock symbol')
    parser.add_argument('--dataperiod', type=str, default='3y', help='Period of data to get from yfinance. \
        Valid periods are: “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”')
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training the model.")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch size for training the model.")
    parser.add_argument('--split_parameter', type=str, default='90days', help="Split parameter for trainig and testing data split. \
        Follows pandas.to_timedelta input format.")
    parser.add_argument('--lookback', type=int, default=60, help="Lookback(timestep) for LSTM.")
    args = parser.parse_args()

    stock_symbol = args.stock_symbol
    dataperiod = args.dataperiod
    epochs = args.epochs
    batch_size = args.batch_size
    split_parameter = args.split_parameter
    lookback = args.lookback

    ohlc_avg = get_ohlc_avg(stock_symbol, dataperiod)
    if ohlc_avg.empty:
        print('No data availabe for stock:',stock_symbol)
        sys.exit(0)

    print(ohlc_avg.shape)
    print(ohlc_avg.head())

    # Split the data into training and testing data
    training_dataframe, testing_dataframe = split_data(ohlc_avg, split_parameter=split_parameter)

    # Visualize data split
    plot_training_testing_distribution(stock_symbol, training_dataframe, testing_dataframe)

    # normalize data between [0,1] using MinMax Scaler
    training_dataset = training_dataframe.values
    training_dataset_scaled = np.copy(training_dataset)
    transformer = fit_data_scalar(training_dataset_scaled)

    # Prepare data with timesteps to feed it into LSTM model
    X_train, y_train = prepare_data_with_timesteps(training_dataset_scaled, lookback)

    # Create and fit the model
    stock_prediction_model = create_and_fit_LSTM(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Predict on test set
    predicted_ohlc_prices = predict_stock_prices(ohlc_avg, testing_dataframe, stock_prediction_model, transformer)

    # Visulaize predictions
    predicted_stock_price_dataframe = pd.DataFrame(data=predicted_ohlc_prices,
                                                index=testing_dataframe.index,
                                                columns=testing_dataframe.columns)
    prediction_vs_real(stock_symbol, testing_dataframe, predicted_stock_price_dataframe)

    # Visulaize training data, testing data and predictions together
    plot_full_graph(stock_symbol, testing_dataframe, predicted_stock_price_dataframe, training_dataframe)