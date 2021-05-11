# importing Libraries
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
import sys

# Finding mean ohlc ( mean of open,higl,low,close )
def get_ohlc_avg(symbol: str, period="5y"):

    data = yf.Ticker(symbol)
    historical_data = data.history(period=period)
    ohlc_avg = pd.DataFrame(
        historical_data.iloc[:, 0:4].mean(axis=1), columns=["OHLC_avg"]
    )
    return ohlc_avg


# Spliting test and train data ,Here we take a total of 90 days but since some data were missing
# the dataframe contains nearly data of 60days
def split_data(data: pd.DataFrame, split_parameter="90days"):
    testing_dataframe = ohlc_avg[
        ohlc_avg.index > datetime.utcnow() - pd.to_timedelta("90days")
    ]
    training_dataframe = ohlc_avg[ohlc_avg.index < testing_dataframe.index[0]]
    print("training dataframe:", training_dataframe.shape)
    print("testing dataframe:", testing_dataframe.shape)
    return training_dataframe, testing_dataframe


# Plotting test and training data of the given Stock
def plot_training_testing_distribution(symbol, training_df, testing_df):
    plt.plot(testing_df, color="red", label="Testing dataset")
    plt.plot(training_df, color="blue", label="Training dataset")
    plt.title("{} Stock Dataset".format(symbol))
    plt.xlabel("Time")
    plt.ylabel("{} Stock Price".format(symbol))
    plt.legend()
    plt.xticks(rotation="vertical")
    plt.show()


# Scaling the dataset values as well creating X_train,y_train for training the model
def get_X_y_values(training_dataframe):

    training_dataset = training_dataframe.values
    transformer = MinMaxScaler(feature_range=(0, 1))
    training_dataset_scaled_x = transformer.fit_transform(training_dataset)

    training_dataset_scaled_y = training_dataset_scaled_x

    X_train = []
    y_train = []

    for i in range(60, len(training_dataset_scaled_x)):
        X_train.append(training_dataset_scaled_x[i - 60 : i, :])
        y_train.append(training_dataset_scaled_y[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], -1)

    return X_train, y_train, transformer


# Training and fitting model using train data sets
def SVR_predict(X_train_data, y_train_data):
    stock_prediction_model = SVR(kernel="linear", C=1e3, gamma=0.1)
    stock_prediction_model.fit(X_train, y_train)

    return stock_prediction_model


# Predicting Buy/Sell based on predicted stock price
def find_buy_sell(historical_ohlc_avg, next_day_prediction):
    mean_of_historical_ohlc_avg = historical_ohlc_avg.mean()[0]
    if next_day_prediction < mean_of_historical_ohlc_avg:
        return "buy"
    elif next_day_prediction > mean_of_historical_ohlc_avg:
        return "sell"
    else:
        return "hold"


# Stock price prediction using Support Vector Regression on test data
def predict_stock_prices(ohlc_avg, testing_dataframe_data, model, transformer):

    real_stock_price = testing_dataframe.iloc[:, 0:1].values
    predicted_stock_price = []
    look_ahead_predicted_stock_price = []
    predicted_stock_price_scaled = []

    inputs = ohlc_avg[len(ohlc_avg) - len(testing_dataframe) - 60 :].values
    inputs = transformer.transform(inputs)
    real_stock_price_scaled = transformer.transform(real_stock_price)
    correct = wrong = total = 0

    for i in range(60, len(inputs)):
        next_day_input = inputs[i - 60 : i, :]
        next_day_input = next_day_input.reshape(1, 60, 1)
        next_day_input = next_day_input.reshape(next_day_input.shape[0], -1)
        next_day_predicted_stock_price = model.predict(next_day_input)
        predicted_stock_price_scaled.append(next_day_predicted_stock_price)
        next_day_predicted_stock_price = transformer.inverse_transform(
            next_day_predicted_stock_price.reshape(-1, 1)
        )
        predicted_stock_price.append(next_day_predicted_stock_price[0][0])
        prediction = find_buy_sell(
            ohlc_avg[
                len(training_dataframe) + i - 60 - 60 : len(training_dataframe) + i - 60
            ],
            next_day_predicted_stock_price[0][0],
        )
        print(
            "Prediction: Tomorrow (utc:{}) it is more favourable to {}".format(
                str(testing_dataframe.index[i - 60].date()), prediction
            )
        )
        real = find_buy_sell(
            ohlc_avg[
                len(training_dataframe) + i - 60 - 60 : len(training_dataframe) + i - 60
            ],
            real_stock_price[i - 60],
        )
        if prediction != real:
            wrong += 1
        else:
            correct += 1
        total += 1

    print("Correct predictions:", correct)
    print("Wrong predictions:", wrong)

    print(
        "Percentage of correct predictions: {}%\n".format(
            round(correct * 100 / total, 2)
        )
    )
    # RMSE value without scaling the stock price
    mse = mean_squared_error(real_stock_price, predicted_stock_price)
    rmse = np.sqrt(mse)
    print("RMSE value without scaling : ", rmse)

    # RMSE value from scaled stock price
    mse_scaled = mean_squared_error(
        real_stock_price_scaled, predicted_stock_price_scaled
    )
    rmse_scaled = np.sqrt(mse_scaled)
    print("RMSE value with scaling : ", rmse_scaled)

    return predicted_stock_price


# Plotting graphs to compare between Real and Predicted stock values
def prediction_vs_real(symbol: str, real_values, predicted_values):

    plt.figure(figsize=(18, 7))
    plt.plot(real_values, color="black", label="{} Stock Price".format(stock_symbol))
    plt.plot(
        predicted_values,
        color="blue",
        label="Predicted {} Stock Price".format((stock_symbol)),
    )
    plt.xticks(predicted_values.index.to_pydatetime())
    plt.title("{} Stock Price Prediction".format(stock_symbol))
    plt.xlabel("Time")
    plt.ylabel("{} Stock Price".format(stock_symbol))
    plt.legend()
    plt.xticks(rotation="vertical")
    plt.savefig("prediction_vs_real.png")


# Plotting graphs between different set of stock values
def plot_full_graph(symbol: str, real_values, predicted_values, train_dataframe):
    plt.figure(figsize=(18, 7))
    plt.plot(
        real_values, color="black", label="{} Real Stock Price".format(stock_symbol)
    )
    plt.plot(
        predicted_values,
        color="blue",
        label="Predicted {} Stock Price".format((stock_symbol)),
    )
    plt.plot(
        train_dataframe,
        color="red",
        label="{} Trained Stock Price".format(stock_symbol),
    )

    plt.title("{} Stock Price Prediction".format(stock_symbol))
    plt.xlabel("Time")
    plt.ylabel("{} Stock Price".format(stock_symbol))
    plt.legend()
    plt.xticks(rotation="vertical")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stock predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stock_symbol", type=str, help="Stock symbol")
    parser.add_argument(
        "--dataperiod",
        type=str,
        default="5y",
        help="Period of data to get from yfinance. \
        Valid periods are: “1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”",
    )
    parser.add_argument(
        "--split_parameter",
        type=str,
        default="90days",
        help="Split parameter for trainig and testing data split. \
        Follows pandas.to_timedelta input format.",
    )
    args = parser.parse_args()

    stock_symbol = args.stock_symbol
    dataperiod = args.dataperiod
    split_parameter = args.split_parameter
    ohlc_avg = get_ohlc_avg(stock_symbol, dataperiod)
    if ohlc_avg.empty:
        print("Empty data :", stock_symbol)
        sys.exit(0)

    print(ohlc_avg.shape)
    print(ohlc_avg.head())

    training_dataframe, testing_dataframe = split_data(
        ohlc_avg, split_parameter=split_parameter
    )

    plot_training_testing_distribution(
        stock_symbol, training_dataframe, testing_dataframe
    )
    X_train, y_train, transformer = get_X_y_values(training_dataframe)

    stock_prediction_model = SVR_predict(X_train, y_train)

    predicted_ohlc_prices = predict_stock_prices(
        ohlc_avg, testing_dataframe, stock_prediction_model, transformer
    )
    predicted_stock_price_dataframe = pd.DataFrame(
        data=predicted_ohlc_prices,
        index=testing_dataframe.index,
        columns=testing_dataframe.columns,
    )
    prediction_vs_real(stock_symbol, testing_dataframe, predicted_stock_price_dataframe)

    plot_full_graph(
        stock_symbol,
        testing_dataframe,
        predicted_stock_price_dataframe,
        training_dataframe,
    )
