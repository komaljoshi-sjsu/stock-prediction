import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler


class StockPrediction:

    def predictData(self, stock, nodays):
        print(stock)

        start = datetime(2019, 1, 1)
        end = datetime.now()

        # Getting the Historical data
        df = yf.download(stock, start=start, end=end)
        df['prediction'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        #last_row = df.tail(1)
        # print(last_row)

        forecast_time = int(nodays)

        # Scaling the data
        X_unscaled = df.drop(['prediction'], 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_unscaled)

        X = X_scaled
        Y = df['prediction']

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=nodays, random_state=0)

        X_prediction = X[-forecast_time:]
        Y_test = Y[-forecast_time:]

        regressor = RandomForestRegressor(n_estimators=10, random_state=0)
        regressor.fit(X_train, Y_train)

        # df2 = pd.DataFrame({'Actual': Y_test, 'Predicted': prediction})
        # print(df2.head())

        prediction = regressor.predict(X_prediction)

        print('RMSE score: ', np.sqrt(
            metrics.mean_squared_error(Y_test, prediction)))
        r2 = metrics.r2_score(
            Y_test, prediction, sample_weight=None, multioutput='uniform_average')
        print('R-squared score: ', r2)
        adjusted_r_squared = 1 - (1-r2)*(len(Y)-1)/(len(Y)-X.shape[1]-1)
        print('Adjusted R-squared: ', adjusted_r_squared)

        plt.figure(figsize=(12, 6))
        # plt.plot(df['Close'], '-o')
        plt.xlabel('Dates')
        plt.ylabel('Close')
        # plt.savefig('TrainingData')

        sdate = datetime.now() - timedelta(days=nodays)
        edate = datetime.now() - timedelta(days=nodays) + timedelta(days=nodays-1)

        delta = edate - sdate

        datesArray = []
        for i in range(delta.days + 1):
            day = sdate + timedelta(days=i)
            datesArray.append(day)

        # plt.figure(figsize=(10,4))
        plt.plot(datesArray, prediction, '-o')

        plt.plot(datesArray, Y_test, '-o')
        plt.legend(["Predicted Values", "Actual Values"])
        # plt.xlabel('Days in future')
        # plt.ylabel('Predicted Close')
        plt.show()
        # plt.savefig('Predicted VS Actual')

        # if (float(prediction[4]) > (float(last_row['Close'])) + 1):
        # output = ("\n\nStock:" + str(stock) + "\nPrior Close:\n" + str(last_row['Close']) + "\n\nPrediction in 1 Day: " + str(
        #    prediction[0]) + "\nPrediction in " + str(nodays) + " Days: " + str(prediction[nodays-1]))
        # print(output)

        def buy_or_sell(historical_close_avg, next_day_prediction):
            mean_of_historical_close_avg = historical_close_avg.mean()
            if next_day_prediction > mean_of_historical_close_avg:
                return 'buy'
            elif next_day_prediction > mean_of_historical_close_avg:
                return 'sell'
            else:
                return 'hold'

        size = 3
        correct = wrong = total = 0
        for t in range(len(Y_test)):
            yhat = prediction[t]
            obs = Y_test[t]
            status = buy_or_sell(
                df.Close[len(df.Close[0:size])-t-size: len(df.Close[0:size])-t], yhat)
            print('Prediction: Tomorrow (utc:{}) it is more favourable to {}'.format(
                str(Y_test.index[t-size].date()), status))
            real = buy_or_sell(
                df.Close[len(df.Close[0:size])-t-size: len(df.Close[0:size])-t], Y_test[t-size])
            if status != real:
                wrong += 1
            else:
                correct += 1
            total += 1
        print("Correct predictions:", correct)
        print("Wrong predictions:", wrong)


def test() -> None:
    stockPrediction = StockPrediction()

    stockPrediction.predictData('MSFT', 30)


if __name__ == "__main__":
    test()
