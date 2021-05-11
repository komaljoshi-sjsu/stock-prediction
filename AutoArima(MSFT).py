from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from pmdarima.arima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot
from datetime import timedelta
import datetime
import re
from math import sqrt
from matplotlib import pyplot
import pickle

class AutoArima:
    def loadData(self, x):
        print(x)
        data = yf.Ticker(x)
        self.df = data.history(start="2018-05-10", end="2021-05-09")
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.xlabel('Dates')
        plt.ylabel('Close Prices')
        plt.plot(self.df.Close)
        plt.title('Microsoft Inc. closing price')
        plt.show()
        plt.savefig("Stock_Close.png")
        # return self.df

    def test_stationarity(self):
        self.df_close = self.df.Close
        # Determing rolling statistics
        rolmean = self.df_close.rolling(12).mean()
        rolstd = self.df_close.rolling(12).std()
        # Plot rolling statistics:
        plt.plot(self.df_close, color='blue', label='Original')
        plt.plot(rolmean, color='red', label='Rolling Mean')
        plt.plot(rolstd, color='black', label='Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean and Standard Deviation')
        plt.savefig("StationarityPlot.png")

        print("Results of dickey fuller test")
        adft = adfuller(self.df_close, autolag='AIC')
        # output for dft will give us without defining what the values are.
        # hence we manually write what values does it explains using a for loop
        print(adft[1])
        output = pd.Series(adft[0:4], index=[
            'Test Statistics', 'p-value', 'No. of lags used', 'Number of observations used'])
        for key, values in adft[4].items():
            output['critical value (%s)' % key] = values
        print(output)
        return adft[1]
    
    def seasonaldecompose(self):
        result = seasonal_decompose(
            self.df_close, model='multiplicative', period=30)
        fig = plt.figure()
        fig = result.plot()
        fig.set_size_inches(16, 9)
        fig.savefig("seasonal_decompose.png")
        
    def autoArima(self):
        # Auto Arima model to find out coeff p,d,q for Arima
        model = pm.auto_arima(self.df.Close, start_p=1, start_q=1,
                              test='adf',       # use adftest to find optimal 'd'
                              max_p=3, max_q=3,  # maximum p and q
                              m=1,              # frequency of series
                              d=None,           # let model determine 'd'
                              seasonal=False,   # No Seasonality
                              start_P=0,
                              D=0,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)

        print(model.summary())
        model.plot_diagnostics(figsize=(15,8))
        plt.savefig('Auto_Arima_residual_plot.png')
        return model
    
    #Suggestion function
    def what_to_do(self,historical_close_avg, next_day_prediction):
        mean_of_historical_close_avg = historical_close_avg.mean()
        if next_day_prediction > mean_of_historical_close_avg:
            return 'buy'
        elif next_day_prediction < mean_of_historical_close_avg:
            return 'sell'
        else:
            return 'hold'

    def walkForward(self):
        size = int(len(self.df.Close))-60
        train, test = self.df.Close[0:size], self.df.Close[size:len(self.df.Close)]
        history = [x for x in train]
        predictions = list()
        correct = wrong = total = 0
        # walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(1,0,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0][0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            prediction = self.what_to_do(self.df.Close[len(self.df.Close[0:size])+t-60:len(self.df.Close[0:size])+t],yhat)
            print('Prediction: Tomorrow (utc:{}) it is more favourable to {}'.format(str(test.index[t-60].date()), prediction))
            real = self.what_to_do(self.df.Close[len(self.df.Close[0:size]) + t - 60:len(self.df.Close[0:size]) + t], test[t-60])
            if prediction != real:
                wrong += 1
            else:
                correct += 1
            total += 1
        print("Correct predictions:", correct)
        print("Wrong predictions:", wrong)
        series = pd.Series(predictions,index=test.index)
        error = mean_squared_error(test, predictions)
        pyplot.figure(figsize=(10,6))
        
        #Plot Graph
        pyplot.title("Prediction of CLOSE values of Microsoft on history")
        pyplot.plot(test,color='blue')
        pyplot.plot(series ,color='red')
        pyplot.savefig('ARIMA_Predicted_vs_Actual.png')


        #Performance Report
        mse = mean_squared_error(test, series)
        print('MSE: '+str(mse))
        mae = mean_absolute_error(test, series)
        print('MAE: '+str(mae))
        rmse = math.sqrt(mean_squared_error(test, series))
        print('RMSE: '+str(rmse))
        r2 = r2_score(test, series, sample_weight=None, multioutput='uniform_average')
        print('R-squared score: ', r2)
        return train,test
        
    def forecastAutoArima(self, model):
        # Forecast
        n_periods = 30
        fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
        index_of_fc = np.arange(
            len(self.df.Close), len(self.df.Close)+n_periods)
        fc_series = pd.Series(fc, index=index_of_fc)

        pred_values = []
        indexes = []
        current_date = self.df.index[-1]

        # Transaforming Data to Dates
        for index, value in fc_series.items():
            current_date = current_date + timedelta(days=1)
            indexes.append(current_date)
            pred_values.append(value)

            # make series for plotting purpose
        fc_series = pd.Series(fc, index=index_of_fc)
        lower_series = pd.Series(confint[:, 0], index=indexes)
        upper_series = pd.Series(confint[:, 1], index=indexes)

        print(f"Todays Value :")
        print(self.df.Close.tail(1))
        print(f"Predicted values for next {n_periods} days")
        new_fc_series = pd.Series(pred_values, index=indexes)
        print(new_fc_series)

        pyplot.figure(figsize=(10, 6))
        pyplot.title("Prediction of CLOSE values of Mircosoft on history")
        pyplot.plot(self.df.Close, color='blue')
        pyplot.legend(loc='best')
        pyplot.plot(new_fc_series, color='red')
        pyplot.savefig("ARIMA_forecast.png")
        pyplot.show()
        # Plot
        plt.plot(self.df.Close)
        plt.plot(new_fc_series, color='yellow')
        plt.fill_between(lower_series.index,
                         lower_series,
                         upper_series,
                         color='red', alpha=.15)

        plt.title("Final Forecast of Microsoft USING AUTO-ARIMA")
        plt.savefig('forecast.png')

    def seasonalArima(self):
        smodel = pm.auto_arima(self.df.Close, start_p=1, start_q=1,
                               test='adf',
                               max_p=3, max_q=3, m=24,
                               start_P=0, seasonal=True,
                               d=None, D=1, trace=True,
                               error_action='ignore',
                               suppress_warnings=True,
                               stepwise=True)

        smodel.summary()
        filename = 'S_model.sav'
        pickle.dump(smodel, open(filename, 'wb'))
        return smodel

    def forecastSeasonalArima(self):
        filename = 'S_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        size = int(len(self.df.Close))-60
        train, test = self.df.Close[0:size], self.df.Close[size:len(self.df.Close)]
        history = [x for x in train]
        predictions = list()
        correct = wrong = total = 0
        # walk-forward validation
        for t in range(len(test)):
            model = ARIMA(history, order=(2,0,0))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0][0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)
            prediction = self.what_to_do(self.df.Close[len(self.df.Close[0:size])+t-60:len(self.df.Close[0:size])+t],yhat)
            print('Prediction: Tomorrow (utc:{}) it is more favourable to {}'.format(str(test.index[t-60].date()), prediction))
            real = self.what_to_do(self.df.Close[len(self.df.Close[0:size]) + t - 60:len(self.df.Close[0:size]) + t], test[t-60])
            if prediction != real:
                wrong += 1
            else:
                correct += 1
            total += 1
        print("Correct predictions:", correct)
        print("Wrong predictions:", wrong)

        series = pd.Series(predictions,index=test.index)
        error = mean_squared_error(test, predictions)
        print(error)
        
        #Plot
        pyplot.figure(figsize=(10,6))
        pyplot.title("Prediction of CLOSE values of MSFT on history")
        pyplot.plot(test,color='blue')
        pyplot.plot(series ,color='red')
        pyplot.savefig("S-Arima-Result.png")


        #Performance Report
        mse = mean_squared_error(test, series)
        print('MSE: '+str(mse))
        mae = mean_absolute_error(test, series)
        print('MAE: '+str(mae))
        rmse = math.sqrt(mean_squared_error(test, series))
        print('RMSE: '+str(rmse))
        r2 = r2_score(test, series, sample_weight=None, multioutput='uniform_average')
        print('R-squared score: ', r2)


def start() -> None:
    autoArima = AutoArima()
    val = input("Enter your value: ")
    autoArima.loadData(val)
    p = autoArima.test_stationarity()
    autoArima.seasonaldecompose()
    model = autoArima.autoArima()
    autoArima.forecastAutoArima(model)
    train,test = autoArima.walkForward()
    smodel = autoArima.seasonalArima()
    autoArima.forecastSeasonalArima()


if __name__ == "__main__":
    # execute only if run as a script
    start()
