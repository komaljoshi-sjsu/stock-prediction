import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LSTM:

    def inputStockName(self):
        print('Stock Name:')
        self.stockName = input()

    def initStockData(self):
        data = yf.Ticker(self.stockName)
        historicalData = data.history(period="5y")
        print(historicalData.columns)
        Y = historicalData['Close'].to_frame().reset_index()
        del historicalData['Close']
        X = historicalData
        return X,Y
    
    def train_test_split(self,X,Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
        mms = MinMaxScaler(feature_range = (0, 1))
        training_set_scaled_X= mms.fit_transform(self.X_train)
        training_set_scaled_Y= mms.fit_transform(self.y_train)
        return training_set_scaled_X,training_set_scaled_Y

    def create_feature_label_set(self,x,y):
        features_set = []
        labels = []
        n = x.shape[0] #total rows of training data
        for i in range(60, n):
            features_set.append(x[i-60:i, 0])
            labels.append(y[i, 0])

if __name__ == "__main__":
    lstm = LSTM()
    lstm.inputStockName()
    X,Y = lstm.initStockData()
    x,y = lstm.train_test_split(X,Y)


