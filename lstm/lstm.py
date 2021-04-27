import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

class LSTM:

    def inputStockName(self):
        print('Stock Name:')
        self.stockName = input()

    def initStockData(self):
        data = yf.Ticker(self.stockName)
        historicalData = data.history(period="1y")
        # print(historicalData.columns)
        # Y = historicalData['Close'].to_frame().reset_index()
        # del historicalData['Close']
        # X = historicalData
        return historicalData
    
    def train_test_split(self,X):
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
        n = X.shape[0]
        n_test = 0.30*n
        n_train = n-n_test
        self.x_train = X.iloc[:n_train]
        x_test = X.iloc[n_train:]

        self.x_train = self.x_train.iloc[:,4:5].values
        x_test = x_test.iloc[:,4:5].values

        self.scaler = MinMaxScaler(feature_range = (0, 1))

        training_set_scaled_X= self.scaler.fit_transform(x_train)
        return training_set_scaled_X,x_test

    def create_feature_label_set(self,x):
        features_set = []
        labels = []
        n = len(x) #total rows of training data
        for i in range(60, n):
            features_set.append(x[i-60:i])
            labels.append(x[i, 0])

        features_set, labels = np.array(features_set), np.array(labels)
        features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
        return features_set,labels

    def training_lstm(self,features_set,labels):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units = 1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        model.fit(features_set, labels, epochs = 100, batch_size = 32)
        return model

    def test_data(self,model,x_test):
        train_test_merge = pd.concat(self.x_train,x_test, axis=0)
        total_test_len = len(self.x_train)+len(x_test)
        test_inputs = train_test_merge[len(self.x_train)-len(x_test)-60:].values
        test_inputs = test_inputs.reshape(-1,1)
        test_inputs = self.scaler.transform(test_inputs)
        test_features = []
        for i in range(60, total_test_len):
            test_features.append(test_inputs[i-60:i, 0])

        test_features = np.array(test_features)
        test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
        return test_features

    def predictions(self,test_features):
        predictions = scaler.inverse_transform(predictions)
        plt.figure(figsize=(10,6))
        plt.plot(self.y_test, color='blue', label='Actual Stock Price')
        plt.plot(predictions , color='red', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    lstm = LSTM()
    lstm.inputStockName()
    X = lstm.initStockData()
    x,x_test = lstm.train_test_split(X)
    feature,label = lstm.create_feature_label_set(x)
    model = lstm.training_lstm(feature,label)
    test_features = lstm.test_data(model,x_test)
    lstm.predictions(test_features)
    #test data



