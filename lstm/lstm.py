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

class StockPredictor:

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

    def stock_prediction_from_interpreter(self):
        from datetime import datetime

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import yfinance as yf
        from keras.layers import LSTM, Dense, Dropout
        from keras.models import Sequential
        from sklearn.preprocessing import MinMaxScaler

        stock_symbol = 'BRK-A'
        data = yf.Ticker(stock_symbol)
        historical_data = data.history(period="3y")
        closing_price_data = historical_data.iloc[:, 0:4]

        # Split dataset into training and testing dataset
        # Data older than 60 days is training dataset and remaining is testing dataset
        testing_dataframe = closing_price_data[closing_price_data.index > datetime.utcnow() - pd.to_timedelta('60days')]
        training_dataframe = closing_price_data[closing_price_data.index < testing_dataframe.index[0]]
        print('training set shape:', training_dataframe.shape)
        print('testing set shape:', testing_dataframe.shape)

        training_dataset = training_dataframe.values
        transformer_x = MinMaxScaler(feature_range = (0, 1))
        training_dataset_scaled_x = transformer_x.fit_transform(training_dataset)

        training_dataset_y = training_dataframe.iloc[:, 3:4].values
        transformer_y = MinMaxScaler(feature_range = (0, 1))
        training_dataset_scaled_y = transformer_y.fit_transform(training_dataset_y)

        X_train = []
        y_train = []

        for i in range(60, len(training_dataset_scaled_x)):
            X_train.append(training_dataset_scaled_x[i-60:i, :])
            y_train.append(training_dataset_scaled_y[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)

        stock_prediction_model = Sequential()
        stock_prediction_model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 4)))
        stock_prediction_model.add(Dropout(0.2))
        stock_prediction_model.add(LSTM(units = 50, return_sequences = True))
        stock_prediction_model.add(Dropout(0.2))
        stock_prediction_model.add(LSTM(units = 50, return_sequences = True))
        stock_prediction_model.add(Dropout(0.2))
        stock_prediction_model.add(LSTM(units = 50))
        stock_prediction_model.add(Dropout(0.2))
        stock_prediction_model.add(Dense(units = 1))
        stock_prediction_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # fit the model
        stock_prediction_model.fit(X_train, y_train, epochs = 100, batch_size = 32)

        # save model
        # stock_prediction_model.save('stock_prediction_model')

        real_stock_price = testing_dataframe.iloc[:, 3:4].values


        inputs = closing_price_data[len(closing_price_data) - len(testing_dataframe) - 60:].values
        # inputs = inputs.reshape(-1,1)
        inputs = transformer_x.transform(inputs)

        X_test = []
        for i in range(60, len(inputs)):
            X_test.append(inputs[i-60:i, :])

        X_test = np.array(X_test)

        predicted_stock_price = stock_prediction_model.predict(X_test)
        predicted_stock_price = transformer_y.inverse_transform(predicted_stock_price)

        plt.plot(real_stock_price, color = 'black', label = '{} Stock Price'.format(stock_symbol))
        plt.plot(predicted_stock_price, color = 'green', label = 'Predicted {} Stock Price'.format((stock_symbol)))
        plt.title('{} Stock Price Prediction'.format(stock_symbol))
        plt.xlabel('Time')
        plt.ylabel('{} Stock Price'.format(stock_symbol))
        plt.legend()
        plt.show()
        plt.savefig('graph.png')

if __name__ == "__main__":
    lstm = StockPredictor()
    # lstm.inputStockName()
    # X = lstm.initStockData()
    # x,x_test = lstm.train_test_split(X)
    # feature,label = lstm.create_feature_label_set(x)
    # model = lstm.training_lstm(feature,label)
    # test_features = lstm.test_data(model,x_test)
    # lstm.predictions(test_features)
    #test data

    lstm.stock_prediction_from_interpreter()



