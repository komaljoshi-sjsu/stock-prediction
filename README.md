# Stock Price Predictions

**PROBLEM STATEMENT**

This stock prediction project will predict trend of the stock. It will also suggest the user whether it is a good time to buy this particular stock or sell it.

Project encapsulated a comprehensive study of 5 different algorithms and showcases findings from each of these.

Accuracy of the trend can be verified from the real-time data we are using by visualizing in the graph.

**Data Set**

We are using [yfinance](https://pypi.org/project/yfinance/) to take real time data of stocks. This api allows user to get data based on any given time-frame.

**Accessing each algo**

Each algorithm has its own folder which contains Readme explaining steps to run the algorithm.


## Techniques for Stock Prediction

**1. LSTM(Long Short Term Memory) (Komal)**

It is located in **lstm folder**.

**Approach**

LSTM is used to train the model, where we have fit our data with 100 epochs and 32 batch size. We created 5 layers. We also used scaling window at every 3 year data.

**Testing**

Currently, the data is split into training and testing where last 90 days represent testing. 3 months ago is treated as present. Since timestep for our model is 60 days, so our model looks back 60 days to predict tomorrow's future value.

**Tools and Libraries**:

sklearn, keras, matplotlib, pandas, yfinance and numpy.

**2. Auto-ARIMA(‘Auto Regressive Integrated Moving Average’) (Akash)**

It is located in **ARIMA folder**.

**Tools and Libraries**:
sklearn, matplotlib, pandas, yfinance, numpy and statsmodels.

**3. KNN(k-Nearest Neighbours Regression) (Neha Poonia)**

It is located in **KNN_stock_price_prediction folder**.

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

**4. Support Vector Machine (Anagha)**

It is located in **SVM_stock_price_prediction folder**.

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

**5. Random Forest Regression**

It is located in **randomForest folder**.

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math


## Findings:

We used three time series based algorithms as can be seen below:

LSTM
