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

**Approach**

ARIMA (Auto Regressive Integrated Moving Average)is used to explain given time series based on past values,lagged values and future lagged forecast errors so that equation can be used to predict future values. In this,the model is trained on three different coefficients (P,Q,D) known for AR,MA and I in ARIMA respectively.

**Testing**

Currently, the data is split into training and testing where last 90 days represent testing. 3 months ago is treated as present. Since timestep for our model is 60 days, so our model looks back 60 days to predict tomorrow's future value.

**Tools and Libraries**:
sklearn, matplotlib, pandas, yfinance, numpy and statsmodels.

**3. KNN(k-Nearest Neighbours Regression) (Neha Poonia)**

It is located in **KNN_stock_price_prediction folder**.

**Approach**

In this study K-nearest neighbors (KNN) algorithm is used to predict the stock price. Here we have taken ohcl as average of Open,High,Low and Close values and used it as feature value to predict the stock price for next day.

K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems .
KNN follows two properties :

Lazy learning algorithm − KNN is a lazy learning algorithm because it does not have a specialized training phase and uses all the data for training while classification.

Non-parametric learning algorithm − KNN is also a non-parametric learning algorithm because it doesn’t assume anything about the underlying data .

**Testing**

Currently, the data is split into training and testing , total data is taken for 3 years where last 10 percent represent testing, so our model looks back to predict tomorrow's future value.

**Tools and Libraries**:
sklearn, matplotlib, pandas, yfinance, numpy and math

**4. Support Vector Machine (Anagha)**

It is located in **SVM_stock_price_prediction folder**.

**Approach**

In this study, support vector regression (SVR) analysis is used as a machine learning technique in order to predict the stock market price as well as to predict stock market trend.Here we calculated the mean of Open,High,Low and Close(ohlc) of each day and is used as input to train the model.Our predicting price is average ohlc of the next day.

**Testing**

Currently, the data is split into training and testing where last 90 days represent testing. 3 months ago is treated as present. Since timestep for our model is 60 days, so our model looks back 60 days to predict tomorrow's future value.

**Tools and Libraries**:
sklearn, matplotlib, pandas, yfinance, numpy and math

**5. Random Forest Regression**

It is located in **randomForest folder**.

**Approach**

Random Forest regressor runs the regression on multiple estimators and selects the best estimation based on score(MSE/MAE).

**Testing**

Currently, the data is split into training and testing where last 30 days represent testing. 1 months ago is treated as present.

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

## Comparison and Findings:

We used three time series based algorithms, comparison with Microsoft can be seen below:

| Algorithm | Accuracy | RMSE            |
| --------- | -------- | --------------- |
| LSTM      | 77%      | 0.05            |
| ARIMA     | >95%     | 12.61(unscaled) |
| SVM       | 80%      | 0.04            |

For stock like Microsoft with erratic trends, Arima outperformed SVM and LSTM. It detected stock trend better whereas SVM and LSTM fared better in terms of value prediction. On a 3 year data, SVM slightly outperformed LSTM, whereas on data with max values, LSTM's accuracy is usually lying above 80%.

For stocks with stable growth, like value investing stocks, LSTM is giving constant better results with accuracy ranging from 91-100%.

For non time series based algorithms, future predictions on large range of test values were not accurate. For KNN and random forest regressor, 30 day testing data was used and both algorithms performed almost in similar fashion.

| Algorithm     | Accuracy |
| ------------- | -------- |
| KNN           | 76%      |
| Random Forest | 70%      |

## Future Scope

Time series based algorithms like lstm, arima and svm can be used together to create an ensemble and predict better results. Buying and selling of stocks can be modified to include buy and sell indicators across the years and then carry out comparison for better testing.
