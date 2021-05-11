# Stock Price Predictions

**PROBLEM STATEMENT**

This stock prediction application will predict trend of the stock. It will also suggest the user whether it is a good time to buy this particular stock or sell it. 

Accuracy of the trend can be verified from the real-time data we are using by visualizing in the graph.

**Data Set**

We are using [yfinance](https://pypi.org/project/yfinance/) to take real time data of stocks. This api allows user to get data based on any given time-frame.

## Techniques for Stock Prediction

**1. LSTM(Long Short Term Memory) (Komal)**

Following link is used to understand about LSTM:

[LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

**Approach**

LSTM is used to train the model, where we have fit our data with 100 epochs and 32 batch size. We created 5 layers.

![picture](images/lstm_timestep_based_pred.png)

**Testing**

Currently, the data is split into training and testing where last 90 days represent testing. 3 months ago is treated as presetn. Since timestep for our model is 60 days, so our model looks back 60 days to predict tomorrow's future value.

**Tools and Libraries**:

sklearn, keras, matplotlib, pandas, yfinance and numpy.

**2. Auto-ARIMA(‘Auto Regressive Integrated Moving Average’) (Akash)**

Following link is used to understand about Auto-ARIMA:

[AUTO-ARIMA](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/#:~:text=ARIMA%2C%20short%20for%20'AutoRegressive%20Integrated,to%20predict%20the%20future%20values)

![picture](images/Auto_ARIMA_pred.png)
**Tools and Libraries**:
sklearn, matplotlib, pandas, yfinance, numpy and statsmodels.

**3. KNN(k-Nearest Neighbours Regression) (Neha Poonia)**

Following link is used to understand about KNN:

[KNN](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

**4. Support Vector Machine (Anagha)** 

[PREDICTION USING SVM](https://www.sciencedirect.com/science/article/pii/S2405918818300060)
![picture](images/SVM.png)

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

  

**5. Random Forest Regression**

Comparison between previos 30 days' values as predicted by the model with the actual values.

![picture](images/Random_Forest.png)

[Random Forest Regression](https://medium.com/swlh/random-forest-and-its-implementation-71824ced454f)

**Tools and Libraries**:

sklearn, matplotlib, pandas, yfinance, numpy and math

