## Stock Price Prediction using Support Vector Machine - Regression (SVR)

The SVM method used in time series is called **Support Vector Regression (SVR)**. In SVR method, one of the most important things to improve the accuracy of forecasting is input selection.
This study uses a machine learning technique called [Support Vector Regression(SVR)](https://towardsdatascience.com/walking-through-support-vector-regression-and-lstms-with-stock-price-prediction-45e11b620650) to predict stock prices for large and small capitalisations.

**Package Installation**

`$ git clone https://github.com/komaljoshi-sjsu/stock-prediction`

**Navigate to Files Containing SVM Implementation**

`$ cd SVM_stock_price_prediction/`

**Install requiremets**

`$ pip install -r requirements.txt`

**Run the python code**

`$ python3 SVR.py <stock_symbol>| tee Output_SVM.txt`

example : `$ python3 SVR.py MSFT| tee Output_SVM.txt`
