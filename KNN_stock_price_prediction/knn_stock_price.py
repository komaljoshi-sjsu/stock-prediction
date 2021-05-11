# -*- coding: utf-8 -*-
"""KNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nYdlpNsdTlEQacMNaqcsmEOKcCuqIUvd
"""

import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib.pyplot as plt
# %matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,5
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 10))
from datetime import datetime , timedelta

from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt

import re
class KNN:
  def inputStockName(self):
    print('Stock Name:')
    self.stockName = input()
  def initStockData(self):
        print(" Welcome to Stock predictor")
        start_date=input("Enter the starting date :(format -->  \"yyyy-mm-dd\" ) ")
        date_pattern = re.compile("[0-2][0-9][0-9][0-9]-[0-1][0-9]-[0-3][0-9]")
        while(len(date_pattern.findall(start_date)) == 0):
          start_date=input("Enter the starting date in correct form :(format -->  \"yyyy-mm-dd\" )")
        start_date=datetime.strptime(start_date,"%Y-%m-%d")
        end_date = start_date + timedelta(days=3*365) # 3 years from start date 
        df1 = yf.download(self.stockName, start=start_date, end=end_date )
        df1=df1.dropna()
        return df1
  def train_test_split(self,df1):
    df1['ohlc']=df1[['Open', "High","Low","Close"]].mean(axis=1)
    df1=df1[['Open', "High","Low","Volume","Adj Close","Close",'ohlc']]
    df_volume=df1[['Volume']]
    df1=df1.drop(['Volume'],axis=1)
    X=df1[['ohlc']]
    Y=df1[['Close']]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)
    print(X_train.shape , Y_train.shape)
    date_train = []
    date_test = []
    for row in X_train.index:
      date_train.append(row)
    for row in X_test.index:
      date_test.append(row)
    return X_train, X_test, Y_train, Y_test
    
  def training_knn(self,X_train, X_test, Y_train, Y_test):
    grid_params={
    'n_neighbors':[i for i in range(1,30)],
    'weights':['uniform','distance'],
    'metric':['euclidean', 'manhattan']}
    gs= GridSearchCV(neighbors.KNeighborsRegressor(),grid_params , verbose = 1 , cv =3 , n_jobs = -1 )
    gs_v = gs.fit(X_train,Y_train)
    print(gs_v.best_score_)
    print(gs_v.best_params_)
    K=int(gs_v.best_params_['n_neighbors'])
    gs_results = neighbors.KNeighborsRegressor(n_neighbors=K , weights='uniform', algorithm='auto')
    gs_results.fit(X_train, Y_train)
    return gs_results
    
  def testing(self,gs_results,X_test,Y_test):
    Y_pred=gs_results.predict(X_test)
    error = sqrt(mean_squared_error(Y_test,Y_pred))
    gs_results.score(X_test,Y_test)
    print("The rmse error is",error)
    return Y_pred
  
  def buy_sell_hold(self,X_train, X_test, Y_train, Y_test,Y_pred):
    date_train = []
    date_test = []
    for row in X_train.index:
        date_train.append(row)
    for row in X_test.index:
        date_test.append(row)
    avg_close = Y_train["Close"].mean()
    print("Average share price of {} - {}" .format(self.stockName,avg_close))
    results=[]
    for i in range (1,len(Y_pred)):
      if Y_pred[i]>Y_pred[i-1] and Y_pred[i-1]>avg_close:
        #the price is increasing so profit is there in buying
        ans="For date {}".format(date_test[i])
        ans=ans[0:-8]
        ans2="suggestion is to buy "
        results.append(ans+ans2)
        print(ans+ans2)
      elif Y_pred[i]<Y_pred[i-1] and Y_pred[i-1]<=avg_close:
        #the price is decreasing and is below average so sell asap 
        ans="For date {}".format(date_test[i])
        ans=ans[0:-8]
        ans2="suggestion is to sell "
        results.append(ans+ans2)
        print(ans+ans2)
      elif Y_pred[i]<Y_pred[i-1] and Y_pred[i-1]>avg_close:
      #the price is decreasing but still above average so wait for it to rise
        ans="For date {}".format(date_test[i])
        ans=ans[0:-8]
        ans2="suggestion is to hold "
        results.append(ans+ans2)
        print(ans+ans2)
      elif Y_pred[i]>Y_pred[i-1] and Y_pred[i-1]<=avg_close:
        #the price is increasing and is below average hence will be benficial to buy
        ans="For date {}".format(date_test[i])
        ans=ans[0:-8]
        ans2="suggestion is to buy"
        results.append(ans+ans2)
        print(ans+ans2)
      elif Y_pred[i]==Y_pred[i-1]:
        #no change in market position so better to hold
        ans="For date {}".format(date_test[i])
        ans=ans[0:-8]
        ans2="suggestion is to hold "
        results.append(ans+ans2)
        print(ans+ans2)

  def predictions(self,Y_test,Y_train,Y_pred):
    Y_test_list = np.array(Y_test.values.tolist()).flatten()
    Y_train_list = np.array(Y_train.values.tolist()).flatten()
    Y_pred=Y_pred.flatten()
    x_ax=range(len(Y_test_list)+len(Y_train_list))
    plt.plot(x_ax[len(Y_train_list):len(Y_test_list)+len(Y_train_list)] , Y_pred[:])
    plt.plot(x_ax[len(Y_train_list):len(Y_test_list)+len(Y_train_list)],Y_test_list[:])
    plt.plot(x_ax[:len(Y_train_list)],Y_train_list[:])
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    #plt.xticks(date_test)
    plt.legend(["predicted", "test_value","train_value"], loc ="lower right")
    plt.show()
    #graph2
    plt.plot(Y_pred)
    plt.plot(Y_test_list)
    plt.xlabel('Days')
    plt.ylabel('Stock Price')
    #plt.xticks(date_test)
    plt.legend(["predicted", "real_value"], loc ="lower right")
    plt.show()
    return Y_test_list,Y_train_list,Y_pred

  def accuracy(self,Y_test_list,Y_train_list,Y_pred):
    error_limit = 10.0
    average_difference=0
    correct=0
    wrong=0
    for i in range(len(Y_pred)):
      if abs(Y_pred[i]-Y_test_list[i])<=error_limit :
        correct+=1
      else:
        wrong+=1 
    average_difference+=abs(Y_pred[i]-Y_test_list[i])
    print("Number of correct predictions with error limit {} is {}".format(error_limit, correct))
    print("Number of incorrect predictions with error limit {} is {}".format(error_limit, wrong))
    print("Average deviation for predictions {}".format(average_difference/len(Y_pred)))
    #print(correct , wrong , type(average_difference))
    accuracy=(correct/(correct+wrong))*100
    print("Accuracy for predictions is", accuracy)
        
def test():
  knn = KNN()
  knn.inputStockName()
  df1=knn.initStockData()

  # Split the data into training and testing data
  X_train, X_test, Y_train, Y_test = knn.train_test_split(df1)

  # Create and fit the model
  gs_results=knn.training_knn(X_train, X_test, Y_train, Y_test)

  # Predict on test set
  Y_pred=knn.testing(gs_results,X_test,Y_test)

  # Buy sell logic
  knn.buy_sell_hold(X_train, X_test, Y_train, Y_test,Y_pred)

  # Visulaize predictions
  Y_test_list,Y_train_list,Y_pred = knn.predictions(Y_test,Y_train,Y_pred)

  # Calculate accuracy
  knn.accuracy(Y_test_list,Y_train_list,Y_pred)
if __name__ == "__main__":
  test()
