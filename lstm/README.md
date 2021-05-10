# USING LSTM FOR STOCK TREND ANALYSIS AND ADVISE

## How to execute?

LSTM implementation can either be visualized in jupyter notebook lstm.ipynb or can be run from stock_predictor.py.

**Steps to execute python file**

User is given various options to make any customization. Following command can be executed to see help options available:

**python3 stock_predictor.py --help**

![help](images/file_help.png)

User must enter stock symbol while executing python code: **python3 stock_predictor.py BRK-A**. Here BRK-A is Berkshire Hathaway stock.
Following are steps to run lstm for stock trend and advise for Berkshire Hathaway stock:

1. Execute python3 stock_predictor.py BRK-A
2. User can see graph of training and test data. For BRK-A, it will look something like below:

![help](images/traintest.png)

During this time, in console, user can see a glimpse of ohlc average, training and testing data shape:

![help](images/datavis.png)

3. After closing the graph, user will see that model training starts as follows:

![help](images/train.png)

In case of any warning(as can be seen in image), just ignore those.

4. Once the model has trained, user can now visualize a graph which shows predicted value with respect to actual value with respect to entire graph as follows:

![help](images/pred.png)

Meanwhile in console, user can see tomorrow's prediction, its accuracy and rmse as follows:

![help](images/advise.png)

Once you close first graph, user will see another graph which shows graph visualization of actual test data with respect to predicted outcome as follows:

![help](images/testgraph.png)


## Logic for Buy and Sell Stock explanation:

There are many applications like robinhood which have their own suggestions based on their own development discussions. Upon exploring robinhood's buy and sell mechanism for continuous days, I came to the conclusion that it works on the principle of average comparison. Taking inspiration from that, I devised this algo which takes 30 day average and then compares next day's prediction. If next day is higher than today, then Sell will be advised, if it is less than today then buy will be advised, else Hold will be advised.




