import yfinance as yf
import pandas as pd
class LSTM:

    def inputStockName(self):
        print('Stock Name:')
        self.stockName = input()

    def initStockData(self):
        data = yf.Ticker(self.stockName)
        historicalData = data.history(period="1d")
        self.Y = historicalData['Close']
        del historicalData['Close']
        print(historicalData.columns)
        self.X = historicalData
    
    def train_test_split(self):
        pass

if __name__ == "__main__":
    lstm = LSTM()
    lstm.inputStockName()
    lstm.initStockData()


