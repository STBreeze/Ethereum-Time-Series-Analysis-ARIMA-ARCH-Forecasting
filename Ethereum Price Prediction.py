import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
import os

eth_data = yf.download(tickers='ETH-USD', period='max', interval='1d')
eth_data.sort_index(inplace=True)
eth_data.columns = eth_data.columns.get_level_values(0)
eth_data.columns.name = None

print(f'Empty Cells: {eth_data.isna().sum()}')
print(f'Null Values: {eth_data.isnull().sum()}')
print('\nStatistical Description:\n')
print(eth_data.describe())

today = pd.Timestamp.today().normalize()

eth_prices = eth_data['Close']
train_prices = eth_prices[eth_prices.index < today]

my_order = (0,1,0)
arima = ARIMA(train_prices, order=my_order).fit()
forecast_price = arima.forecast(steps=1).iloc[0]

train_residuals = (train_prices - arima.fittedvalues).dropna()

arch = arch_model(train_residuals, p=5, q=0).fit(disp='off')
volatility = np.sqrt(arch.forecast(horizon=1).variance.iloc[-1,0])

forecast_log = pd.DataFrame({'Date': [today],
                             'Close Price': [forecast_price],
                             'Volatility': [volatility]})


log_file = 'Ethereum Forecast Log.csv'

if os.path.exists(log_file):
    existing_log = pd.read_csv(log_file, index_col='Date', parse_dates=True)
    
    if today in existing_log.index:
        print("Today's forecast already exists.")
    else:
        forecast_log.to_csv(log_file, mode='a', header=False)
        print("Forecast appended to log.")
else:
    forecast_log.to_csv(log_file, mode='w', header=True)
    print("Forecast log created.")




