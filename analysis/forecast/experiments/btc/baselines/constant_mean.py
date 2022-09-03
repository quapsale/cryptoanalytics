"""
File: constant_mean.py
Description: Experiments with Constant Mean baseline model.
File Created: 06/04/2022
Python Version: 3.9
"""

# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import csv

# File Properties
data_path = './data/datasets/binance.csv'
baseline_path = './analysis/forecast/experiments/btc/baseline/'
predictions_path = './analysis/forecast/experiments/btc/predictions/'
metrics_path = './analysis/forecast/experiments/btc/metrics.csv'
test_start = '2022-05-11 15:57:00'
np.random.seed(123)

# Load data
data = pd.read_csv(data_path, sep='\t', index_col='Date')

# Split train/test (ratio=90/10)
train = data.loc[:test_start].copy()
test = data.loc[test_start:].copy()
test = test.iloc[1:]

# Create series objects
y_train = train['BTC']
y_test = test['BTC']

# Constant mean
pred = np.mean(y_train)
predictions = [pred]*len(y_test)
np.savetxt(predictions_path + 'constant_mean.txt', predictions)

# Compute and store metrics
now = datetime.datetime.now()
mse = mean_squared_error(y_test, predictions, squared=True)
rmse = mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
with open(metrics_path, 'a') as metrics:
    writer = csv.writer(metrics)
    writer.writerow([now, 'constant_mean', '1', mse, rmse, mae, mape])
