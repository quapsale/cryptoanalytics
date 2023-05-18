"""
File: constant_mean.py
Description: Experiments with Constant Mean baseline model.
File Created: 01/02/2023
Python Version: 3.9
"""

# Imports
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import csv

# File Properties
root = sys.path[1]
os.chdir(root)
data_path = 'forecast/data/binance.csv'
baseline_path = 'forecast/experiments/btc/baseline/'
predictions_path = 'forecast/experiments/btc/predictions/'
metrics_path = 'forecast/experiments/btc/metrics.csv'
test_start = '2022-11-05 13:02:00'
np.random.seed(123)

# Load data
data = pd.read_csv(data_path, sep=',', index_col='Date')

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
