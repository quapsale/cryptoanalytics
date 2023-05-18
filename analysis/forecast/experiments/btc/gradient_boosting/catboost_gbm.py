"""
File: catboosty.py
Description: Experiments with CatBoost model.
File Created: 01/02/2023
Python Version: 3.9
"""

# Imports
import os
import sys
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import csv

# File Properties
root = sys.path[1]
os.chdir(root)
data_path = 'forecast/data/binance.csv'
gb_path = 'forecast/experiments/btc/gradient_boosting/'
predictions_path = 'forecast/experiments/btc/predictions/'
pretrained_path = 'forecast/experiments/btc/pretrained/'
logs_path = 'forecast/experiments/btc/loggers/catboost/metrics.csv'
metrics_path = 'forecast/experiments/btc/metrics.csv'
validation_start = '2022-07-19 02:05:00'
test_start = '2022-11-05 13:02:00'
np.random.seed(123)

# Load data
data = pd.read_csv(data_path, sep=',', index_col='Date')

# Drop ETH and alt-coins that don't Granger-cause BTC
data.drop('ETH', axis=1, inplace=True)
data.drop('BNB', axis=1, inplace=True)

# Split train/validation/test (ratio=80/10/10)
train = data.loc[:validation_start].copy()
validation = data.loc[validation_start:test_start].copy()
validation = validation.iloc[1:]
test = data.loc[test_start:].copy()
test = test.iloc[1:]

# Scale benchmark and alt-coins
train_scaler = MinMaxScaler()
valid_scaler = MinMaxScaler()
test_scaler = MinMaxScaler()
train_scaled = pd.DataFrame(train_scaler.fit_transform(train), index=train.index, columns=train.columns)
validation_scaled = pd.DataFrame(valid_scaler.fit_transform(validation), index=validation.index,
                                 columns=validation.columns)
test_scaled = pd.DataFrame(test_scaler.fit_transform(test), index=test.index, columns=test.columns)

# Create Dataset objects
x_train = train_scaled.loc[:, train_scaled.columns != 'BTC']
y_train = train_scaled['BTC']
x_valid = validation_scaled.loc[:, validation_scaled.columns != 'BTC']
y_valid = validation_scaled['BTC']
x_test = test_scaled.loc[:, test_scaled.columns != 'BTC']

# CatBoost
model = CatBoostRegressor(iterations=500)

# Grid search learning rate
learning_rate_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
param_grid = dict(learning_rate=learning_rate_list)
grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=10)
grid_result = grid_search.fit(data.loc[:, train_scaled.columns != 'BTC'], data['BTC'])

# Experiment Properties
version = '1'
learning_rate = list(grid_result.best_params_.values())[0]
print(learning_rate)
n_iterations = 500
n_rounds = 20

# Optimal lr: 0.1

# CatBoost
model = CatBoostRegressor(iterations=n_iterations, learning_rate=learning_rate)

# Training/Validation
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], early_stopping_rounds=n_rounds,
          use_best_model=True)

# Save model
model.save_model(pretrained_path + 'catboost' + version + '.txt')

# Save Log
results = model.evals_result_
log_metric = pd.DataFrame(data=list(zip(results['validation_0']['RMSE'], results['validation_1']['RMSE'])),
                          columns=['Train Loss', 'Validation Loss'])
log_metric.to_csv(logs_path, sep='\t', encoding='utf-8')

# Predict and store predictions (in original scale)
predictions = model.predict(x_test)
test_pred = test_scaled.copy()
test_pred['BTC'] = predictions
test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred), index=test_pred.index, columns=test_pred.columns)
np.savetxt(predictions_path + 'catboost' + version + '.txt', test_pred['BTC'])

# Compute and store metrics
now = datetime.datetime.now()
mse = mean_squared_error(test['BTC'], test_pred['BTC'], squared=True)
rmse = mean_squared_error(test['BTC'], test_pred['BTC'], squared=False)
mae = mean_absolute_error(test['BTC'], test_pred['BTC'])
mape = mean_absolute_percentage_error(test['BTC'], test_pred['BTC'])
with open(metrics_path, 'a') as metrics:
    writer = csv.writer(metrics)
    writer.writerow([now, 'catboost', version, mse, rmse, mae, mape])
