"""
File: lightgbm.py
Description: Experiments with LightGBM model.
File Created: 06/04/2022
Python Version: 3.9
"""

# Imports
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import csv

# File Properties
data_path = './data/datasets/binance.csv'
gb_path = './analysis/forecast/experiments/eth/gradient_boosting/'
predictions_path = './analysis/forecast/experiments/eth/predictions/'
pretrained_path = './analysis/forecast/experiments/eth/pretrained/'
logs_path = './analysis/forecast/experiments/eth/loggers/lightgbm/metrics.csv'
metrics_path = './analysis/forecast/experiments/eth/metrics.csv'
validation_start = '2022-03-31 07:29:00'
test_start = '2022-05-11 15:57:00'
np.random.seed(123)

# Load data
data = pd.read_csv(data_path, sep='\t', index_col='Date')

# Drop BTC and alt-coins that don't Granger-cause ETH
data.drop('BTC', axis=1, inplace=True)
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
x_train = train_scaled.loc[:, train_scaled.columns != 'ETH']
y_train = train_scaled['ETH']
x_valid = validation_scaled.loc[:, validation_scaled.columns != 'ETH']
y_valid = validation_scaled['ETH']
x_test = test_scaled.loc[:, test_scaled.columns != 'ETH']

# LightGBM
model = lgb.LGBMRegressor(n_estimators=500)

# Grid search learning rate
learning_rate_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
param_grid = dict(learning_rate=learning_rate_list)
grid_search = GridSearchCV(model, param_grid, scoring='neg_root_mean_squared_error', cv=10)
grid_result = grid_search.fit(data.loc[:, train_scaled.columns != 'ETH'], data['ETH'])

# Experiment Properties
version = '1'
learning_rate = list(grid_result.best_params_.values())[0]
n_estimators = 500
n_rounds = 20

# Optimal lr: 0.1

# LightGBM
model = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate)

# Training/Validation
model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
          callbacks=[lgb.early_stopping(stopping_rounds=n_rounds)], eval_metric='rmse')

# Save model
model.booster_.save_model(pretrained_path + 'lightgbm' + version + '.txt')

# Save Log
results = model.evals_result_
log_metric = pd.DataFrame(data=list(zip(results['training']['rmse'], results['valid_1']['rmse'])),
                          columns=['Train Loss', 'Validation Loss'])
log_metric.to_csv(logs_path, sep='\t', encoding='utf-8')

# Predict and store predictions (in original scale)
predictions = model.predict(x_test)
test_pred = test_scaled.copy()
test_pred['ETH'] = predictions
test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred), index=test_pred.index, columns=test_pred.columns)
np.savetxt(predictions_path + 'lightgbm' + version + '.txt', test_pred['ETH'])

# Compute and store metrics
now = datetime.datetime.now()
mse = mean_squared_error(test['ETH'], test_pred['ETH'], squared=True)
rmse = mean_squared_error(test['ETH'], test_pred['ETH'], squared=False)
mae = mean_absolute_error(test['ETH'], test_pred['ETH'])
mape = mean_absolute_percentage_error(test['ETH'], test_pred['ETH'])
with open(metrics_path, 'a') as metrics:
    writer = csv.writer(metrics)
    writer.writerow([now, 'lightgbm', version, mse, rmse, mae, mape])
