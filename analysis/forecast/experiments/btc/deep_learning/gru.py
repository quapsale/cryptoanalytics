"""
File: gru.py
Description: Experiments with GRU model.
File Created: 06/04/2022
Python Version: 3.9
"""

# Imports
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from analysis.forecast.experiments.btc.deep_learning.dataset.datasets import DatasetV1
from torch.utils.data import DataLoader
from analysis.forecast.experiments.btc.deep_learning.models.gru import GRU
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime
import csv

# File Properties
data_path = './data/datasets/binance.csv'
rnn_path = './analysis/forecast/experiments/btc/deep_learning/'
predictions_path = './analysis/forecast/experiments/btc/predictions/'
pretrained_path = './analysis/forecast/experiments/btc/pretrained/'
checkpoint_path = './analysis/forecast/experiments/btc/checkpoints/'
logger_path = './analysis/forecast/experiments/btc/loggers/'
metrics_path = './analysis/forecast/experiments/btc/metrics.csv'
validation_start = '2022-03-31 07:29:00'
test_start = '2022-05-11 15:57:00'
pl.seed_everything(123)
batch_size = 256
num_workers = 16

# Send to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = pd.read_csv(data_path, sep='\t', index_col='Date')

# Drop ETH and alt-coins that don't Granger-cause BTC
data.drop('ETH', axis=1, inplace=True)
data.drop('ADA', axis=1, inplace=True)
data.drop('BNB', axis=1, inplace=True)
data.drop('XMR', axis=1, inplace=True)

# Split train/validation/test (ratio=90/5/5)
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
target = 'BTC'
features = list(data.columns.difference([target]))
train_dataset = DatasetV1(train_scaled, target=target, features=features)
validation_dataset = DatasetV1(validation_scaled, target=target, features=features)
test_dataset = DatasetV1(test_scaled, target=target, features=features)

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, shuffle=False)

# Experiment Properties
version = '1'
learning_rate = 1e-6
n_layers = 10
hidden_units = 100
n_epochs = 500
gpus = 1 if torch.cuda.is_available() else 0

# Callbacks
early_stopping = EarlyStopping('val_loss', patience=20)
checkpoint = ModelCheckpoint(checkpoint_path, filename='gru-{epoch:02d}-{val_loss:.2f}',
                             monitor='val_loss', mode='min')

# Create logger
logger = CSVLogger(logger_path, name='gru', version=version)

# GRU and trainer
model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers, lr=learning_rate)
trainer = pl.Trainer(callbacks=[early_stopping, checkpoint], max_epochs=n_epochs, gpus=gpus, logger=logger)

# Optimal lr finder
lr_finder = trainer.tuner.lr_find(
    model,
    train_dataloaders=train_loader,
    val_dataloaders=validation_loader,
    max_lr=10.0,
    min_lr=1e-6,
)
learning_rate = lr_finder.suggestion()

# Optimal lr v1: 0.0014

# Update model
model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers, lr=learning_rate)

# Fit
trainer.fit(model, train_loader, validation_loader)

# Save pretrained GRU
torch.save(model.state_dict(), pretrained_path + 'gru' + version + '.pth')

# Predict and store predictions (in original scale)
predictions = []
output = trainer.predict(dataloaders=test_loader, ckpt_path='best')
for tensor in output:
    predictions.append(tensor.item())
test_pred = test_scaled.copy()
test_pred['BTC'] = predictions
test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred), index=test_pred.index, columns=test_pred.columns)
np.savetxt(predictions_path + 'gru' + version + '.txt', test_pred['BTC'])

# Compute and store metrics
now = datetime.datetime.now()
mse = mean_squared_error(test['BTC'], test_pred['BTC'], squared=True)
rmse = mean_squared_error(test['BTC'], test_pred['BTC'], squared=False)
mae = mean_absolute_error(test['BTC'], test_pred['BTC'])
mape = mean_absolute_percentage_error(test['BTC'], test_pred['BTC'])
with open(metrics_path, 'a') as metrics:
    writer = csv.writer(metrics)
    writer.writerow([now, 'gru', version, mse, rmse, mae, mape])
