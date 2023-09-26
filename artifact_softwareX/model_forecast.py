"""
File: model_pretrain.py
Description: Model forecast.
File Created: 06/09/2023
Python Version: 3.9
"""

# Imports
import json
import os
import argparse
import sys
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pretrain.datasets import DatasetV1
from torch.utils.data import DataLoader
from pretrain.gru import GRU
from pretrain.lstm import LSTM
import xgboost as xgb
import catboost
import lightgbm as lgb
import pytorch_lightning as pl
import warnings
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning, PossibleUserWarning

# Ignore PL warnings
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Parser for CLI
parser = argparse.ArgumentParser(description='forecast crypto-coins prices with ML models')

# Test
parser.add_argument('-hz', '--horizon', type=int, nargs=1, help='forecasting horizon, i.e. number of '
                                                                'future prices to predict')

# Pretrained
parser.add_argument('-pt', '--pretrained', type=str, nargs=1, help='path to the pretrained model')

# Target
parser.add_argument('-t', '--target', type=str, nargs=1, help='target coin to predict (same as pretraining)')

# Features
parser.add_argument('-ft', '--features', type=str, nargs='?',
                    help='path for json with feature variable list (default are all coins, same as pretraining)')

# Model
parser.add_argument('-m', '--model', type=str, nargs=1, help='model to use for inference')

# Config
parser.add_argument('-c', '--config', type=str, nargs=1,
                    help='path for json with config for prediction (same as pretraining)')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the predictions (default is current directory)')

# Filename
parser.add_argument('-f', '--filename', type=str, nargs='?',
                    help='filename for predictions (default is predictions_model_TODAY)')

# Arg parse
args = parser.parse_args()

# Check GPU
gpus = 1 if torch.cuda.is_available() else 0

# Warning (invariance respect to training)
print('User info: for the prediction phase, always use the same settings of '
      'the pre-training (config, target, features).')

# Exception (invalid path)
if not (os.path.exists(args.path)):
    print('Invalid path provided: folder does not exist!')
    sys.exit(1)

# Validate horizon
if not args.horizon:
    print('Missing argument: --horizon is required!')
    sys.exit(1)

# Validate target
if not args.target:
    print('Missing argument: --target is required!')
    sys.exit(1)

# Validate config
if not args.config:
    print('Missing argument: --config is required!')
    sys.exit(1)

# Validate pretrained
if not args.pretrained:
    print('Missing argument: --pretrained is required!')
    sys.exit(1)

# Validate model
if not args.model:
    print('Missing argument: --model is required!')
    sys.exit(1)

# Validate args
(target,) = args.target
(conf,) = args.config
(mdl,) = args.model
(horizon,) = args.horizon
(prt,) = args.pretrained

# Validate filename
if not args.filename:
    now = datetime.now()
    today = datetime.strftime(now, '%d-%m-%Y')
    filename = 'predictions_' + mdl + '_' + today
    filename = filename.replace('-', '')

else:
    filename = args.filename

# Print args
print({'--horizon': horizon, '--target': target, '--features': args.features, '--pretrained': prt,
       '--config': conf, '--model': mdl,  '--path': args.path, '--filename': filename})

# Predict
try:
    f = open(conf)
    config = json.load(f)

    try:
        test = pd.read_csv(test, sep=',')

        try:
            test['Date'] = pd.to_datetime(test['Date'])
            test = test.set_index('Date')

            # Target not in data
            if target not in test.columns:
                print('Invalid data format: target is not in dataset!')
                sys.exit(1)

            # If features json is provided
            if args.features:
                try:
                    f = open(args.features)
                    features = json.load(f)

                    try:
                        features = features['features']

                        # Exception (not a list)
                        if not isinstance(features, list):
                            print('Invalid data format: wrong features list provided!')
                            sys.exit(1)

                        # Target in features
                        if target in features:
                            print('Invalid data format: target is in features list!')
                            sys.exit(1)

                        # Features not in data
                        if not all(elem in test for elem in features):
                            print('Invalid data format: features are not in dataset!')
                            sys.exit(1)

                        # Subset
                        target_feature = features + [target]
                        test = test[target_feature]
                        test_scaler = MinMaxScaler()
                        test_scaled = pd.DataFrame(test_scaler.fit_transform(test),
                                                   index=test.index, columns=test.columns)

                        # GRU
                        if mdl == 'gru':
                            test_dataset = DatasetV1(test_scaled, target=target, features=features)

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_layers = config['n_layers']
                                hidden_units = config['hidden_units']
                                num_workers = config['num_workers']

                                # Create DataLoader object
                                test_loader = DataLoader(test_dataset, batch_size=1,
                                                         num_workers=num_workers, shuffle=False)

                                # Predict
                                model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                            lr=learning_rate)
                                trainer = pl.Trainer(gpus=gpus)
                                predictions = []
                                try:
                                    output = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=prt)
                                    for tensor in output:
                                        predictions.append(tensor.item())
                                    test_pred = test_scaled.copy()
                                    test_pred[target] = predictions
                                    test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                             index=test_pred.index, columns=test_pred.columns)
                                    # Create csv
                                    file_name = os.path.join(args.path, filename + '.csv')
                                    test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                                    print(test_pred[target])

                                # Exception: wrong model
                                except RuntimeError:
                                    print('Invalid model format: wrong pretrained model provided!')
                                    sys.exit(1)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # LSTM
                        if mdl == 'lstm':
                            test_dataset = DatasetV1(test_scaled, target=target, features=features)

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_layers = config['n_layers']
                                hidden_units = config['hidden_units']
                                num_workers = config['num_workers']

                                # Create DataLoader object
                                test_loader = DataLoader(test_dataset, batch_size=1,
                                                         num_workers=num_workers, shuffle=False)

                                # Predict
                                model = LSTM(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                             lr=learning_rate)
                                trainer = pl.Trainer(gpus=gpus)
                                predictions = []
                                try:
                                    output = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=prt)
                                    for tensor in output:
                                        predictions.append(tensor.item())
                                    test_pred = test_scaled.copy()
                                    test_pred[target] = predictions
                                    test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                             index=test_pred.index, columns=test_pred.columns)

                                    # Create csv
                                    file_name = os.path.join(args.path, filename + '.csv')
                                    test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                                    print(test_pred[target])

                                # Exception: wrong model
                                except RuntimeError:
                                    print('Invalid model format: wrong pretrained model provided!')
                                    sys.exit(1)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # XGBoost
                        if mdl == 'xgboost':

                            x_test = test_scaled.loc[:, test_scaled.columns != target]

                            try:
                                pl.seed_everything(config['seed'])
                                n_trees = config['n_trees']

                                # Predict
                                model = xgb.XGBRegressor(n_estimators=n_trees)

                                try:
                                    model.load_model(prt)
                                    predictions = model.predict(x_test, ntree_limit=model.best_iteration)
                                    test_pred = test_scaled.copy()
                                    test_pred[target] = predictions
                                    test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                             index=test_pred.index, columns=test_pred.columns)

                                    # Create csv
                                    file_name = os.path.join(args.path, filename + '.csv')
                                    test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                                    print(test_pred[target])

                                # Exception: wrong model
                                except ValueError:
                                    print('Invalid model format: wrong pretrained model provided!')
                                    sys.exit(1)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # LightGBM
                        if mdl == 'lightgbm':

                            x_test = test_scaled.loc[:, test_scaled.columns != target]

                            try:
                                pl.seed_everything(config['seed'])

                                try:
                                    model = lgb.Booster(model_file=prt)
                                    predictions = model.predict(x_test)
                                    test_pred = test_scaled.copy()
                                    test_pred[target] = predictions
                                    test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                             index=test_pred.index, columns=test_pred.columns)

                                    # Create csv
                                    file_name = os.path.join(args.path, filename + '.csv')
                                    test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                                    print(test_pred[target])

                                # Exception: wrong model
                                except ValueError:
                                    print('Invalid model format: wrong pretrained model provided!')
                                    sys.exit(1)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # CatBoost
                        if mdl == 'catboost':

                            x_test = test_scaled.loc[:, test_scaled.columns != target]

                            try:
                                pl.seed_everything(config['seed'])
                                n_trees = config['n_trees']

                                # Predict
                                model = catboost.CatBoostRegressor(iterations=n_trees)

                                try:
                                    model.load_model(prt)
                                    predictions = model.predict(x_test)
                                    test_pred = test_scaled.copy()
                                    test_pred[target] = predictions
                                    test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                             index=test_pred.index, columns=test_pred.columns)

                                    # Create csv
                                    file_name = os.path.join(args.path, filename + '.csv')
                                    test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                                    print(test_pred[target])

                                # Exception: wrong model
                                except catboost.CatboostError:
                                    print('Invalid model format: wrong pretrained model provided!')
                                    sys.exit(1)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # Not allowed
                        elif mdl not in ['gru', 'lstm', 'xgboost', 'lightgbm', 'catboost']:
                            print('Invalid model selected: allowed are gru, lstm, xgboost, lightgbm and catboost!')
                            sys.exit(1)

                    # Exception: bad formatted json
                    except KeyError:
                        print('Invalid data format: wrong features list provided!')
                        sys.exit(1)

                # Exception: file not found
                except FileNotFoundError:
                    print('Invalid path provided: json file does not exist!')
                    sys.exit(1)

                # Exception: not a json
                except ValueError:
                    print('Invalid data format: coin list is not a json!')
                    sys.exit(1)

            # If not provided, use all coins
            features = [i for i in test.columns if i not in [target]]
            test_scaler = MinMaxScaler()
            test_scaled = pd.DataFrame(test_scaler.fit_transform(test),
                                       index=test.index, columns=test.columns)

            # GRU
            if mdl == 'gru':
                test_dataset = DatasetV1(test_scaled, target=target, features=features)

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_layers = config['n_layers']
                    hidden_units = config['hidden_units']
                    num_workers = config['num_workers']

                    # Create DataLoader object
                    test_loader = DataLoader(test_dataset, batch_size=1,
                                             num_workers=num_workers, shuffle=False)

                    # Predict
                    model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                lr=learning_rate)
                    trainer = pl.Trainer(gpus=gpus)
                    predictions = []
                    try:
                        output = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=prt)
                        for tensor in output:
                            predictions.append(tensor.item())
                        test_pred = test_scaled.copy()
                        test_pred[target] = predictions
                        test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                 index=test_pred.index, columns=test_pred.columns)
                        # Create csv
                        file_name = os.path.join(args.path, filename + '.csv')
                        test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                        print(test_pred[target])

                    # Exception: wrong model
                    except RuntimeError:
                        print('Invalid model format: wrong pretrained model provided!')
                        sys.exit(1)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # LSTM
            if mdl == 'lstm':
                test_dataset = DatasetV1(test_scaled, target=target, features=features)

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_layers = config['n_layers']
                    hidden_units = config['hidden_units']
                    num_workers = config['num_workers']

                    # Create DataLoader object
                    test_loader = DataLoader(test_dataset, batch_size=1,
                                             num_workers=num_workers, shuffle=False)

                    # Predict
                    model = LSTM(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                 lr=learning_rate)
                    trainer = pl.Trainer(gpus=gpus)
                    predictions = []
                    try:
                        output = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=prt)
                        for tensor in output:
                            predictions.append(tensor.item())
                        test_pred = test_scaled.copy()
                        test_pred[target] = predictions
                        test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                 index=test_pred.index, columns=test_pred.columns)

                        # Create csv
                        file_name = os.path.join(args.path, filename + '.csv')
                        test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                        print(test_pred[target])

                    # Exception: wrong model
                    except RuntimeError:
                        print('Invalid model format: wrong pretrained model provided!')
                        sys.exit(1)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # XGBoost
            if mdl == 'xgboost':

                x_test = test_scaled.loc[:, test_scaled.columns != target]

                try:
                    pl.seed_everything(config['seed'])
                    n_trees = config['n_trees']

                    # Predict
                    model = xgb.XGBRegressor(n_estimators=n_trees)

                    try:
                        model.load_model(prt)
                        predictions = model.predict(x_test, ntree_limit=model.best_iteration)
                        test_pred = test_scaled.copy()
                        test_pred[target] = predictions
                        test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                 index=test_pred.index, columns=test_pred.columns)

                        # Create csv
                        file_name = os.path.join(args.path, filename + '.csv')
                        test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                        print(test_pred[target])

                    # Exception: wrong model
                    except ValueError:
                        print('Invalid model format: wrong pretrained model provided!')
                        sys.exit(1)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # LightGBM
            if mdl == 'lightgbm':

                x_test = test_scaled.loc[:, test_scaled.columns != target]

                try:
                    pl.seed_everything(config['seed'])

                    try:
                        model = lgb.Booster(model_file=prt)
                        predictions = model.predict(x_test)
                        test_pred = test_scaled.copy()
                        test_pred[target] = predictions
                        test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                 index=test_pred.index, columns=test_pred.columns)

                        # Create csv
                        file_name = os.path.join(args.path, filename + '.csv')
                        test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                        print(test_pred[target])

                    # Exception: wrong model
                    except ValueError:
                        print('Invalid model format: wrong pretrained model provided!')
                        sys.exit(1)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # CatBoost
            if mdl == 'catboost':

                x_test = test_scaled.loc[:, test_scaled.columns != target]

                try:
                    pl.seed_everything(config['seed'])
                    n_trees = config['n_trees']

                    # Predict
                    model = catboost.CatBoostRegressor(iterations=n_trees)

                    try:
                        model.load_model(prt)
                        predictions = model.predict(x_test)
                        test_pred = test_scaled.copy()
                        test_pred[target] = predictions
                        test_pred = pd.DataFrame(test_scaler.inverse_transform(test_pred),
                                                 index=test_pred.index, columns=test_pred.columns)

                        # Create csv
                        file_name = os.path.join(args.path, filename + '.csv')
                        test_pred[target].to_csv(file_name, sep=',', encoding='utf-8', index=True)
                        print(test_pred[target])

                    # Exception: wrong model
                    except catboost.CatboostError:
                        print('Invalid model format: wrong pretrained model provided!')
                        sys.exit(1)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # Not allowed
            elif mdl not in ['gru', 'lstm', 'xgboost', 'lightgbm', 'catboost']:
                print('Invalid model selected: allowed are gru, lstm, xgboost, lightgbm and catboost!')
                sys.exit(1)

        # Exception: bad formatted csv
        except KeyError:
            print('Invalid data format: wrong coin data provided!')
            sys.exit(1)

    # Exception: file not found
    except FileNotFoundError:
        print('Invalid path provided: csv file does not exist!')
        sys.exit(1)

# Exception: file not found
except FileNotFoundError:
    print('Invalid path provided: json file does not exist!')
    sys.exit(1)

# Exception: not a json
except ValueError:
    print('Invalid data format: config list is not a json!')
    sys.exit(1)
