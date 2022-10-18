"""
File: model_pretrain.py
Description: Model pretrain.
File Created: 06/10/2022
Python Version: 3.9
"""

# Imports
import os
import argparse
import sys
import pandas as pd
import json
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from pretrain.datasets import DatasetV1
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pretrain.gru import GRU
from pretrain.lstm import LSTM
import xgboost as xgb
import lightgbm as lgb
import catboost
import warnings
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning, PossibleUserWarning


# Ignore PL warnings
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
warnings.filterwarnings("ignore", category=PossibleUserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


# Parser for CLI
parser = argparse.ArgumentParser(description='pretrain ML models for crypto-coins forecast')

# Train
parser.add_argument('-tr', '--train', type=str, nargs=1, help='path to the csv training dataset')

# Valid
parser.add_argument('-vd', '--valid', type=str, nargs=1, help='path to the csv valid dataset')

# Target
parser.add_argument('-t', '--target', type=str, nargs=1, help='target coin to predict')

# Features
parser.add_argument('-ft', '--features', type=str, nargs='?',
                    help='path for json with feature variable list (default are all coins)')

# Model
parser.add_argument('-m', '--model', type=str, nargs=1, help='model to train for inference')

# Config
parser.add_argument('-c', '--config', type=str, nargs=1,
                    help='path for json with config for pretraining')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the pretrained model (default is current directory)')

# Filename
parser.add_argument('-f', '--filename', type=str, nargs='?',
                    help='filename for model (default is model_TODAY)')

# Arg parse
args = parser.parse_args()

# Check GPU
gpus = 1 if torch.cuda.is_available() else 0

# Exception (invalid path)
if not (os.path.exists(args.path)):
    print('Invalid path provided: folder does not exist!')
    sys.exit(1)

# Validate train
if not args.train:
    print('Missing argument: --train is required!')
    sys.exit(1)

# Validate valid
if not args.valid:
    print('Missing argument: --valid is required!')
    sys.exit(1)

# Validate config
if not args.config:
    print('Missing argument: --config is required!')
    sys.exit(1)

# Validate target
if not args.target:
    print('Missing argument: --target is required!')
    sys.exit(1)

# Validate model
if not args.model:
    print('Missing argument: --model is required!')
    sys.exit(1)

# Validate args
(target,) = args.target
(conf,) = args.config
(mdl,) = args.model
(train,) = args.train
(valid,) = args.valid

# Validate filename
if not args.filename:
    now = datetime.now()
    today = datetime.strftime(now, '%d-%m-%Y')
    filename = mdl + '_' + today
    filename = filename.replace('-', '')

else:
    filename = args.filename

# Print args
print({'--train': train, '--valid': valid, '--target': target, '--features': args.features,
       '--config': conf, '--model': mdl,  '--path': args.path, '--filename': filename})

try:
    f = open(conf)
    config = json.load(f)

    # Pretrain
    try:
        train = pd.read_csv(train, sep=',')
        valid = pd.read_csv(valid, sep=',')

        try:
            train['Date'] = pd.to_datetime(train['Date'])
            train = train.set_index('Date')
            valid['Date'] = pd.to_datetime(valid['Date'])
            valid = valid.set_index('Date')

            # Target not in data
            if target not in train.columns:
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
                        if not all(elem in train for elem in features):
                            print('Invalid data format: features are not in dataset!')
                            sys.exit(1)

                        # Subset
                        target_feature = features + [target]
                        train = train[target_feature]
                        train_scaler = MinMaxScaler()
                        train_scaled = pd.DataFrame(train_scaler.fit_transform(train),
                                                    index=train.index, columns=train.columns)
                        valid = valid[target_feature]
                        valid_scaler = MinMaxScaler()
                        valid_scaled = pd.DataFrame(valid_scaler.fit_transform(valid),
                                                    index=valid.index, columns=valid.columns)

                        # GRU
                        if mdl == 'gru':
                            if os.path.exists(os.path.join(args.path, filename + '.ckpt')):
                                print('A file ckpt already exists, please remove the old checkpoint '
                                      'before storing a new model.')
                                sys.exit(1)

                            train_dataset = DatasetV1(train_scaled, target=target, features=features)
                            validation_dataset = DatasetV1(valid_scaled, target=target, features=features)

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_layers = config['n_layers']
                                hidden_units = config['hidden_units']
                                n_epochs = config['n_epochs']
                                patience = config['patience']
                                batch_size = config['batch_size']
                                num_workers = config['num_workers']

                                # Create DataLoader objects
                                train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                          num_workers=num_workers, shuffle=True)
                                validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                                               num_workers=num_workers, shuffle=False)

                                # Callbacks
                                early_stopping = EarlyStopping('val_loss', patience=patience)
                                checkpoint = ModelCheckpoint(args.path, filename=filename,
                                                             monitor='val_loss', mode='min')

                                # GRU and trainer
                                model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                            lr=learning_rate)
                                trainer = pl.Trainer(callbacks=[early_stopping, checkpoint], max_epochs=n_epochs,
                                                     gpus=gpus)

                                # Fit
                                trainer.fit(model, train_loader, validation_loader)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # LSTM
                        elif mdl == 'lstm':
                            if os.path.exists(os.path.join(args.path, filename + '.ckpt')):
                                print('A file ckpt already exists, please remove the old checkpoint '
                                      'before storing a new model.')
                                sys.exit(1)

                            train_dataset = DatasetV1(train, target=target, features=features)
                            validation_dataset = DatasetV1(valid, target=target, features=features)

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_layers = config['n_layers']
                                hidden_units = config['hidden_units']
                                n_epochs = config['n_epochs']
                                patience = config['patience']
                                batch_size = config['batch_size']
                                num_workers = config['num_workers']

                                # Create DataLoader objects
                                train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                                          num_workers=num_workers, shuffle=True)
                                validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                                               num_workers=num_workers, shuffle=False)

                                # Callbacks
                                early_stopping = EarlyStopping('val_loss', patience=patience)
                                checkpoint = ModelCheckpoint(args.path, filename=filename,
                                                             monitor='val_loss', mode='min')

                                # LSTM and trainer
                                model = LSTM(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                             lr=learning_rate)
                                trainer = pl.Trainer(callbacks=[early_stopping, checkpoint], max_epochs=n_epochs,
                                                     gpus=gpus)

                                # Fit
                                trainer.fit(model, train_loader, validation_loader)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # XGBoost
                        elif mdl == 'xgboost':

                            if os.path.exists(os.path.join(args.path, filename + '.txt')):
                                print('A file .txt already exists, please remove the old checkpoint '
                                      'before storing a new model.')
                                sys.exit(1)

                            x_train = train.loc[:, train.columns != target]
                            y_train = train[target]
                            x_valid = valid.loc[:, valid.columns != target]
                            y_valid = valid[target]

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_trees = config['n_trees']
                                patience = config['patience']

                                # XGBoost
                                model = xgb.XGBRegressor(n_estimators=n_trees, learning_rate=learning_rate)

                                # Training/Validation
                                model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                                          early_stopping_rounds=patience)

                                # Save pretrained XGBoost
                                file_path = os.path.join(args.path, filename + '.txt')
                                model.save_model(file_path)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # LightGBM
                        elif mdl == 'lightgbm':
                            if os.path.exists(os.path.join(args.path, filename + '.txt')):
                                print('A file .txt already exists, please remove the old checkpoint '
                                      'before storing a new model.')
                                sys.exit(1)

                            x_train = train.loc[:, train.columns != target]
                            y_train = train[target]
                            x_valid = valid.loc[:, valid.columns != target]
                            y_valid = valid[target]

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_trees = config['n_trees']
                                patience = config['patience']

                                # LightGBM
                                model = lgb.LGBMRegressor(n_estimators=n_trees, learning_rate=learning_rate)

                                # Training/Validation
                                model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                                          callbacks=[lgb.early_stopping(stopping_rounds=patience)],
                                          eval_metric='rmse')

                                # Save pretrained LightGBM
                                file_path = os.path.join(args.path, filename + '.txt')
                                model.booster_.save_model(file_path, num_iteration=model.booster_.best_iteration)

                            # Exception: bad formatted json
                            except KeyError:
                                print('Invalid data format: wrong config list provided!')
                                sys.exit(1)

                        # Catboost
                        elif mdl == 'catboost':
                            if os.path.exists(os.path.join(args.path, filename + '.txt')):
                                print('A file .txt already exists, please remove the old checkpoint '
                                      'before storing a new model.')
                                sys.exit(1)

                            x_train = train.loc[:, train.columns != target]
                            y_train = train[target]
                            x_valid = valid.loc[:, valid.columns != target]
                            y_valid = valid[target]

                            try:
                                pl.seed_everything(config['seed'])
                                learning_rate = config['learning_rate']
                                n_trees = config['n_trees']
                                patience = config['patience']

                                # CatBoost
                                model = catboost.CatBoostRegressor(iterations=n_trees, learning_rate=learning_rate)

                                # Training/Validation
                                model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                                          early_stopping_rounds=patience, use_best_model=True)

                                # Save pretrained LightGBM
                                file_path = os.path.join(args.path, filename + '.txt')
                                model.save_model(file_path)

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
            features = [i for i in train.columns if i not in [target]]
            train_scaler = MinMaxScaler()
            train_scaled = pd.DataFrame(train_scaler.fit_transform(train),
                                        index=train.index, columns=train.columns)
            valid_scaler = MinMaxScaler()
            valid_scaled = pd.DataFrame(valid_scaler.fit_transform(valid),
                                        index=valid.index, columns=valid.columns)

            # GRU
            if mdl == 'gru':
                if os.path.exists(os.path.join(args.path, filename + '.ckpt')):
                    print('A file .ckpt already exists, please remove the old checkpoint '
                          'before storing a new model.')
                    sys.exit(1)

                train_dataset = DatasetV1(train_scaled, target=target, features=features)
                validation_dataset = DatasetV1(valid_scaled, target=target, features=features)

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_layers = config['n_layers']
                    hidden_units = config['hidden_units']
                    n_epochs = config['n_epochs']
                    patience = config['patience']
                    batch_size = config['batch_size']
                    num_workers = config['num_workers']

                    # Create DataLoader objects
                    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
                    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=False)

                    # Callbacks
                    early_stopping = EarlyStopping('val_loss', patience=patience)
                    checkpoint = ModelCheckpoint(args.path, filename=filename,
                                                 monitor='val_loss', mode='min')

                    # GRU and trainer
                    model = GRU(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                lr=learning_rate)
                    trainer = pl.Trainer(callbacks=[early_stopping, checkpoint], max_epochs=n_epochs,
                                         gpus=gpus)

                    # Fit
                    trainer.fit(model, train_loader, validation_loader)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # LSTM
            elif mdl == 'lstm':
                if os.path.exists(os.path.join(args.path, filename + '.ckpt')):
                    print('A file .ckpt already exists, please remove the old checkpoint '
                          'before storing a new model.')
                    sys.exit(1)

                train_dataset = DatasetV1(train, target=target, features=features)
                validation_dataset = DatasetV1(valid, target=target, features=features)

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_layers = config['n_layers']
                    hidden_units = config['hidden_units']
                    n_epochs = config['n_epochs']
                    patience = config['patience']
                    batch_size = config['batch_size']
                    num_workers = config['num_workers']

                    # Create DataLoader objects
                    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                              num_workers=num_workers, shuffle=True)
                    validation_loader = DataLoader(validation_dataset, batch_size=batch_size,
                                                   num_workers=num_workers, shuffle=False)

                    # Callbacks
                    early_stopping = EarlyStopping('val_loss', patience=patience)
                    checkpoint = ModelCheckpoint(args.path, filename=filename,
                                                 monitor='val_loss', mode='min')

                    # LSTM and trainer
                    model = LSTM(n_features=len(features), hidden_units=hidden_units, n_layers=n_layers,
                                 lr=learning_rate)
                    trainer = pl.Trainer(callbacks=[early_stopping, checkpoint], max_epochs=n_epochs,
                                         gpus=gpus)

                    # Fit
                    trainer.fit(model, train_loader, validation_loader)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # XGBoost
            elif mdl == 'xgboost':
                if os.path.exists(os.path.join(args.path, filename + '.txt')):
                    print('A file .txt already exists, please remove the old checkpoint '
                          'before storing a new model.')
                    sys.exit(1)

                x_train = train.loc[:, train.columns != target]
                y_train = train[target]
                x_valid = valid.loc[:, valid.columns != target]
                y_valid = valid[target]

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_trees = config['n_trees']
                    patience = config['patience']

                    # XGBoost
                    model = xgb.XGBRegressor(n_estimators=n_trees, learning_rate=learning_rate)

                    # Training/Validation
                    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                              early_stopping_rounds=patience)

                    # Save pretrained XGBoost
                    file_path = os.path.join(args.path, filename + '.txt')
                    model.save_model(file_path)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # LightGBM
            elif mdl == 'lightgbm':
                if os.path.exists(os.path.join(args.path, filename + '.txt')):
                    print('A file .txt already exists, please remove the old checkpoint '
                          'before storing a new model.')
                    sys.exit(1)

                x_train = train.loc[:, train.columns != target]
                y_train = train[target]
                x_valid = valid.loc[:, valid.columns != target]
                y_valid = valid[target]

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_trees = config['n_trees']
                    patience = config['patience']

                    # LightGBM
                    model = lgb.LGBMRegressor(n_estimators=n_trees, learning_rate=learning_rate)

                    # Training/Validation
                    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                              callbacks=[lgb.early_stopping(stopping_rounds=patience)],
                              eval_metric='rmse')

                    # Save pretrained LightGBM
                    file_path = os.path.join(args.path, filename + '.txt')
                    model.booster_.save_model(file_path)

                # Exception: bad formatted json
                except KeyError:
                    print('Invalid data format: wrong config list provided!')
                    sys.exit(1)

            # Catboost
            elif mdl == 'catboost':
                if os.path.exists(os.path.join(args.path, filename + '.txt')):
                    print('A file .txt already exists, please remove the old checkpoint '
                          'before storing a new model.')
                    sys.exit(1)

                x_train = train.loc[:, train.columns != target]
                y_train = train[target]
                x_valid = valid.loc[:, valid.columns != target]
                y_valid = valid[target]

                try:
                    pl.seed_everything(config['seed'])
                    learning_rate = config['learning_rate']
                    n_trees = config['n_trees']
                    patience = config['patience']

                    # CatBoost
                    model = catboost.CatBoostRegressor(iterations=n_trees, learning_rate=learning_rate)

                    # Training/Validation
                    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)],
                              early_stopping_rounds=patience, use_best_model=True)

                    # Save pretrained LightGBM
                    file_path = os.path.join(args.path, filename + '.txt')
                    model.save_model(file_path)

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
