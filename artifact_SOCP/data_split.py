"""
File: data_split.py
Description: Train-valid-test split.
File Created: 06/10/2022
Python Version: 3.9
"""

# Imports
import os
import argparse
from datetime import datetime
import sys
import pandas as pd

# Parser for CLI
parser = argparse.ArgumentParser(description='split coin price dataset for model training and forecast')

# Data
parser.add_argument('-d', '--data', type=str, nargs=1, help='path to the csv dataset')

# Variable
parser.add_argument('-v', '--variable', type=str, nargs='?', default='avg_ohlc',
                    help='variable to consider for computations (default is avg ohlc prices)')

# Train
parser.add_argument('-tr', '--train', type=float, nargs=1, help='ratio for train set split')

# Valid
parser.add_argument('-vd', '--valid', type=float, nargs=1, help='ratio for valid set split')

# Test
parser.add_argument('-ts', '--test', type=float, nargs=1, help='ratio for test set split')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the splits (default is current directory)')

# Filename
parser.add_argument('-f', '--filenames', nargs='+',
                    help='filenames for dataset splits (defaults are train_TODAY, etc)')

# Arg parse
args = parser.parse_args()

# Validate data
if not args.data:
    print('Missing argument: --data is required!')
    sys.exit(1)

# No data split
if not (args.train and args.valid and args.test):
    print('Missing arguments: --train, --valid and --test are required!')
    sys.exit(1)

# Exception (invalid path)
if not os.path.exists(args.path):
    print('Invalid path provided: destination does not exist!')
    sys.exit(1)

# Validate ratios
(tr,) = args.train
(vd,) = args.valid
(ts,) = args.test
(data,) = args.data

# Sum != 1
if sum([tr, vd, ts]) != 1:
    print('Invalid arguments provided: sum of train, valid and test ratios is not 1!')
    sys.exit(1)

# Validate filename
if not args.filenames:
    now = datetime.now()
    today = datetime.strftime(now, '%d-%m-%Y')
    filename_1 = 'train_' + today
    filename_1 = filename_1.replace('-', '')
    filename_2 = 'valid_' + today
    filename_2 = filename_2.replace('-', '')
    filename_3 = 'test_' + today
    filename_3 = filename_3.replace('-', '')

else:
    # More than 3 filenames
    if len(args.filenames) != 3:
        print('Invalid arguments provided: number of filenames is not 3!')
        sys.exit(1)
    filenames = args.filenames
    filename_1 = filenames[0]
    filename_2 = filenames[1]
    filename_3 = filenames[2]

# Print args
print({'--data': data, '--variable': args.variable, '--train': tr,
       '--valid': vd, '--test': ts, '--path': args.path,  '--filenames': [filename_1, filename_2, filename_3]})

# Extract and process data
try:
    data = pd.read_csv(data, sep=',')
    try:
        data['Date'] = pd.to_datetime(data['Date'])

        # Avg OHLC
        if args.variable == 'avg_ohlc':
            df = pd.DataFrame(data['Date'].unique(), columns=['Date'])
            df = df.set_index('Date')
            data['Avg OHLC Price'] = data[['Open', 'High', 'Low', 'Close']].mean(axis=1)
            data.drop(['Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
            data = data.set_index('Date')
            group = data.groupby('Coin')
            for i in data['Coin'].unique():
                coin = group.get_group(i)
                coin = coin.rename(columns={'Avg OHLC Price': i})
                df[i] = coin[i]
            df = df.sort_index()
            df = df.reset_index()

            try:
                ratio_tr = round(len(df) * tr)
                ratio_vd = ratio_tr + round(len(df) * vd)
                ratio_ts = ratio_vd + round(len(df) * ts)
                train = df.loc[:ratio_tr].copy()
                train = train.set_index('Date')
                valid = df.loc[1 + ratio_tr:ratio_vd].copy()
                valid = valid.set_index('Date')
                test = df.loc[1 + ratio_vd:ratio_ts].copy()
                test = test.set_index('Date')
                file_train = os.path.join(args.path, filename_1 + '.csv')
                train.to_csv(file_train, sep=',', encoding='utf-8', index=True)
                file_valid = os.path.join(args.path,  filename_2 + '.csv')
                valid.to_csv(file_valid, sep=',', encoding='utf-8', index=True)
                file_test = os.path.join(args.path,  filename_3 + '.csv')
                test.to_csv(file_test, sep=',', encoding='utf-8', index=True)

            except TypeError:
                print('Invalid ratio format: should be a number!')
                sys.exit(1)

        # Close
        elif args.variable == 'close':
            df = pd.DataFrame(data['Date'].unique(), columns=['Date'])
            df = df.set_index('Date')
            data.drop(['Open', 'High', 'Low'], axis=1, inplace=True)
            data = data.set_index('Date')
            group = data.groupby('Coin')
            for i in data['Coin'].unique():
                coin = group.get_group(i)
                coin = coin.rename(columns={'Close': i})
                df[i] = coin[i]
            df = df.sort_index()
            df = df.reset_index()

            try:
                ratio_tr = round(len(df) * tr)
                ratio_vd = ratio_tr + round(len(df) * vd)
                ratio_ts = ratio_vd + round(len(df) * ts)
                train = df.loc[:ratio_tr].copy()
                train = train.set_index('Date')
                valid = df.loc[1 + ratio_tr:ratio_vd].copy()
                valid = valid.set_index('Date')
                test = df.loc[1 + ratio_vd:ratio_ts].copy()
                test = test.set_index('Date')
                file_train = os.path.join(args.path,  filename_1 + '.csv')
                train.to_csv(file_train, sep=',', encoding='utf-8', index=True)
                file_valid = os.path.join(args.path,  filename_2 + '.csv')
                valid.to_csv(file_valid, sep=',', encoding='utf-8', index=True)
                file_test = os.path.join(args.path,  filename_3 + '.csv')
                test.to_csv(file_test, sep=',', encoding='utf-8', index=True)

            except TypeError:
                print('Invalid ratio format: should be a number!')
                sys.exit(1)

        # Not allowed
        elif args.variable not in ['avg_ohlc', 'close']:
            print('Invalid variable selected: allowed are avg_ohlc and close!')
            sys.exit(1)

    # Exception: bad formatted csv
    except KeyError:
        print('Invalid data format: wrong coin data provided!')
        sys.exit(1)

# Exception: file not found
except FileNotFoundError:
    print('Invalid path provided: csv file does not exist!')
    sys.exit(1)
