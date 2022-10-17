"""
File: correlations_analysis.py
Description: Correlation Analysis.
File Created: 06/10/2022
Python Version: 3.9
"""


# Imports
import os
import argparse
import sys
import pandas as pd

# Parser for CLI
parser = argparse.ArgumentParser(description='compute correlations between coin prices (avg ohlc or close)')

# Path
parser.add_argument('-d', '--data', type=str, nargs='?', default=os.path.join(os.getcwd(), 'dataset_coinmarketcap.csv'),
                    help='path to the csv dataset (default is current directory)')

# Variable
parser.add_argument('-v', '--variable', type=str, nargs='?', default='avg_ohlc',
                    help='variable on which computing correlations (default is avg ohlc prices)')

# Segment
parser.add_argument('-w', '--window', type=str, nargs='?', default='daily',
                    help='sliding window to use for computations (default is daily)')

# Method
parser.add_argument('-m', '--method', type=str, nargs='?', default='pearson',
                    help='method to compute correlations (default is pearson)')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the correlation dataset (default is current directory)')

# Arg parse
args = parser.parse_args()

# Exception (invalid path)
if not os.path.exists(args.path):
    print('Invalid path provided: destination does not exist!')
    sys.exit(1)

# Extract and process data
try:
    data = pd.read_csv(args.data, sep='\t')
    try:
        data['Date'] = pd.to_datetime(data['Date'])

        # If daily
        if args.window == 'daily':

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

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
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

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
                    sys.exit(1)

            # Not allowed
            elif args.variable not in ['avg_ohlc', 'close']:
                print('Invalid variable selected: allowed are avg_ohlc and close!')
                sys.exit(1)

        # If weekly
        elif args.window == 'weekly':

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
                    coin = coin.rolling('7D', on=coin.index).mean(numeric_only=True).sort_values(by='Date')
                    coin = coin.rename(columns={'Avg OHLC Price': i})
                    df[i] = coin[i]

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
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
                    coin = coin.rolling('7D', on=coin.index).mean(numeric_only=True).sort_values(by='Date')
                    coin = coin.rename(columns={'Close': i})
                    df[i] = coin[i]

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
                    sys.exit(1)

            # Not allowed
            elif args.variable not in ['avg_ohlc', 'close']:
                print('Invalid variable selected: allowed are avg_ohlc and close!')
                sys.exit(1)

        # If monthly
        elif args.window == 'monthly':

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
                    coin = coin.rolling('30D', on=coin.index).mean(numeric_only=True).sort_values(by='Date')
                    coin = coin.rename(columns={'Avg OHLC Price': i})
                    df[i] = coin[i]

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
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
                    coin = coin.rolling('30D', on=coin.index).mean(numeric_only=True).sort_values(by='Date')
                    coin = coin.rename(columns={'Close': i})
                    df[i] = coin[i]

                # Pearson
                if args.method == 'pearson':
                    corr = df.corr(method='pearson')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Kendall
                elif args.method == 'kendall':
                    corr = df.corr(method='kendall')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Spearman
                elif args.method == 'spearman':
                    corr = df.corr(method='spearman')
                    file_name = os.path.join(args.path, 'correlation.csv')
                    corr.to_csv(file_name, sep='\t', encoding='utf-8', index=True)

                # Not allowed
                elif args.method not in ['pearson', 'kendall', 'spearman']:
                    print('Invalid method selected: allowed are pearson, kendall and spearman!')
                    sys.exit(1)

            # Not allowed
            elif args.variable not in ['avg_ohlc', 'close']:
                print('Invalid variable selected: allowed are avg_ohlc and close!')
                sys.exit(1)

        # Not allowed
        elif args.window not in ['daily', 'weekly', 'monthly']:
            print('Invalid window selected: allowed are daily, weekly and monthly!')
            sys.exit(1)

    # Exception: bad formatted csv
    except KeyError:
        print('Invalid data format: wrong coin data provided!')
        sys.exit(1)

# Exception: file not found
except FileNotFoundError:
    print('Invalid path provided: csv file does not exist!')
    sys.exit(1)
