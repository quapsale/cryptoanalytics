"""
File: data_pull.py
Description: Dataset generator (source: CoinMarketCap).
File Created: 06/10/2022
Python Version: 3.9
"""

# Imports
import os
import json
import argparse
import sys
from cryptocmd import CmcScraper
from datetime import datetime, timedelta
import pandas as pd

# Default args
now = datetime.now()
before = now - timedelta(days=1)

# Parser for CLI
parser = argparse.ArgumentParser(description='generate crypto ohlc price dataset from CoinMarketCap')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the dataset (default is current directory)')

# Filename
parser.add_argument('-f', '--filename', type=str, nargs='?',
                    help='filename for dataset (default is dataset_coinmarketcap_START_END)')

# Coins
parser.add_argument('-c', '--coins', type=str, nargs='?',
                    help='path for json with coin list (default are top-20 coins)')

# Starting date
parser.add_argument('-s', '--start', type=str,  nargs='?', default=before.strftime('%d-%m-%Y'),
                    help='starting date of the dataset (default is yesterday)')

# Ending date
parser.add_argument('-e', '--end', type=str,  nargs='?', default=now.strftime('%d-%m-%Y'),
                    help='ending date of the dataset (default is today)')

# Arg parse
args = parser.parse_args()
dfs = []
start = args.start
end = args.end

# Validate filename
if not args.filename:
    filename = 'dataset_coinmarketcap_' + start + '_' + end
    filename = filename.replace('-', '')
else:
    filename = args.filename

# Validate dates
try:
    start = datetime.strptime(start, '%d-%m-%Y')
    end = datetime.strptime(end, '%d-%m-%Y')

    # Exception (start>end)
    if start > end:
        print('Invalid dates provided: end time precedes start time!')
        sys.exit(1)

    start = datetime.strftime(start, '%d-%m-%Y')
    end = datetime.strftime(end, '%d-%m-%Y')

    # Exception (invalid path)
    if not os.path.exists(args.path):
        print('Invalid path provided: destination does not exist!')
        sys.exit(1)

    # If crypto json is provided
    if args.coins:
        try:
            f = open(args.coins)
            coins = json.load(f)

            try:
                cryptos = coins['coins']

                # Print args
                print({'--path': args.path, '--filename': filename,
                       '--coins': cryptos, '--start': start, '--end': end})

                # Exception (not a list)
                if not isinstance(cryptos, list):
                    print('Invalid data format: wrong coin list provided!')
                    sys.exit(1)

                # Scraper
                for crypto in cryptos:
                    try:

                        scraper = CmcScraper(crypto, start, end)
                        df = scraper.get_dataframe()
                        df.drop('Market Cap', axis=1, inplace=True)
                        df.drop('Volume', axis=1, inplace=True)
                        df['Coin'] = str(crypto)
                        dfs.append(df)

                    # Unavailable coins in the time frame
                    except IndexError:
                        print("The coin" + " " + str(crypto) +
                              " is available on CoinMarketCap but not for the time frame selected. "
                              "Retrying with next coins in the list.")

                    # Unavailable coins (non-existent)
                    except TypeError:
                        print("The coin" + " " + str(crypto) +
                              " is not available on CoinMarketCap. Retrying with next coins in the list.")

                # Generate df
                df_final = pd.concat(dfs)

                # Clean coins with missing data
                last = datetime.strptime(end, '%d-%m-%Y')
                first = datetime.strptime(start, '%d-%m-%Y')
                n_days = (last - first).days - 1
                v_count = df_final['Coin'].value_counts()
                to_remove = v_count[v_count < n_days].index
                df_final = df_final[~df_final.Coin.isin(to_remove)]
                df_final.reset_index()

                # Create csv
                file_name = os.path.join(args.path, filename + '.csv')
                df_final.to_csv(file_name, sep=',', encoding='utf-8', index=False)

            # Exception: bad formatted json
            except KeyError:
                print('Invalid data format: wrong coin list provided!')
                sys.exit(1)

        # Exception: file not found
        except FileNotFoundError:
            print('Invalid path provided: json file does not exist!')
            sys.exit(1)

        # Exception: not a json
        except ValueError:
            print('Invalid data format: coin list is not a json!')
            sys.exit(1)

    # Otherwise create default crypto list
    elif not args.coins:
        cryptos = ['btc', 'eth', 'usdt', 'usdc', 'bnb', 'xrp', 'busd', 'ada', 'sol', 'doge', 'matic', 'dot', 'dai', 'shib',
                   'trx', 'avax', 'uni', 'wbtc', 'leo', 'ltc']

        # Print args
        print({'--path': args.path, '--filename': filename,
               '--coins': cryptos, '--start': start, '--end': end})

        # Scraper
        for crypto in cryptos:
            try:

                scraper = CmcScraper(crypto, start, end)
                df = scraper.get_dataframe()
                df.drop('Market Cap', axis=1, inplace=True)
                df.drop('Volume', axis=1, inplace=True)
                df['Coin'] = str(crypto)
                dfs.append(df)

            # Unavailable coins in the time frame
            except IndexError:
                print("The coin" + " " + str(crypto) +
                      " is available on CoinMarketCap but not for the time frame selected. "
                      "Retrying with next coins in the list.")

            # Unavailable coins (non-existent)
            except TypeError:
                print("The coin" + " " + str(crypto) +
                      " is not available on CoinMarketCap. Retrying with next coins in the list.")

        # Generate df
        df_final = pd.concat(dfs)

        # Clean coins with missing data
        last = datetime.strptime(end, '%d-%m-%Y')
        first = datetime.strptime(start, '%d-%m-%Y')
        n_days = (last - first).days - 1
        v_count = df_final['Coin'].value_counts()
        to_remove = v_count[v_count < n_days].index
        df_final = df_final[~df_final.Coin.isin(to_remove)]
        df_final.reset_index()

        # Create csv
        file_name = os.path.join(args.path, filename + '.csv')
        df_final.to_csv(file_name, sep=',', encoding='utf-8', index=False)

# Wrong date format
except ValueError:
    print('Invalid date format: should be %d-%m-%Y!')
    sys.exit(1)
