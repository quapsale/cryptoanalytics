"""
File: coinmarketcap_data_pull.py
Description: Dataset generator (source: CoinMarketCap).
File Created: 06/01/2022
Python Version: 3.9
"""

# Imports
import os
from cryptocmd import CmcScraper
from datetime import datetime
import pandas as pd

# File properties
data_path = './data/datasets'
starting = '13-03-2020'  # replace with desired start date in format %d-%m-%Y
today = '22-06-2022'

# Create crypto list
cryptos = ['btc', 'eth', 'bnb', 'sol', 'ada', 'xrp', 'luna', 'dot', 'avax', 'doge', 'matic', 'shib', 'cro', 'link',
           'wbtc', 'near', 'uni', 'ltc', 'algo', 'atom', 'ftm', 'bch', 'xlm', 'trx', 'icp', 'ftt', 'mana', 'vet',
           'hbar', 'sand', 'axs', 'btcb', 'fil', 'theta', 'egld', 'etc', 'one', 'xmr', 'xtz', 'klay', 'leo', 'hnt',
           'miota', 'aave', 'cake', 'grt', 'eos', 'stx', 'btt', 'flow', 'gala', 'ksm', 'rune', 'crv', 'enj', 'mkr',
           'bsv', 'lrc', 'qnt', 'celo', 'zec', 'xec', 'amp', 'rose', 'ar', 'kda', 'neo', 'bat', 'kcs', 'chz', 'okb',
           'waves', 'ht', 'dash', 'nexo', 'comp', 'yfi', 'mina', 'xdc', 'hot', '1inch', 'xem', 'rvn', 'iotx', 'scrt',
           'sushi', 'tfuel', 'woo', 'vlx', 'omg', 'bnt', 'bora', 'dcr', 'qtum']
dfs = []

# Scraper
for crypto in cryptos:
    scraper = CmcScraper(crypto, starting, today)
    df = scraper.get_dataframe()
    df['Coin'] = str(crypto)
    dfs.append(df)

# Generate df
df_final = pd.concat(dfs)

# Clean coins with missing data
last = datetime.strptime(today, '%d-%m-%Y')
first = datetime.strptime(starting, '%d-%m-%Y')
n_days = (last-first).days - 1
v_count = df_final['Coin'].value_counts()
to_remove = v_count[v_count < n_days].index
df_final = df_final[~df_final.Coin.isin(to_remove)]
df_final.reset_index()

# Describe data
df_final.head(10)
df_final.describe()

# Create csv
file_name = os.path.join(data_path, 'coinmarketcap.csv')
df_final.to_csv(file_name, sep='\t', encoding='utf-8', index=False)
