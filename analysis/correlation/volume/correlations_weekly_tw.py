"""
File: correlations_weekly_tw.py
Description: Correlation Analysis with weekly granularity and thumbing window (source: coinmarketcap dataset).
File Created: 06/01/2022
Python Version: 3.9
"""

# Imports
import os
import numpy as np
import pandas as pd

# File properties
data_path = './data/datasets/coinmarketcap.csv'
corr_data_path = './analysis/correlation/correlogram_data'
time_frame = 'W'
column_names = ['coin', 'x', 'y', 'radius', 'arc_begin', 'arc_end', 'color']
arc_begin = 0
arc_end = 360
color = 1

# Extract btc and eth benchmarks
data = pd.read_csv(data_path, sep='\t')
data['Date'] = pd.to_datetime(data['Date'])
data.drop(['Market Cap', 'Open', 'High', 'Low', 'Close'], axis=1, inplace=True)
group = data.groupby('Coin')
btc = group.get_group('btc')
btc = btc.resample(time_frame, on='Date').mean().reset_index().sort_values(by='Date')
btc = btc.set_index('Date')
eth = group.get_group('eth')
eth = eth.resample(time_frame, on='Date').mean().reset_index().sort_values(by='Date')
eth = eth.set_index('Date')
data = data[~data.Coin.isin(['btc', 'eth'])]

# Create dict of correlation matrices
coins = data['Coin'].unique()
coin_dict = dict()
btc_dict = dict()
eth_dict = dict()
for coin in coins:
    subset = data[data['Coin'] == coin]
    subset = subset.resample(time_frame, on='Date').mean().reset_index().sort_values(by='Date')
    subset = subset.set_index('Date')
    coin_dict.update({coin: subset})
for coin in coins:
    subset = coin_dict[coin]
    corr_btc = subset.corrwith(btc, axis=0)
    corr_eth = subset.corrwith(eth, axis=0)
    btc_dict.update({coin: corr_btc})
    eth_dict.update({coin: corr_eth})

# Create csv for correlogram
list_data_btc = []
list_data_eth = []
len_columns = len(column_names)-1
for coin in coins:
    coin_name = coin
    x_coordinate = np.where(coins == coin)[0].item()+1
    y_coordinate = 2
    radius_value_btc = btc_dict[coin][0]
    if radius_value_btc >= 0:
        color_value_btc = 1
    elif radius_value_btc < 0:
        color_value_btc = 2
    radius_value_btc = abs(radius_value_btc)
    radius_value_eth = eth_dict[coin][0]
    if radius_value_eth >= 0:
        color_value_eth = 1
    elif radius_value_eth < 0:
        color_value_eth = 2
    radius_value_eth = abs(radius_value_eth)
    arc_begin_value = arc_begin
    arc_end_value = arc_end
    data_coin_btc = [coin_name, x_coordinate, y_coordinate, radius_value_btc, arc_begin_value, arc_end_value,
                     color_value_btc]
    data_coin_eth = [coin_name, x_coordinate, y_coordinate, radius_value_eth, arc_begin_value, arc_end_value,
                     color_value_eth]
    corr_data_coin_btc = pd.DataFrame(data=data_coin_btc).T
    corr_data_coin_eth = pd.DataFrame(data=data_coin_eth).T
    corr_data_coin_btc.columns = column_names
    corr_data_coin_eth.columns = column_names
    list_data_btc.append(corr_data_coin_btc)
    list_data_eth.append(corr_data_coin_eth)
corr_data_btc = pd.concat(list_data_btc)
corr_data_eth = pd.concat(list_data_eth)
file_name_btc = os.path.join(corr_data_path, 'weekly_tw_btc_VOL.csv')
file_name_eth = os.path.join(corr_data_path, 'weekly_tw_eth_VOL.csv')
corr_data_btc.to_csv(file_name_btc, sep='\t', encoding='utf-8', index=False)
corr_data_eth.to_csv(file_name_eth, sep='\t', encoding='utf-8', index=False)
