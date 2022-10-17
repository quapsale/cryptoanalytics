# CryptoAnalytics-CLI

CryptoAnalytics-CLI is a Python Command Line Interface for the analysis and forecasting of financial time series and cryptocurrency price trends.

## Installation

The Python version used in this project is 3.9. Use the following command to generate a new virtual environment.

```bash
python3.9 -m venv venv
```

With the following you can activate your virtual environment.

```bash
source venv/bin/activate
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the project requirements as it follows.

```bash
pip freeze > requirements.txt
```

## CLI Structure
This CLI is organized as it follows:

1. data_pull.py: command to generate a new dataset of OHLC cryptocoin prices from [CoinMarketCap](https://coinmarketcap.com/CoinMarketCap).
2. correlation_analysis.py: command to analyze correlations among cryptocoin prices.
3. data_split.py: command to generate train, validation and test sets from the original data.
4. model_pretrain.py: command to pretrain ML models for cryptocoin prices forecast.
5. model_forecast.py: command to use the pretrained ML models to forecast cryptocoin prices.

## Usage
### Data Pull

```bash
python data_pull.py -p "destination_path" -c "examples\coins.json" -s "01-01-2020" -e "01-01-2022"
```
Args:
1. -p, --path: destination folder for dataset storage (OPTIONAL, defaults to current directory).
2. -c, --coins: path to .json file with list of coins to include in the dataset (OPTIONAL, defaults to top-20 coins for market cap).
3. -s, --start: starting date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to yesterday).
4. -e, --end: ending date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to today).

### Correlation Analysis

```bash
python correlation_analysis.py -p "destination_path" -d "data_path" -v "avg_ohlc" -w "daily" -m "pearson"
```
Args:
1. -p, --path: destination folder for correlations storage (OPTIONAL, defaults to current directory).
2. -d, --data: path to .csv file dataset (OPTIONAL, defaults to current directory).
3. -v, --variable: price variable on which computing correlations, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
4. -w, --window: sliding time window to use for computations, either **daily**, **monthly** or **weekly** (OPTIONAL, defaults to daily).
5. -m, --method: method to compute correlations, either **pearson**, **kendall** or **spearman** (OPTIONAL, defaults to pearson).

### Data Split

```bash
python data_split.py -p "destination_path" -d "data_path" -v "avg_ohlc" -tr 0.8 -vd 0.1 -ts 0.1
```
Args:
1. -p, --path: destination folder for train, validation and test dataset storage (OPTIONAL, defaults to current directory).
2. -d, --data: path to original .csv file dataset (OPTIONAL, defaults to current directory).
3. -v, --variable: price variable to consider, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
4. -tr, --train: ratio for train set split (REQUIRED).
5. -vd, --valid: ratio for validation set split (REQUIRED).
6. -ts, --test: ratio for test set split (REQUIRED).

### Model Pretrain

```bash
python model_pretrain.py -p "destination_path" -tr "train_path" -vd "valid_path" -t "btc" -f "examples\features.json" -m "lstm" -c "examples\config_nn.json"
```
Args:
1. -p, --path: destination folder for pretrained model storage (OPTIONAL, defaults to current directory).
2. -tr, --train: path to the .csv file train dataset (OPTIONAL, defaults to current directory).
3. -vd, --valid: path to the .csv file validation dataset (OPTIONAL, defaults to current directory).
4. -t, --target: target coin to predict (REQUIRED).
5. -f, --features: path to .json file with list of coins to use as feature/predicting variables (OPTIONAL, defaults to all coins).
6. -m, --model: model to pretrain for inference, either **lstm**, **gru**, **xgboost**, **lightgbm** or **catboost** (REQUIRED).
7. -c, --config: path to .json file with list of configs to use for pretraining (REQUIRED).

### Model Forecast

```bash
python model_forecast.py -p "destination_path" -ts "test_path" -pt "pretrained_path" -t "btc" -f "examples\features.json" -m "lstm" -c "examples\config_nn.json"
```
Args:
1. -p, --path: destination folder for predictions storage (OPTIONAL, defaults to current directory).
2. -ts, --test: path to the .csv file test dataset (OPTIONAL, defaults to current directory).
3. -pt, --pretrained: path to the pretrained model (REQUIRED).
4. -t, --target: target coin to predict, **same as pretraining** (REQUIRED).
5. -f, --features: path to .json file with list of coins to use as feature/predicting variables, **same as pretraining** (OPTIONAL, defaults to all coins).
6. -m, --model: model to use for inference, **same as pretraining** (REQUIRED).
7. -c, --config: path to .json file with list of configs to use for prediction, **same as pretraining** (REQUIRED).

## Examples
You can find examples of coin list (for data pull), configs (for pretraining/forecast) and feature variable list (for pretraining/forecast) in the folder /examples.

## License
[MIT](https://choosealicense.com/licenses/mit/)