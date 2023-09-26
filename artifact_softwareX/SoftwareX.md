# CryptoAnalytics

CryptoAnalytics is a software artifact for the analysis and forecasting of financial time series and cryptocurrency price trends.

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
pip install -r requirements.txt
```
## Demo
A demo for the price prediction flow of Bitcoin (BTC), using a set of predefined args, is available on CodeOcean.

The demo uses a pre-built dataset (using the data_pull command) with the following characteristics:

1. coins: "btc", "eth", "usdt", "usdc", "bnb", "xrp", "busd", "ada", "sol", "doge", "ltc".

2. start: 01-10-2020.

3. end: 01-10-2022.

To launch the demo, start a reproducible run on our CodeOcean [capsule](https://codeocean.com/capsule/7158163/tree).

## Structure
This artifact is organized as it follows:

### Price prediction flow

1. data_pull.py: command to generate a new dataset of OHLC cryptocoin prices from [CoinMarketCap](https://coinmarketcap.com/).
2. data_split.py: command to generate train, validation and test sets from the original data.
3. model_pretrain.py: command to pretrain ML models for cryptocoin prices forecast.
4. model_forecast.py: command to use the pretrained ML models to forecast cryptocoin prices.

### Extra

1. correlation_analysis.py: command to analyze correlations among cryptocoin prices, can be useful for pre-selection of feature variables.

## Usage

### Price prediction flow

#### Data Pull

```bash
python data_pull.py -p "destination_path" -f "filename" -c "examples\coins.json" -s "01-01-2020" -e "01-01-2022"
```
Args:
1. -p, --path: destination directory for dataset storage (OPTIONAL, defaults to current directory).
2. -f, --filename: file name for pulled dataset (OPTIONAL, defaults to dataset_coinmarketcap_START_END).
3. -c, --coins: path to .json file with list of coins to include in the dataset (OPTIONAL, defaults to top-20 coins for market cap).
4. -s, --start: starting date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to yesterday).
5. -e, --end: ending date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to today).

#### Data Split

```bash
python data_split.py -p "destination_path" -f "filename1" "filename2" "filename3" -d "data_path" -v "avg_ohlc" -tr 0.8 -vd 0.1 -ts 0.1
```
Args:
1. -p, --path: destination directory for train, validation and test dataset storage (OPTIONAL, defaults to current directory).
2. -f, --filenames: file names for pulled dataset (OPTIONAL, defaults to [train_TODAY, valid_TODAY, test_TODAY]).
3. -d, --data: path to original .csv file dataset (REQUIRED).
4. -v, --variable: price variable to consider, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
5. -tr, --train: ratio for train set split (REQUIRED).
6. -vd, --valid: ratio for validation set split (REQUIRED).
7. -ts, --test: ratio for test set split (REQUIRED).

#### Model Pretrain

```bash
python model_pretrain.py -p "destination_path" -f "filename" -tr "train_path" -vd "valid_path" -t "btc" -ft "examples\features.json" -m "lstm" -c "examples\config_nn.json"
```
Args:
1. -p, --path: destination directory for pretrained model storage (OPTIONAL, defaults to current directory).
2. -f, --filename: file name for pretrained model (OPTIONAL, defaults to model_TODAY).
3. -tr, --train: path to the .csv file train dataset (REQUIRED).
4. -vd, --valid: path to the .csv file validation dataset (REQUIRED).
5. -t, --target: target coin to predict (REQUIRED).
6. -ft, --features: path to .json file with list of coins to use as feature/predicting variables (OPTIONAL, defaults to all coins).
7. -m, --model: model to pretrain for inference, either **lstm**, **gru**, **xgboost**, **lightgbm** or **catboost** (REQUIRED).
8. -c, --config: path to .json file with list of configs to use for pretraining (REQUIRED).

#### Model Forecast

```bash
python model_forecast.py -p "destination_path" -f "filename" -ts "test_path" -pt "pretrained_path" -t "btc" -ft "examples\features.json" -m "lstm" -c "examples\config_nn.json"
```
Args:
1. -p, --path: destination directory for predictions storage (OPTIONAL, defaults to current directory).
2. -f, --filename: file name for predictions (OPTIONAL, defaults to predictions_model_TODAY).
3. -ts, --test: path to the .csv file test dataset (REQUIRED).
4. -pt, --pretrained: path to the pretrained model (REQUIRED).
5. -t, --target: target coin to predict, **same as pretraining** (REQUIRED).
6. -ft, --features: path to .json file with list of coins to use as feature/predicting variables, **same as pretraining** (OPTIONAL, defaults to all coins).
7. -m, --model: model to use for inference, **same as pretraining** (REQUIRED).
8. -c, --config: path to .json file with list of configs to use for prediction, **same as pretraining** (REQUIRED).

### Extra

#### Correlation Analysis

```bash
python correlation_analysis.py -p "destination_path" -f "filename" -d "data_path" -v "avg_ohlc" -w "daily" -m "pearson"
```
Args:
1. -p, --path: destination directory for correlations storage (OPTIONAL, defaults to current directory).
2. -f, --filename: file name for correlations dataset (OPTIONAL, defaults to correlations_TODAY).
3. -d, --data: path to .csv file dataset (REQUIRED).
4. -v, --variable: price variable on which computing correlations, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
5. -w, --window: sliding time window to use for computations, either **daily**, **monthly** or **weekly** (OPTIONAL, defaults to daily).
6. -m, --method: method to compute correlations, either **pearson**, **kendall** or **spearman** (OPTIONAL, defaults to pearson).

## Examples
You can find examples of coin list (for data pull), configs (for pretraining/forecast) and feature variable list (for pretraining/forecast) in the directory /examples.

## License
[MIT](https://choosealicense.com/licenses/mit/)