# CryptoAnalytics

CryptoAnalytics is a Python and R project for the analysis and forecasting of financial time series and cryptocurrency price trends.

## Project Structure
This repository is organized as it follows:
<li><b>Analysis:</b> collection of scripts designed to:</li>
<ul type = "square">
<li> Study correlation patterns among cryptocurrencies and generate representations in the form of correlograms.</li>
<li> Implement the Toda-Yamamoto procedure to test for Granger-causality between correlated cryptocoins.</li>
<li> Train and test SOTA machine learning models to forecast cryptocoin price series (namely GRU, LSTM, CatBoost, LightGBM and XGBoost).</li>
</ul>
<li><b>Data:</b> pre-built datasets adopted in the above-mentioned analyses. </li>

## Data
The data sources used to gather information about cryptocurrency trends are [CoinMarketCap](https://www.coinmarketcap.com/) and [Binance](https://www.binance.com/).
The two pre-built datasets (coinmarketcap.csv and binance.csv) are available in a compressed .zip format.

## Getting Started
The Python version used in this project is 3.9. The R version is 3.6. A list of the external Python libraries/dependencies can be found in the file requirements.txt.

## Authors
<b>Pasquale De Rosa</b>, University of Neuchâtel, [pasquale.derosa@unine.ch](mailto:pasquale.derosa@unine.ch). <br/>
Pascal Felber, University of Neuchâtel, [pascal.felber@unine.ch](mailto:pascal.felber@unine.ch). <br/>
Valerio Schiavoni, University of Neuchâtel, [valerio.schiavoni@unine.ch](mailto:valerio.schiavoni@unine.ch).

## References
<li> <i> H. Y. Toda and T. Yamamoto, “Statistical inference in vector autoregressions with possibly integrated processes,” Journal of Econometrics,
vol. 66, no. 1, pp. 225–250, 1995. </i></li>

<li> <i> K. Cho, B. van Merriënboer, D. Bahdanau, and Y. Bengio, “On the
properties of neural machine translation: Encoder–decoder approaches,”
in Proceedings of SSST-8. </i></li>

<li> <i> S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural
Comput., vol. 9, no. 8, p. 1735–1780, nov 1997.</i></li>

<li> <i> L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin,
“CatBoost: Unbiased Boosting with Categorical Features,” in NIPS’18. </i></li>

<li> <i> G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y.
Liu, “LightGBM: A Highly Efficient Gradient Boosting Decision Tree,”
in NIPS’17. </i></li>

<li> <i> T. Chen and C. Guestrin, “XGBoost: A Scalable Tree Boosting System,”
in ACM SIGKDD 2016. </i></li>

## License
[MIT](https://choosealicense.com/licenses/mit/)
