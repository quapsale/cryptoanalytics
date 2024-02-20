# Cryptocoins Analytics

Cryptocoins Analytics is a Python and R project for the analysis and forecasting of financial time series and cryptocurrency price trends.

## Project Structure
This repository is organized as it follows:
<li><b>Analysis:</b> collection of scripts designed to:</li>
<ul type = "square">
<li> Study correlation patterns among cryptocurrencies and generate representations in the form of correlograms.</li>
<li> Implement the Toda-Yamamoto procedure to test for Granger-causality between correlated cryptocoins.</li>
<li> Train and test SOTA machine learning models to forecast cryptocoin price series (namely GRU, LSTM, CatBoost, LightGBM and XGBoost).</li>
</ul>
<li><b>Data:</b> pre-built datasets adopted in the above-mentioned analyses, spanning 33 months from 20-02-2020 to 26-02-2023. </li>

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
<li> <i> Pasquale De Rosa, Pascal Felber and Valerio Schiavoni. 2023. Practical Forecasting of Cryptocoins Timeseries using Correlation Patterns. In: Proceedings of the 17th ACM International Conference on Distributed and Event-based Systems. DEBS 2023. https://doi.org/10.1145/3583678.3596888. </i></li>


<li> <i> Pasquale De Rosa and Valerio Schiavoni. 2022. Understanding Cryptocoins Trends Correlations. In: Distributed Applications and Interoperable Systems. DAIS 2022. https://doi.org/10.1007/978-3-031-16092-9_3. </i></li>

## License
[MIT](https://choosealicense.com/licenses/mit/)
