# File: toda_yamamoto.R
# Description: Implementation of the Toda-Yamamoto (1995) procedure to test for Granger causality.
# File Created: 01/02/2023
# R Version: 3.6


# Imports
library(readr)
library(fUnitRoots)
library(urca)
library(vars)
library(aod)
library(zoo)
library(tseries)

# File properties
wd <- getwd()
data_path <- file.path(wd, 'data/binance_notest.csv')

# Set seed
set.seed(1234)

# Load data
data <- read_delim(data_path, delim = ',')
coins <- names(data[, 2:17])

# Transform into daily observations
daily_freq <- as.Date(cut(data$Date, 'day'))
new_data <- aggregate(DOGE ~ daily_freq, data, mean)
colnames(new_data) <- c('Date', 'XLM')

for (i in coins[2:16])
{ coin <- aggregate(data[[i]] ~ daily_freq, data, mean)
  colnames(coin) <- c('Date', i)
  new_data <- cbind(new_data, coin[2])
}

# Write output
sink('results/output.txt')

# ADF test
for (i in coins)
  { print(i)
    print(adf.test(new_data[[i]]))
}

# Rejected stationarity for all coins. Re-run with integration order=1
for (i in coins)
{ print(i)
  print(adf.test(diff(new_data[[i]], 1)))
}

# Accepted stationarity for all with order=1

# KPSS test
for (i in coins)
{ print(i)
  print(kpss.test(new_data[[i]]))
}

# Accepted non-stationarity for all coins. Re-run with integration order=1
for (i in coins)
{ print(i)
  print(kpss.test(diff(new_data[[i]], 1)))
}

# Rejected non-stationarity for all with order=1

# 1st order differentiation eliminates the unit root, then the maximum order of integration is 1.

# Set up VAR select using all the possible coin pairs to find optimal lag (lower AIC)
coins_subset <- coins[-c(3, 5)]
btc_lags <- c()
eth_lags <- c()

# BTC
for (i in coins_subset)
{ VAR_selec1 <- VARselect(new_data[c('BTC', i)], lag.max=40, type='both')
btc_lags <- append(btc_lags, VAR_selec1$selection[[1]])
}

# ETH
for (i in coins_subset)
{ VAR_selec2 <- VARselect(new_data[c('ETH', i)], lag.max=40, type='both')
eth_lags <- append(eth_lags, VAR_selec2$selection[[1]])
}

# Toda-Yamamoto test for each coin pair

# BTC
for (i in coins_subset)
{ index <- grep(i, coins_subset)
lag <- btc_lags[index]

# New optimal VAR with 1 additional lag
V <- VAR(new_data[c('BTC', i)], p=lag+1, type='both')

# Wald-test 1 (H0: alt-coin does not Granger-cause BTC)
waldtest1 <- wald.test(b=coef(V$varresult[[1]]), Sigma=vcov(V$varresult[[1]]), Terms=c(seq(2, by=2, length=lag)))
cat(i, 'does not Granger-cause BTC', '\n')
print(waldtest1$result)
}

# ETH
for (i in coins_subset)
{ index <- grep(i, coins_subset)
lag <- eth_lags[index]

# New optimal VAR with 1 additional lag
V <- VAR(new_data[c('ETH', i)], p=lag+1, type='both')

# Wald-test 1 (H0: alt-coin does not Granger-cause ETH)
waldtest1 <- wald.test(b=coef(V$varresult[[1]]), Sigma=vcov(V$varresult[[1]]), Terms=c(seq(2, by=2, length=lag)))
cat(i, 'does not Granger-cause ETH', '\n')
print(waldtest1$result)
}

# Stop writing output
sink()
