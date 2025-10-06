
from typing import Dict
import numpy as np
import pandas as pd
import mplfinance as mpf

import yfinance as yf
tickers = ["AAPL", "MSFT", "GOOG"]
# data = yf.download(tickers, start="2020-01-01")["Adj Close"]
data = yf.download(tickers, start="2025-10-04")

returns = data.pct_change().dropna()

print(returns.head())