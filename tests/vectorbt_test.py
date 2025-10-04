import numpy as np
import pandas as pd
from datetime import datetime

import vectorbt as vbt


# Prepare data
start = '2025-01-01 UTC'  # crypto is in UTC
end = '2025-09-01 UTC'
btc_price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')

print(type(btc_price))
print(btc_price)
