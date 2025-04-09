import pandas as pd
import numpy as np

# generate date series, periods means number of data. every one day, D means 1 day, 3D means 3 days
rng = pd.date_range("2016/07/01", periods=90, freq="D") 

# generate random series, index is rng
ts = pd.Series(np.random.randn(len(rng)), index=rng)  # 20 data

# sample by month, then calculate the sum of each month
res = ts.resample("ME").sum()
print(res)

# create a rolling window of 10 days
r = ts.rolling(window=10)

print(r.mean())  # rolling mean

