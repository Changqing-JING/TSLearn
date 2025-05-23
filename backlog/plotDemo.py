from __future__ import absolute_import, division, print_function
import sys
import os
import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.float_format", lambda x: "%.5f" % x)
np.set_printoptions(precision=5, suppress=True)
pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
sns.set(style="ticks", context="poster")

data = [10930, 10318, 10595, 10972, 7706, 6756, 9092, 10551, 9722, 10913, 11151, 8186, 6422,
        6337, 11649, 11652, 10310, 12043, 7937, 6476, 9662, 9570, 9981, 9331, 9449, 6773, 6304,
        9355, 10477, 10148, 10395, 11261, 8713, 7299, 10424, 10795, 11069, 11602, 11427, 9095, 7707,
        10767, 12136, 12812, 12006, 12528, 10329, 7818, 11719, 11683, 12603, 11495, 13670, 11337, 10232,
        13261, 13230, 15535, 16837, 19598, 14823, 11622, 19391, 18177, 19994, 14723, 15694, 13248,
        9543, 12872, 13101, 15053, 12619, 13749, 10228, 9725, 14729, 12518, 14564, 15085, 14722,
        11999, 9390, 13481, 14795, 15845, 15271, 14686, 11054, 10395]

sentiment_short = data

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(sentiment_short, lags = 20, ax=ax1)
ax1.xaxis.set_ticks_position("bottom")
fig.tight_layout()

ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(sentiment_short, lags = 20, ax=ax2)
ax2.xaxis.set_ticks_position("bottom")
fig.tight_layout()
fig.show()
plt.show()