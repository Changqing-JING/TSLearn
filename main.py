from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

stockFile = "data/T10yr.csv"

stock = pd.read_csv(stockFile, index_col=0, parse_dates=[0])

stock_week = stock["Close"].resample("W-MON").mean()
stock_train = stock_week["2000":"2015"]

stock_train.plot(figsize=(12, 8))
plt.legend(bbox_to_anchor=(1.25, 0.5))
plt.title("Stock Close")
sns.despine()
plt.show(block=False)

# The graph show data is not stable. Try diff
stock_diff = stock_train.diff().dropna()

# Create a new figure window for displaying the differenced data
plt.figure()
plt.plot(stock_diff)  # Plot the differenced data (changes between consecutive time points)
plt.title("Diff 1")  # Add a title to the plot
plt.show(block=False)  # Display the plot but don't pause execution

# ACF (AutoCorrelation Function) shows how a data point correlates with previous data points
# This helps identify patterns and seasonality in the time series
acf = plot_acf(stock_diff, lags=20)  # Plot autocorrelation for up to 20 time periods back
plt.title("ACF")  # Add a title to the plot
acf.show()  # Display the ACF plot

# PACF (Partial AutoCorrelation Function) shows direct correlation between observations
# This helps determine how many AR (Auto-Regressive) terms to use in our model
pacf = plot_pacf(stock_diff, lags=20)  # Plot partial autocorrelation for up to 20 time periods back
plt.title("PACF")  # Add a title to the plot
pacf.show()  # Display the PACF plot

# ARIMA stands for AutoRegressive Integrated Moving Average
# It's a popular model for forecasting time series data
# order=(1,1,1) means: 1 AR term, 1 differencing step, 1 MA (Moving Average) term
model = ARIMA(stock_train, order=(1, 1, 1), freq="W-MON", trend="t")  # Create the ARIMA model
result = model.fit()  # Train the model on our data

# Now we use the trained model to make predictions
# Start date must be inside training data period to have a reference point
pred = result.predict("20140609", "20160701", dynamic=True)  # Predict values from June 2014 to July 2016
print(pred)  # Print the predicted values

# Create a new figure to compare predictions with actual training data
plt.figure(figsize=(6, 6))  # Create a square figure of size 6x6 inches
plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees for better readability
plt.plot(pred)  # Plot the predicted values
plt.plot(stock_train)  # Plot the actual training data for comparison
plt.title("pred vs train")  # Add a title to the plot
plt.show(block=False)  # Display the plot but don't pause execution

print("done")  # Print a message to indicate processing is complete

plt.show()  # This final call blocks execution and keeps all plots open until manually closed
            # Without this, the program would end and all plots would disappear