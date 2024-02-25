# 1. Import the following libraries: 

# pandas (for data manipulation)
import pandas as pd
# numpy (for numerical operations)
import numpy as np
# matplotlib (for data visualization)
import matplotlib.pyplot as plt 
# statsmodels (to use ARIMA)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# scikit-learn (for error metrics)
from sklearn.metrics import mean_squared_error
# !pip install alpha_vantage
from alpha_vantage.timeseries import TimeSeries

#Free API key
key = 'Y0YSKSFBF1079669'

import app

def predictNextDayClose(stock):
    # Create TimeSeries object
    ts = TimeSeries(key, output_format='pandas') #requires input of key and output format (default is json)

    #objects come with a data and a meta component (will likely only need data component), require a stock symbol + interval + output size
    predata, meta = ts.get_daily(stock, outputsize='full')
    data = predata[:1000]


    # 3. Modify the dataset

    # Keep relevant columns - closing price and date
    closing_data = data['4. close']

    # 4. Separate the data into training and testing datasets

    # Create a DataFrame using the closing price data
    df = pd.DataFrame(closing_data)

    # Split data into 70-30 training-testing data sets
    # Sort df by ascending order (most recent date comes last)
    df.sort_index(ascending=True, inplace=True)

    # train_size represents the first 90% of all rows; use this index to slice the data
    train_size = int(len(df) * 0.90)

    # Training data will include all rows starting from the first row up to the row indicated by 'train_size'
    train_data = df.iloc[:train_size]
    # The rest 10% will be included in testing data
    test_data = df.iloc[train_size:]

    # 5.2 Conduct the ADF test and determine the value for d 
    # Conduct the ADF test, which determines if the data is stationary or not
    result = adfuller(df)
    # Keeps track of the number of times differencing was needed 
    d = 0

    # If the data is not stationary, differencing is needed to convert it into stationary data
    # The p-value has to be less than 0.05 for the dataset to be considered stationary 
    while True:
        if result[1] < 0.05:
            break
        else:
            result = adfuller(df.diff().dropna())
            d += 1


    df.dropna()
    len(df.dropna())

    import warnings
    warnings.filterwarnings("ignore")

    # Defines ranges for p and q
    p = q = range(0, 4)

    # List to store all pdq combinations
    pdq = []
    # Appends each combination of p,d,q to the list
    for p_value in p:
        for q_value in q:
            pdq.append((p_value, d, q_value))

    # Initialize best aic to start at infinity
    best_aic = float('inf')  
    # Initialize best pqd combination 
    best_pdq = None

    # Test each combination 
    for combination in pdq:
        # Fit the arima model 
        model = ARIMA(df, order=(combination))
        model_fit = model.fit()
    
        # If the aic value is lower than the current best aic, it will be replaced
        if model_fit.aic < best_aic:
            best_aic = model_fit.aic
            best_pdq = combination

    # Extract the p and q values
    p = best_pdq[0]
    q = best_pdq[2]

    # 6.2 Create the ARIMA model with the p, d, q values determined
    model = ARIMA(df, order=(p,d,q))
    model_fit = model.fit()

    # 6.3 Train the ARIMA model with the training data and the p, d, q values determined
    model = ARIMA(train_data['4. close'], order=(p,d,q))
    model_fit = model.fit()

    # 7. Make out-of-sample predictions (forecasting)

    # fit the model to the training data
    model = ARIMA(train_data, order=(p,d,q))
    model_fit = model.fit()

    # make future predictions for the next week
    forecast_steps = 7
    # use steps parameter for # of future steps to forecast
    forecast = model_fit.forecast(steps=forecast_steps)

    test_data.dropna()

    # 8. Make a prediction for the closing price of the next day

    # Fit the model to all the data
    model = ARIMA(df, order=(p,d,q))
    model_fit = model.fit()

    # Make prediction for the next day
    forecast = model_fit.forecast(steps=1)
    return forecast.iloc[0]