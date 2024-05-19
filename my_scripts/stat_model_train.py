# Functions for training/ forecasting statistical models

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Recursive forecasts

def recursive_forc_ma(df: pd.DataFrame,
                        train_len: int,
                        horizon: int,
                        window: int):
    '''
    Forecast timeseries using a moving average model. The MA model is set to the same order
    as the length of window

            Parameters:
                    df (pd.DataFrame): Dataframe containing training and testing data combined
                    train_len (int): How many rows are considered training data
                    horizon (int): How long of a horizon to forecast
                    window (int): Number of points in a single rolling window

            Returns:
                    pred_mv (Array): Predicted values for the horizon
    '''
    total_len = train_len + horizon
    

    pred_mv = []
    for i in np.arange(train_len, total_len, window):
        # Train the model parameters on this subset of data
        model = SARIMAX(df[:i], order=(0,0,window))
        res = model.fit(disp=False)
        endpoint = i + window - 1

        # We can go over the horizon if the window is large,
        # The last batch doesn't necessarily have to return "window" values

        predictions = res.get_prediction(0,
                                           endpoint)
        # This object contains multiple attributes, the "pure" prediction is the mean
        last_val = predictions.predicted_mean.iloc[-window:]

        pred_mv.extend(last_val)
    
    if len(pred_mv) > horizon:
        pred_mv = pred_mv[:horizon]


    return pred_mv

def recursive_forc_ar(df: pd.DataFrame,
                        train_len: int,
                        horizon: int,
                        window: int):
    '''
    Forecast timeseries using a moving average model. The MA model is set to the same order
    as the length of window

            Parameters:
                    df (pd.DataFrame): Dataframe containing training and testing data combined
                    train_len (int): How many rows are considered training data
                    horizon (int): How long of a horizon to forecast
                    window (int): Number of points in a single rolling window

            Returns:
                    pred_mv (Array): Predicted values for the horizon
    '''
    total_len = train_len + horizon
    

    pred_ar = []
    for i in np.arange(train_len, total_len, window):
        # Train the model parameters on this subset of data
        model = SARIMAX(df[:i], order=(window,0,0))
        res = model.fit(disp=False)
        endpoint = i + window - 1

        # We can go over the horizon if the window is large,
        # The last batch doesn't necessarily have to return "window" values

        predictions = res.get_prediction(0,
                                           endpoint)
        # This object contains multiple attributes, the "pure" prediction is the mean
        last_val = predictions.predicted_mean.iloc[-window:]

        pred_ar.extend(last_val)
    
    if len(pred_ar) > horizon:
        pred_ar = pred_ar[:horizon]


    return pred_ar

def recursive_forc_last(df: pd.DataFrame,
                        train_len: int,
                        horizon: int,
                        window: int):
    '''
    Forecast timeseries by using "window" last values recursively

            Parameters:
                    df (pd.DataFrame): Dataframe containing training and testing data combined
                    train_len (int): How many rows are considered training data
                    horizon (int): How long of a horizon to forecast
                    window (int): Number of points in a single rolling window

            Returns:
                    pred_mv (Array): Predicted values for the horizon
    '''
    total_len = train_len + horizon

    pred_last_value = []
    for i in np.arange(train_len, total_len, window):
        endpoint = i -1 + window
        # The very last datapoint can't be included because that's the last value for (total_len + 1) th point!
        if (endpoint > (total_len - 1)):
            endpoint = total_len - 1 # equal to last index in data. iloc is exclusive - won't be included
        last_val = df.iloc[(i -1):endpoint].Value
        pred_last_value.extend(last_val)
    
    return pred_last_value

def recursive_forc_mean(df: pd.DataFrame,
                        train_len: int,
                        horizon: int,
                        window: int):
    '''
    Forecast timeseries by using the mean at given points. Rolling - after window observations, take new data and change the mean to that.

            Parameters:
                    df (pd.DataFrame): Dataframe containing training and testing data combined
                    train_len (int): How many rows are considered training data
                    horizon (int): How long of a horizon to forecast
                    window (int): Number of points in a single rolling window

            Returns:
                    pred_mv (Array): Predicted values for the horizon
    '''
    total_len = train_len + horizon

    pred_mean = []
    for i in np.arange(train_len, total_len, window):
        endpoint = i -1 + window
        # The very last datapoint can't be included because that's the last value for (total_len + 1) th point!
        if (endpoint > (total_len - 1)):
            endpoint = total_len - 1 # equal to last index in data. iloc is exclusive - won't be included
        current_mean = np.mean(df.iloc[:endpoint].Value)
        # multiple values, length of window
        current_mean = np.tile(current_mean, window)
        pred_mean.extend(current_mean)
    

    if len(pred_mean) > horizon:
        pred_mean = pred_mean[:horizon]

    return pred_mean