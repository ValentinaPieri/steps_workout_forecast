import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from xgboost import XGBRegressor
from steps_workout_forecast.src.utils import accuracy_metrics

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


def xgboost_model(ds, train, test):
    """
    Train an XGBoost model on the provided training data.
    Args:
        ds (pd.Series): The full dataset.
        train (pd.Series): The training set.
        test (pd.Series): The test set.

    Returns:
        model (XGBRegressor): The trained XGBoost model.
        forecast (list): The forecasted values for the test set.
    """
    
    lookback = 7
    len_test = len(test)

    x_train, y_train = create_dataset(train, lookback)

    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(x_train, y_train)

    # Doing lookback number of steps ahead forecast
    xinput = x_train[-1]
    forecast = []
    for i in range(len_test):
        forecast.append(model.predict(xinput.reshape(1, lookback))[0])
        xinput = np.roll(xinput, -1)
        xinput[-1] = forecast[-1]

    # Calculate accuracy metrics
    metrics = accuracy_metrics(forecast, test.values)
    print("Accuracy Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return model, forecast
