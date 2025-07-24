import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def quick_accuracy_metrics(actual, forecast):
    print("\nTest Evaluation on Original Scale:")
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    mae = mean_absolute_error(actual, forecast)
    mape = mean_absolute_percentage_error(actual, forecast) * 100
    print(f"RMSE: {rmse:.4f} steps, MAE: {mae:.4f} steps, MAPE: {mape:.2f}%")

def accuracy_metrics(actual, forecast, model_name=None):
    """
    Calculate various accuracy metrics between actual and forecasted values.
    Parameters:
    - actual: array-like, actual values
    - forecast: array-like, forecasted values
    Returns:
    - me: Mean Error
    - mae: Mean Absolute Error
    - mpe: Mean Percentage Error
    - rmse: Root Mean Squared Error
    - corr: Correlation Coefficient
    - mape: Mean Absolute Percentage Error
    """
    actual = np.array(actual).flatten()
    forecast = np.array(forecast).flatten()
    me   = np.mean(forecast - actual)           # ME
    mae  = np.mean(np.abs(forecast - actual))   # MAE
    mpe  = np.mean((forecast - actual)/actual)  # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]
    mape = (np.mean(np.abs(forecast - actual)/np.abs(actual)))* 100  # MAPE in percentage

    print(f"{model_name} accuracy metrics: \n"
      f"ME: {me:.4f}, \n"
      f"MAE: {mae:.4f}, \n"
      f"MPE: {mpe:.4f}, \n"
      f"RMSE: {rmse:.4f}, \n"
      f"Correlation: {corr:.4f}, \n"
      f"MAPE: {mape:.2f}%")

    return {
        'ME': me,
        'MAE': mae,
        'MPE': mpe,
        'RMSE': rmse,
        'Correlation': corr,
        'MAPE': mape
    }

def create_dataset(dataset, look_back):
    """
    Create a dataset with look_back time steps.
    Parameters:
    - dataset: array-like, input data
    - look_back: int, number of previous time steps to consider

    Returns:
    - dataX: array-like, input features
    - dataY: array-like, target values
    """
    dataX, dataY = [], []
    for i in range(int(len(dataset)) - int(look_back)):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

def invert_boxcox(value, lam):
    """
    Invert Box-Cox transformation.
    Parameters:
    - value: array-like, transformed values
    - lam: float, lambda value used in Box-Cox transformation

    Returns:
    - Inverted values
    """
    # Handle arrays and scalars
    arr = np.array(value)
    if lam == 0:
        inv = np.exp(arr)
    else:
        inv = np.power(lam * arr + 1, 1 / lam)
    # Preserve pandas Series index
    if hasattr(value, 'index'):
        return pd.Series(inv, index=value.index)
    return inv

def plotting_models(train, test, train_proc, test_proc, forecast, model_name, boxcox_lambda=None):
   """
   Plot the training, test, and forecast data.
   Parameters:
   - train: pd.Series of training data
   - test: pd.Series of test data
   - forecast: pd.Series of forecasted values
   - model_name: str, name of the model for the plot title
   """
   if boxcox_lambda is not None:
       train_proc = invert_boxcox(train_proc, boxcox_lambda)
       test_proc = invert_boxcox(test_proc, boxcox_lambda)

   plt.figure(figsize=(12, 6))
   plt.plot(train.index, train, label='Train', color='blue', linewidth=2)
   plt.plot(test.index, test, label='Test', color='orange', linewidth=2)
   plt.plot(train.index, train_proc, label='Train', color='lightblue')
   plt.plot(test.index, test_proc, label='Test', color='red')
   plt.plot(forecast.index, forecast, label='Forecast', color='purple', linestyle='--', linewidth=2)
   plt.title('' + model_name + ' Forecast')
   plt.xlabel('Time')
   plt.ylabel('Steps')
   plt.legend()
   plt.grid(True)
   plt.show()