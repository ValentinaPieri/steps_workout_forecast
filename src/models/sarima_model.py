import pmdarima as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from steps_workout_forecast.src.utils import accuracy_metrics

def sarima_model(train, test, m=7):
    """
    Fit SARIMA model on train data and forecast on test data.
    
    Parameters:
    - train: pd.Series of training data
    - test: pd.Series of test data
    - m: seasonal period (default=28 for monthly/weekly cycles)
    """
    model = pm.auto_arima(
        train,
        start_p=1,
        start_q=1,
        test='adf',
        max_p=3,
        max_q=3,
        m=m,
        start_P=0,
        seasonal=True,
        d=None,
        D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    # Forecasting
    forecast = model.predict(n_periods=len(test))
    y_pred = model.predict_in_sample()

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(train, label='Train', linewidth=2)
    plt.plot(test.index, test, label='Test', linewidth=2, color='red')
    plt.plot(test.index, forecast, label='Forecast', linestyle='--', color='purple')
    plt.title('SARIMA Forecast')
    plt.xlabel('Time')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluation
    print("Forecast Accuracy Metrics:")
    metrics = accuracy_metrics(forecast, test)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return forecast
