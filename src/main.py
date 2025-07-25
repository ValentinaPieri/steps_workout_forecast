import pandas as pd
from preprocessing import preprocess_data
from compare_metrics import compare_metrics
from models.xgboost import xgboost_model
from models.sarima import sarima_model
from utils import plotting_models
from models.mlp import mlp_model
from utils import invert_boxcox

# Load dataset
dataset = pd.read_csv(
    'steps_workout_forecast/dataset/daily_steps - Copia.csv',
    usecols=['Date','Steps'],
    sep=';',
    decimal=',',
    parse_dates=['Date'],
    dayfirst=True                 
).set_index('Date')

dataset = dataset.asfreq('D')

ds, train_clean, test_clean, train_proc, test_proc, boxcox_lambda = preprocess_data(dataset)

look_back_list = [7, 14, 21, 28]

# Fit SARIMA model
"""
After running search_sarima_model inside sarima_model, the best seasonal period and orders are determined.
The SARIMA model is then fitted to the training data, and forecasts are generated for the test set.
The best model was decided based on the lowest AIC value.

Output:
Selected SARIMA(2, 1, 2) x (1, 0, 1, 7) (m=7) with lowest AIC=1443.11
"""
order = (2, 1, 2)
seasonal_order = (1, 0, 1, 7)
sarima_pred = sarima_model(train_proc, test_proc, None, boxcox_lambda, None, None)
plotting_models(train_clean, test_clean, train_proc, test_proc, sarima_pred, "SARIMA", boxcox_lambda)

# Fit MLP model
nn_mod, nn_pred, y_pred = mlp_model(train_proc, test_proc, None, boxcox_lambda)
plotting_models(train_clean, test_clean, train_proc, test_proc, nn_pred, "MLP", boxcox_lambda)

# Fit XGBoost model
"""
The XGBoost model is fitted to the training data, and forecasts are generated for the test set.
The model is trained using a grid search over hyperparameters and look_back values.
The best model is selected based on the lowest RMSE.
Output:
Selected best look_back: 14
   Best params: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.9}
   Achieved RMSE=2.9336
"""
best_xgboost_params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50, 'subsample': 0.9}
xgb_pred = xgboost_model(train_proc, test_proc, boxcox_lambda, None, None)
plotting_models(train_clean, test_clean, train_proc, test_proc, xgb_pred, "XGBoost", boxcox_lambda)


# Compare models using compare_metrics
test_inv = invert_boxcox(test_proc, boxcox_lambda)
best_model = compare_metrics(test_inv, sarima_pred, xgb_pred, nn_pred)
