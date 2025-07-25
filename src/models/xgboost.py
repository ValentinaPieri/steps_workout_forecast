import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from utils import create_dataset, invert_boxcox, quick_accuracy_metrics

def xgboost_model(train, test, boxcox_lambda=None, look_back=None, best_params=None):
    """
    Fit an XGBoost model to the training data and predict on the test data.

    Parameters:
    - train: pd.Series, training data
    - test: pd.Series, test data
    - boxcox_lambda: float, lambda value used in Box-Cox transformation
    - look_back: int, number of previous time steps to consider
    - best_params: dict, optional, pre-defined hyperparameters for XGBoost
    
    Returns:
    - preds: pd.Series, predictions for the test set
    """
    if best_params is None:
        if look_back is None:
            look_back = [7, 14, 21, 28]
        best_params, look_back = search_xgboost_model(train, look_back)
        print("Best configuration:", best_params)

    X_train, y_train = create_dataset(np.array(train), look_back)
    test_proc_copy = np.concatenate([train[-look_back:], test])
    X_test, y_test = create_dataset(np.array(test_proc_copy), look_back)

    best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    preds=[]
    for i in range(len(y_test)):
        pred = best_model.predict(X_test[i].reshape(1, -1))
        preds.append(pred[0])

    preds_boxcox = pd.Series(preds, index=test.index[-len(preds):])
    if boxcox_lambda is not None:
        preds = invert_boxcox(preds_boxcox, boxcox_lambda)
        y_test = invert_boxcox(pd.Series(y_test), boxcox_lambda)
    else:
        preds = preds_boxcox
        y_test = pd.Series(y_test)

    quick_accuracy_metrics(y_test, preds)

    return preds

def search_xgboost_model(train, look_back_list):
    """
    Perform a grid search to find the best hyperparameters for the XGBoost model.

    Parameters:
    - train: pd.Series, training data
    - look_back_list: list of int, possible look_back values

    Returns:
    - best_params: dict of best hyperparameters
    - best_lb: int, best look_back value
    """
    param_grid = {
        'max_depth':    [3, 5, 7],
        'learning_rate':[0.01, 0.05, 0.1],
        'n_estimators': [50, 75, 100],
        'subsample':    [0.8, 0.9, 1.0]
    }

    best_rmse    = np.inf
    best_lb      = None
    best_params  = None

    for lb in look_back_list:
        print(f"\n→ Trying look_back = {lb:2d}")
        # prepare data
        X, y = create_dataset(np.array(train), lb)
        X = np.array(X).reshape(-1, lb)
        y = np.array(y).ravel()

        tscv = TimeSeriesSplit(n_splits=5)

        grid = GridSearchCV(
            XGBRegressor(objective='reg:squarederror', random_state=42),
            param_grid,
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            verbose=1,
            n_jobs=-1
        )
        grid.fit(X, y)

        rmse = -grid.best_score_
        print(f"RMSE={rmse:.4f}  params={grid.best_params_}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_lb     = lb
            best_params = grid.best_params_

    print("\nSelected best look_back:", best_lb)
    print("   Best params:", best_params)
    print(f"   Achieved RMSE={best_rmse:.4f}\n")

    return best_params, best_lb