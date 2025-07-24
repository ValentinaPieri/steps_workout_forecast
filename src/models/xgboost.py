import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from utils import create_dataset, invert_boxcox, quick_accuracy_metrics

def fit_xgboost_model(train_proc, test_proc, boxcox_lambda, look_back, best_params=None):

    if best_params is None:
        if look_back is None:
            look_back = [7, 14, 21, 28]
        best_params, look_back = xgboost_model_tuned(train_proc, look_back)
        print("Best configuration:", best_params)

    X_train, y_train = create_dataset(np.array(train_proc), look_back)
    test_proc_copy = np.concatenate([train_proc[-look_back:], test_proc])
    X_test, y_test = create_dataset(np.array(test_proc_copy), look_back)

    best_model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    preds=[]
    for i in range(len(y_test)):
        pred = best_model.predict(X_test[i].reshape(1, -1))
        preds.append(pred[0])

    preds_boxcox = pd.Series(preds, index=test_proc.index[-len(preds):])
    preds = invert_boxcox(preds_boxcox, boxcox_lambda)
    y_test = invert_boxcox(pd.Series(y_test), boxcox_lambda)

    quick_accuracy_metrics(y_test, preds)

    return preds

def xgboost_model_tuned(train, look_back_list):
    """
    For each look_back in look_back_list:
      - Build lagged dataset
      - GridSearchCV over XGB hyperparams
      - Record the best (neg-RMSE), estimator, and look_back
    Returns:
      best_estimator (fitted), best_look_back, best_score (positive RMSE)
    """
    param_grid = {
        'max_depth':    [3, 5, 7],
        'learning_rate':[0.01, 0.05, 0.1],
        'n_estimators': [50, 75, 100],
        'subsample':    [0.8, 0.9, 1.0]
    }

    best_rmse    = np.inf
    best_model   = None
    best_lb      = None
    best_params  = None

    for lb in look_back_list:
        print(f"\n→ Trying look_back = {lb:2d}")
        # prepare data
        X, y = create_dataset(np.array(train), lb)
        X = np.array(X).reshape(-1, lb)
        y = np.array(y).ravel()

        # time-series CV: you can customize n_splits if you like
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

        # recover positive RMSE
        rmse = -grid.best_score_
        print(f"RMSE={rmse:.4f}  params={grid.best_params_}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = grid.best_estimator_
            best_lb     = lb
            best_params = grid.best_params_

    print("\nSelected best look_back:", best_lb)
    print("   Best params:", best_params)
    print(f"   Achieved RMSE={best_rmse:.4f}\n")

    return best_params, best_lb