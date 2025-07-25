import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils import invert_boxcox, quick_accuracy_metrics

def search_sarima_model(train, seasonal_m_list):
    """
    Search for the optimal SARIMA model parameters over multiple seasonal periods.

    Parameters:
    - train: pd.Series of training data
    - seasonal_m_list: list of ints, candidate seasonal periods

    Returns:
    - best_m:       int, chosen seasonal period
    - best_order:   tuple, (p,d,q)
    - best_seasonal tuple, (P,D,Q,m)
    """
    best_aic = float('inf')
    best_cfg = None

    for m in seasonal_m_list:
        print(f"\n→ Trying m = {m}")
        auto = pm.auto_arima(
            train,
            start_p=0, max_p=3,
            start_q=0, max_q=3,
            start_d=0, max_d=2,
            seasonal=True, m=m,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            start_D=0, max_D=1,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        aic = auto.aic()
        order = auto.order
        seas_order = auto.seasonal_order
        print(f"   SARIMA{order} x {seas_order} (m={m}) → AIC = {aic:.2f}")

        if aic < best_aic:
            best_aic = aic
            best_cfg = (m, order, seas_order)

    best_m, best_order, best_seasonal = best_cfg
    print(f"\nSelected SARIMA{best_order} x {best_seasonal} (m={best_m}) with lowest AIC={best_aic:.2f}")
    return best_m, best_order, best_seasonal


def sarima_model(train, test, seasonal_m_list, boxcox_lambda=None, order=None, seasonal_order=None):
    """
    Fits a SARIMA model by searching over seasonal_m_list, then forecasts the test set.

    Parameters:
    - train: pd.Series of training data
    - test:  pd.Series of test data
    - boxcox_lambda: float or None, for Box–Cox inversion
    - seasonal_m_list: list of ints, seasonal periods to try

    Returns:
    - pd.Series of forecasts (original scale)
    """
    # Search for the best SARIMA model if not provided
    if (order is None or seasonal_order is None):
        if seasonal_m_list is None:
            seasonal_m_list = [7, 14, 21, 28]
        best_m, order, seasonal_order = search_sarima_model(train, seasonal_m_list)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    print(f"Final model AIC: {fitted.aic:.2f}")

    fc_bc = fitted.get_forecast(len(test)).predicted_mean

    if boxcox_lambda is not None:
        forecast = invert_boxcox(fc_bc, boxcox_lambda)
        test_orig = invert_boxcox(test, boxcox_lambda)
    else:
        forecast = fc_bc
        test_orig = test

    quick_accuracy_metrics(test_orig, forecast)

    return forecast