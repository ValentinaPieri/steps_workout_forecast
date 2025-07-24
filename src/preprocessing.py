from math import log,exp
import pandas as pd
import numpy as np
from scipy import stats
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox


def preprocess_data(series_df):

   n = len(series_df)

   #check if there are missing values
   if series_df.isnull().values.any():
      print("Missing values detected in the dataset.")
      ds = fill_missing_values(series_df['Steps'])
   else:
      print("No missing values detected in the dataset.")
      ds = series_df['Steps']

   # check for Outliers
   long_walks = outlier_detection(ds)

   # Plot the ds with outliers
   plt.figure(figsize=(10, 5))
   plt.plot(ds, label='Original Data', color='blue', linewidth=2)
   plt.scatter(ds.index[long_walks], ds.iloc[long_walks], color='red', label='Outliers')
   plt.title('Daily Steps Dataset with Outliers')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()

   n_test = 28
   train = ds[:-n_test]
   test = ds[-n_test:]

   print(f"Dataset length: {len(ds)}")
   print(f"Training set length: {len(train)}")
   print(f"Test set length: {len(test)}")

   plt.figure(figsize=(10, 5))
   plt.plot(ds, label='Full Dataset', color='blue', linewidth=2)
   plt.plot(train, label='Training Set', color='green')
   plt.plot(test, label='Test Set', color='red')
   plt.title('Daily Steps Dataset with Train, Validation, and Test Sets')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()

   # Check if the series is stationary
   result  = seasonal_decompose(train, model='multiplicative', period=7, extrapolate_trend='freq')
   result.plot()
   plt.show()
   
   # Check stationarity
   print("Stationarity Test:")
   stationarity_test(train, m=7)

   # Check normality
   print("Normality Test:")
   normality_test(train)

   train_boxcox, boxcox_lambda = fit_boxcox(train)
   test_boxcox = apply_boxcox(test, boxcox_lambda)

   # Check autocorrelation
   print("\nAutocorrelation Check:")
   check_autocorrelation(train)
   check_autocorrelation(train_boxcox)

   print(f"\nApplying Box-Cox transformation...")
   # Check stationarity after Box-Cox transformation
   print("Stationarity Test after Box-Cox Transformation:")
   stationarity_test(train_boxcox, m=7)
   # Check normality after Box-Cox transformation
   print("Normality Test after Box-Cox Transformation:")
   normality_test(train_boxcox)

   print(f"\nApplying Kalman Filter...")
   # Apply Kalman filter
   kf_model   = fit_kalman(train_boxcox)
   train_kf   = apply_kalman(kf_model, train_boxcox)
   test_kf    = apply_kalman(kf_model, test_boxcox)

   print("Stationarity Test after Kalman Filter:")
   stationarity_test(train_kf, m=7)

   print("Normality Test after Kalman Filter:")
   normality_test(train_kf)

   result  = seasonal_decompose(train_kf, model='multiplicative', period=7, extrapolate_trend='freq')
   result.plot()
   plt.show()

   #plot kalman filter results
   plt.figure(figsize=(10, 5))
   plt.plot(train.index, train_boxcox, label='Original Train', color='blue', linewidth=2)
   plt.plot(train.index, train_kf, label='Kalman Filtered Train', color='green')
   plt.plot(test.index, test_boxcox, label='Original Test', color='red')
   plt.plot(test.index, test_kf, label='Kalman Filtered Test', color='orange')  
   plt.title('Daily Steps Dataset with Kalman Filter')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()

   return ds, train, test, train_kf, test_kf, boxcox_lambda

def fill_missing_values(df):
   df_prefilled = df.interpolate(method='linear', limit_direction='both')

   # Compare models to decide additive or multiplicative
   add = seasonal_decompose(df_prefilled, model='additive', period=7, extrapolate_trend='freq')
   mul = seasonal_decompose(df_prefilled, model='multiplicative', period=7, extrapolate_trend='freq')
   add_var = np.nanvar(add.seasonal) + np.nanvar(add.resid)
   mul_var = np.nanvar(mul.seasonal) + np.nanvar(mul.resid)

   model_type = 'multiplicative' if mul_var < add_var else 'additive'
   print("Using", model_type, "model for decomposition")   

   # Decompose with chosen model
   decomp = seasonal_decompose(df_prefilled, model=model_type, period=7, extrapolate_trend='freq')
   trend, season, resid = decomp.trend, decomp.seasonal, decomp.resid

   # Deseasonalize + interpolate original
   deseason = df - season  # preserves NaNs where original was missing
   deseason_interp = deseason.interpolate(method='linear')
   df_imputed = deseason_interp + season

   plt.figure(figsize=(12,6))
   df.plot(label='Original (with gaps)', marker='o')
   df_imputed.plot(label='Imputed', linestyle='--')
   plt.legend()
   plt.show()

   return df_imputed

def outlier_detection(df):
   """
   Detect outliers in a time series using the Hampel identifier.
   
   Parameters:
   ds (array-like): Time series data.
   
   Returns:
   list: Indices of detected outliers.
   """
   q1 = df.quantile(0.25)
   q3 = df.quantile(0.75)
   iqr = q3 - q1

   # 4. Define bounds and detect outliers
   lower = q1 - 1.5 * iqr
   upper = q3 + 1.5 * iqr
   outliers = (df < lower) | (df > upper)
   num_outliers = np.sum(outliers)

   print(f"Lower bound: {lower}, Upper bound: {upper}")
   print(f"Number of outliers detected: {num_outliers}")

   # Indices of outliers
   if num_outliers > 0:
      print(f"Valori outlier: {df[outliers]}")

   return np.where(outliers)[0].tolist()

def stationarity_test(ds, m=7):
   adf_result = adfuller(ds)
   print(f'ADF Statistic: {adf_result[0]}')
   print(f'p-value: {adf_result[1]}')
   if adf_result[1] < 0.05:
        print("The series is stationary (p < 0.05)\n")
   else:
        print("The series is non-stationary (p >= 0.05)\n")

def normality_test(ds):
   shapiro_test = stats.shapiro(ds)
   print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
   if shapiro_test.pvalue > 0.05:
       print("Time series is normal (p >= 0.05)\n")
   else:
       print("Time series is not normal (p < 0.05)\n")

def check_autocorrelation(ds):
   plot_acf(ds, lags=56, alpha=0.05)
   plt.title('Autocorrelation Function')
   plt.xlabel('Lags - Number of Days')
   plt.ylabel('Autocorrelation')
   plt.grid()
   plt.show()

def fit_boxcox(train):

   train_positive = train + 1e-6 if (train <= 0).any() else train
   train_boxcox, lmbda = boxcox(train_positive)

   print(f"✓ Fitted Box-Cox transformation on train (lambda={lmbda:.4f})")

   return pd.Series(train_boxcox, index=train.index), lmbda

def apply_boxcox(test, lmbda):
   test_positive = test + 1e-6 if (test <= 0).any() else test
   test_boxcox = boxcox(test_positive, lmbda=lmbda)

   print(f"✓ Applied Box-Cox transformation on test (lambda={lmbda:.4f})")

   return pd.Series(test_boxcox, index=test.index)

def invert_boxcox(value, lam):
    if lam == 0:
        return np.exp(value)
    return np.power(lam * value + 1, 1 / lam)

def fit_kalman(train):
   kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=train.iloc[0],
        n_dim_state=1
    )
   
   return kf.em(train.values)

def apply_kalman(kf_fitted, series):
   
   state_means, _ = kf_fitted.smooth(series.values)

   return pd.Series(state_means.flatten(), index=series.index)