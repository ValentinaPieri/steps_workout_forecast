import os
import pandas as pd
import numpy as np
from scipy import stats
from pykalman import KalmanFilter
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import boxcox


def preprocess_data(file_path):

   n = len(file_path)

   # Interpolate missing values
   ds = pd.Series(file_path['Steps'].interpolate().values.flatten())

   # split with train and test with train being 80% and test 20%
   train_size = int(len(ds) * 0.8)
   train = ds[:train_size]
   test = ds[train_size:]

   print(f"Dataset length: {len(ds)}")
   print(f"Training set length: {len(train)}")
   print(f"Test set length: {len(test)}")

   plt.figure(figsize=(10, 5)) 
   plt.plot(ds, label='Full Dataset', color='blue')
   plt.plot(train, label='Training Set', color='green')
   plt.plot(test, label='Test Set', color='red')
   plt.title('Daily Steps Dataset with Train, Validation, and Test Sets')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()

   # check for Outliers
   long_walks = outlier_detection(ds)

   #plot differences between train and outliers
   plt.figure(figsize=(10, 5))
   plt.plot(ds, label='Train Data', color='blue')
   plt.scatter(long_walks, ds[long_walks], color='red', label='Outliers')
   plt.title('Train Data with Outliers')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()

   # Check autocorrelation
   check_autocorrelation(train)

   # Check stationarity
   print("Stationarity Test:")
   stationarity_test(train, m=28)

   # Check normality
   print("Normality Test:")
   normality_test(train)

   # do seasonal decomposition
   result = seasonal_decompose(train, model='additive', period=28)
   result.plot()
   plt.show()

   train_processed, test_processed, boxcox_lambda = box_cox_transform(train, test)

   # Check stationarity after Box-Cox transformation
   print("Stationarity Test after Box-Cox Transformation:")
   stationarity_test(train_processed, m=28)
   # Check normality after Box-Cox transformation
   print("Normality Test after Box-Cox Transformation:")
   normality_test(train_processed)

   # Apply Kalman filter
   train_kf = kalman_filter(train_processed)
   test_kf = kalman_filter(test_processed)

   # Plot the results
   plt.figure(figsize=(15, 10))
   plt.plot(train.index, train_processed, label='Train Set (Box-Cox)', color='green')
   plt.plot(test.index, test_processed, label='Test Set (Box-Cox)', color='red')
   plt.plot(train.index, train_kf, label='Train Set (Kalman Filter)', color='orange')
   plt.plot(test.index, test_kf, label='Test Set (Kalman Filter)', color='purple')
   plt.title('Processed Train and Test Sets with Kalman Filter')
   plt.xlabel('Days')
   plt.ylabel('Steps')
   plt.legend()
   plt.show()
   
   # Check normality of Kalman filtered data
   print("Normality Test after Kalman Filter:")
   normality_test(train_kf)

   return ds, train, test


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

   # Outliers are not deleted from the dataset, but their indices are returned to be used later
   if num_outliers > 0:
      print(f"Valori outlier: {df[outliers]}")

   return np.where(outliers)[0].tolist()


def stationarity_test(ds, m=29):
   adf_result = adfuller(ds)
   print(f'ADF Statistic: {adf_result[0]}')
   print(f'p-value: {adf_result[1]}')
   if adf_result[1] < 0.05:
        print("The series is stationary (p < 0.05)")
   else:
        print("The series is non-stationary (p >= 0.05)")

def normality_test(ds):
   shapiro_test = stats.shapiro(ds)
   print(f"Shapiro-Wilk Test: Statistic={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
   if shapiro_test.pvalue > 0.05:
       print("Time series is normal (p >= 0.05)")
   else:
       print("Time series is not normal (p < 0.05)")

def check_autocorrelation(ds):
   """
   Plot the autocorrelation function of a time series.
   
   Parameters:
   ds (array-like): Time series data.
   """
   plot_acf(ds, lags=84)
   plt.title('Autocorrelation Function')
   plt.xlabel('Lags - Number of Days')
   plt.ylabel('Autocorrelation')
   plt.grid()
   plt.show()

def box_cox_transform(train, test):
   train_positive = train + 1e-6 if (train <= 0).any() else train
   train_transformed, lmbda = boxcox(train_positive)
   boxcox_lambda = lmbda

   # Apply to all sets using the same lambda
   test_positive = test + 1e-6 if (test <= 0).any() else test

   train_transformed = pd.Series(train_transformed, index=train.index)
   test_transformed = pd.Series(boxcox(test_positive, lmbda=lmbda), index=test.index)

   print(f"✓ Applied Box-Cox transformation (lambda={lmbda:.4f})")

   return train_transformed, test_transformed, boxcox_lambda

def kalman_filter(ds):
   """
   Apply Kalman filter to a time series.
   
   Parameters:
   ds (array-like): Time series data.
   
   Returns:
   np.ndarray: Filtered state means.
   """

   measurements = ds.to_numpy()
   kf = KalmanFilter(transition_matrices=[1],
                     observation_matrices=[1],
                     initial_state_mean=measurements[0],
                     initial_state_covariance=1,
                     observation_covariance=10,
                     transition_covariance=10) 
   state_means, _ = kf.filter(measurements)
   return state_means