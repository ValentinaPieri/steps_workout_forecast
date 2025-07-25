# Report on Daily Steps Forecasting Results - steps_workout_forecast_analysis

This project presents the results of the analysis of the forecasting of a time series on daily activity metrics, comparing three different models: SARIMA, MLP, and XGBoost.

## Preprocessing on the Dataset
- **Missing Values:** Handled by filling with previous values
- **Outlier Detection:** Identified long walks as outliers
- **Data Splitting:** 28 days for testing, remaining for training
- **Transformation Method:** Box-Cox transformation for variance stabilization
- **Smoothing:** Kalman filter applied for trend extraction
- **Scaling:** Applied only for MLP model (Standard scaler)

## Models and Results

### 1. SARIMA

**Performance Metrics:**
| Metric | Test |
|--------|------|
| RMSE (steps) | 2238.9532 |
| MAE (steps) | 1539.8650 |
| MAPE (%) | 27.93 |

### 2. MLP (Multi-Layer Perceptron)

**Performance Metrics:**
| Metric | Test |
|--------|------|
| RMSE (steps) | 2914.3375 |
| MAE (steps) | 1879.4926 |
| MAPE (%) | 29.55 |

### 3. XGBoost

**Performance Metrics:**
| Metric | Test |
|--------|------|
| RMSE (steps) | 2330.7660 |
| MAE (steps) | 1430.4454 |
| MAPE (%) | 21.51 |

## Diebold-Mariano Test Results
The Diebold-Mariano test was conducted to compare the forecast accuracy of the models.
The null hypothesis states that the two forecasts have the same accuracy.
A p-value < 0.05 indicates a significant difference in forecast accuracy.

### DM Test Results
- SARIMA vs MLP: DM stat: -2.0773, p-value: 0.0474
- SARIMA vs XGBoost: DM stat: -0.2857, p-value: 0.7773
- MLP vs XGBoost: DM stat: 1.3420, p-value: 0.1908

The best model was **SARIMA** having the lowest RMSE of **2238.9532 steps**
