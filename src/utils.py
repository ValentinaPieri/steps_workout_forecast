import numpy as np


def accuracy_metrics(forecast, actual):
   mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
   me   = np.mean(forecast - actual)           # ME
   mae  = np.mean(np.abs(forecast - actual))   # MAE
   mpe  = np.mean((forecast - actual)/actual)  # MPE
   rmse = np.mean((forecast - actual)**2)**.5  # RMSE
   corr = np.corrcoef(forecast, actual)[0,1]   # correlation coeff
   return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse,'corr':corr})