import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import preprocess_data
from models import xgboost_model, sarima_model

# Load dataset
dataset = pd.read_csv('project/daily_steps - Copia.csv', decimal=',', sep=';')

ds, train, test = preprocess_data(dataset)
