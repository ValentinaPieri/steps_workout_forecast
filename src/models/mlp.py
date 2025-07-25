import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterSampler
from utils import create_dataset, invert_boxcox, quick_accuracy_metrics

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0

    def __call__(self, current):
        if current + self.min_delta < self.best:
            self.best = current
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def scaling_data(train, val, test):
    """
    Scale training and test data using StandardScaler.
    Returns scaled train and test sets.
    """
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    val_scaled = scaler.transform(val.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()

    return train_scaled, val_scaled, test_scaled, scaler

def mlp_model(train, test, look_back_list, boxcox_lambda=None, best_params=None):
    """
    Fit a Multi-Layer Perceptron (MLP) model to the training data and predict on the test set.

    Parameters:
    - train: pd.Series, training data
    - test: pd.Series, test data
    - look_back_list: list, different look_back values to try
    - boxcox_lambda: float, lambda for Box-Cox transformation
    - best_params: dict, hyperparameters for the MLP model

    Returns:
    - preds: pd.Series, predictions on the test set
    """
    val_len = int(0.1 * len(train))
    val_to_scale = train[-val_len:]
    train_to_scale = train[:-val_len]

    train_scaled, val_scaled, test_scaled, scaler = scaling_data(train_to_scale, val_to_scale, test)

    if best_params is None:
        if look_back_list is None:
            look_back_list = [7, 14, 21, 28]
        best_params, best_lb = search_mlp_model(train_scaled, val_scaled, look_back_list, scaler, n_trials=30)

    look_back = best_lb

    # 2) Retrain on train+val
    scaled_all = np.concatenate([train_scaled, val_scaled])
    X_all, y_all = create_dataset(scaled_all, look_back)
    test_scaled = np.concatenate([scaled_all[-look_back:], test_scaled])
    X_test, y_test = create_dataset(test_scaled, look_back)

    X_all_t = torch.FloatTensor(X_all)
    y_all_t = torch.FloatTensor(y_all).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    dims = [look_back] + [best_params['hidden_size']]*best_params['n_layers'] + [1]
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)

    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    loss_fn   = nn.MSELoss()
    model.train()
    for _ in range(best_params['epochs']):
        perm = torch.randperm(len(X_all_t))
        for i in range(0, len(perm), best_params['batch_size']):
            idx = perm[i:i+best_params['batch_size']]
            xb, yb = X_all_t[idx], y_all_t[idx]
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    preds_t = []
    with torch.no_grad():
        preds_t = model(X_test_t).squeeze().numpy()
    preds = scaler.inverse_transform(preds_t.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    if boxcox_lambda is not None:
        preds = invert_boxcox(preds, boxcox_lambda)
        y_true = invert_boxcox(y_true, boxcox_lambda)

    # Ensure indices match
    if len(test.index) < len(preds):
        preds = preds[:len(test.index)]
    elif len(test.index) > len(preds):
        y_true = y_true[:len(test.index)]

    quick_accuracy_metrics(y_true, preds)

    return pd.Series(preds, index=test.index[-len(preds):])

def search_mlp_model(train, val, look_back_list, scaler, param_grid=None, n_trials=50):
    """
    Search for the best MLP hyperparameters.

    Parameters:
    - train: pd.Series, training data
    - val: pd.Series, validation data
    - look_back_list: list, different look_back values to try
    - scaler: StandardScaler, scaler used for the data
    - param_grid: dict, hyperparameters to search
    - n_trials: int, number of trials for hyperparameter search

    Returns:
    - best_params: dict, best hyperparameters found
    - best_look_back: int, best look_back value found
    """
    print("\nSearching best MLP hyperparameters...")

    if param_grid is None:
        param_grid = {
            'n_layers': [2, 3],
            'hidden_size': [8, 32, 56, 64],
            'lr': [1e-1, 1e-2, 1e-3],
            'epochs': [50, 100, 200],
            'batch_size': [16, 32, 64]
        }

    sampler = list(ParameterSampler(param_grid, n_iter=n_trials, random_state=42))

    best = {
        'rmse': float('inf'),
        'params': None,
        'look_back': None
    }

    for trial in sampler:
        lr = trial['lr']
        bs = trial['batch_size']
        n_layers = trial['n_layers']
        hidden_size = trial['hidden_size']
        epochs = trial['epochs']

        for look_back in look_back_list:
            X_tr, y_tr = create_dataset(train, look_back)
            X_v, y_v = create_dataset(val, look_back)

            X_tr_t = torch.FloatTensor(X_tr)
            y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1)
            X_v_t = torch.FloatTensor(X_v)
            y_v_t = torch.FloatTensor(y_v).unsqueeze(1)

            # Define model
            dims = [look_back] + [hidden_size]*n_layers + [1]
            layers = []
            for i in range(len(dims)-1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims)-2:
                    layers.append(nn.ReLU())
            model = nn.Sequential(*layers)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            early = EarlyStopper(patience=10, min_delta=1e-4)
            val_rmse = None

            for ep in range(1, epochs+1):
                model.train()
                # Shuffle train batch
                perm = torch.randperm(len(X_tr_t))
                for i in range(0, len(perm), bs):
                    idx = perm[i:i+bs]
                    xb, yb = X_tr_t[idx], y_tr_t[idx]
                    optimizer.zero_grad()
                    loss_fn(model(xb), yb).backward()
                    optimizer.step()

                # Validate
                model.eval()
                with torch.no_grad():
                    preds_v = model(X_v_t).squeeze().numpy()
                preds_v = scaler.inverse_transform(preds_v.reshape(-1, 1)).flatten()
                y_v_inv = scaler.inverse_transform(y_v.reshape(-1, 1)).flatten()
                rmse_v = np.sqrt(mean_squared_error(y_v_inv, preds_v))

                if early(rmse_v):
                    break  # early stopping

                val_rmse = rmse_v

            # Track best
            if val_rmse is not None and val_rmse < best['rmse']:
                best.update({
                    'rmse': val_rmse,
                    'params': trial,
                    'look_back': look_back
                })

    print("🔍 Best config:", best['params'], "look_back:", best['look_back'], f"RMSE={best['rmse']:.4f}")
    return best['params'], best['look_back']
