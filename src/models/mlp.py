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

# def grid_search_mlp(train, val, val_original, look_back_list, scaler, param_grid=None):
#     """
#     Performs grid search over both look_back and MLP hyperparameters.
#     Returns best config including look_back.
#     """
#     if param_grid is None:
#         param_grid = {
#             'n_layers':    [2],
#             'hidden_size': [8, 32, 56, 64],
#             'lr':          [1e-1, 1e-2, 1e-3],
#             'epochs':      [50, 100, 200],
#             'batch_size':  [16, 32, 64]
#         }

#     best = {
#         'rmse': float('inf'),
#         'params': None,
#         'look_back': None,
#         'X_scaler': None,
#         'y_scaler': None
#     }

#     # Loop over look_back values
#     for look_back in look_back_list:
#         # prepare data
#         X_tr, y_tr = create_dataset(train, look_back)
#         X_v, y_v   = create_dataset(val, look_back)

#         # convert to tensors
#         X_tr_t = torch.FloatTensor(X_tr)
#         y_tr_t = torch.FloatTensor(y_tr).unsqueeze(1)
#         X_v_t  = torch.FloatTensor(X_v)
#         y_v_t  = torch.FloatTensor(y_v).unsqueeze(1)
    
#         # hyperparameter combinations
#         for n_layers, hidden_size, lr, epochs, batch_size in product(
#             param_grid['n_layers'],
#             param_grid['hidden_size'],
#             param_grid['lr'],
#             param_grid['epochs'],
#             param_grid['batch_size']
#         ):
#             # build MLP
#             dims = [look_back] + [hidden_size]*n_layers + [1]
#             layers = []
#             for i in range(len(dims)-1):
#                 layers.append(nn.Linear(dims[i], dims[i+1]))
#                 if i < len(dims)-2:
#                     layers.append(nn.ReLU())
#             model = nn.Sequential(*layers)

#             optimizer = optim.Adam(model.parameters(), lr=lr)
#             loss_fn   = nn.MSELoss()

#             # train
#             model.train()
#             for _ in range(epochs):
#                 perm = np.random.permutation(len(X_tr_t))
#                 for i in range(0, len(perm), batch_size):
#                     idx = perm[i:i+batch_size]
#                     xb, yb = X_tr_t[idx], y_tr_t[idx]
#                     optimizer.zero_grad()
#                     loss_fn(model(xb), yb).backward()
#                     optimizer.step()

#             # evaluate
#             model.eval()
#             with torch.no_grad():
#                 preds_v_s = model(X_v_t).numpy().flatten()
#             preds_v = scaler.inverse_transform(preds_v_s.reshape(-1,1)).flatten()
#             rmse_v  = np.sqrt(mean_squared_error(val_original.values[look_back:], preds_v))

#             print(f"[look_back={look_back}] params (layers={n_layers}, hid={hidden_size}, lr={lr},"
#                   f" epochs={epochs}, bs={batch_size}) → RMSE {rmse_v:.4f}")

#             if rmse_v < best['rmse']:
#                 best.update({
#                     'rmse': rmse_v,
#                     'params': {
#                         'n_layers':    n_layers,
#                         'hidden_size': hidden_size,
#                         'lr':          lr,
#                         'epochs':      epochs,
#                         'batch_size':  batch_size
#                     },
#                     'look_back': look_back
#                 })

#     print("🔍 Best look_back & params:", best['look_back'], best['params'], f"RMSE={best['rmse']:.4f}")
#     return best


def nn_model(train, test, look_back_list, boxcox_lambda=None, params=None):
    """
    Uses grid_search_mlp to find best look_back + params, then retrains on full train and forecasts test.
    """
    val_len = int(0.1 * len(train))
    val_to_scale = train[-val_len:]
    train_to_scale = train[:-val_len]

    train_scaled, val_scaled, test_scaled, scaler = scaling_data(train_to_scale, val_to_scale, test)

    if params is None:
        if look_back_list is None:
            look_back_list = [7, 14, 21, 28]
        # best = grid_search_mlp(train_scaled, val_scaled, val_to_scale, look_back_list, scaler)
        best_params, best_lb = search_best_mlp(train_scaled, val_scaled, look_back_list, scaler, n_trials=30)

    look_back = best_lb
    params = best_params

    # 2) Retrain on train+val
    scaled_all = np.concatenate([train_scaled, val_scaled])
    X_all, y_all = create_dataset(scaled_all, look_back)
    test_scaled = np.concatenate([scaled_all[-look_back:], test_scaled])
    X_test, y_test = create_dataset(test_scaled, look_back)

    X_all_t = torch.FloatTensor(X_all)
    y_all_t = torch.FloatTensor(y_all).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # build final model
    dims = [look_back] + [params['hidden_size']]*params['n_layers'] + [1]
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn   = nn.MSELoss()
    model.train()
    for _ in range(params['epochs']):
        perm = torch.randperm(len(X_all_t))
        for i in range(0, len(perm), params['batch_size']):
            idx = perm[i:i+params['batch_size']]
            xb, yb = X_all_t[idx], y_all_t[idx]
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

    # 3) Forecast
    model.eval()
    preds_t = []
    with torch.no_grad():
        preds_t = model(X_test_t).squeeze().numpy()
    preds = scaler.inverse_transform(preds_t.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    if boxcox_lambda is not None:
        preds = invert_boxcox(preds, boxcox_lambda)
        y_true = invert_boxcox(y_true, boxcox_lambda)

    # Ensure preds and y_true are aligned
    if len(preds) != len(y_true):
        raise ValueError(f"Predictions and true values have different lengths: {len(preds)} vs {len(y_true)}")

    # Ensure indices match
    if len(test.index) < len(preds):
        preds = preds[:len(test.index)]
    elif len(test.index) > len(preds):
        y_true = y_true[:len(test.index)]

    if np.any(np.isnan(preds)) or np.any(np.isnan(y_true)) or len(preds) == 0 or len(y_true) == 0:
        print("⚠️ Prediction or target contains NaN or is empty. Skipping evaluation.")
        print(f"Preds length: {len(preds)}, y_true length: {len(y_true)}")

    # eval
    quick_accuracy_metrics(y_true, preds)

    return model, pd.Series(preds, index=test.index[-len(preds):]), y_true

def search_best_mlp(train, val, look_back_list, scaler, param_grid=None, n_trials=30):
    print("\nSearching best MLP hyperparameters...")

    if param_grid is None:
        param_grid = {
            'n_layers': [1, 2, 3],
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
