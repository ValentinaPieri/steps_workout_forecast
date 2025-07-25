from utils import accuracy_metrics
from dm_test import dm_test

def check_dm_test(actual_lst, sarima_preds, mlp_preds, xgboost_preds, h=1, crit="MSE", power=2):
    print("\nPerforming Diebold-Mariano tests for model comparison...\n")
    print(f"If p-value < 0.05, the null hypothesis is rejected, indicating a significant difference in forecast accuracy.\n")
    print(f"If p-value >= 0.05, the null hypothesis is not rejected, indicating no significant difference in forecast accuracy.\n")

    dm_sarima_nn = dm_test(actual_lst, sarima_preds, mlp_preds, h=1, crit="MSE")
    print(f"SARIMA vs Neural Network - DM stat: {dm_sarima_nn.DM:.4f}, p-value: {dm_sarima_nn.p_value:.4f}")
    # Confronto SARIMA vs XGBoost
    dm_sarima_xgb = dm_test(actual_lst, sarima_preds, xgboost_preds, h=1, crit="MSE")
    print(f"SARIMA vs XGBoost - DM stat: {dm_sarima_xgb.DM:.4f}, p-value: {dm_sarima_xgb.p_value:.4f}")
    # Confronto Neural Network vs XGBoost
    dm_nn_xgb = dm_test(actual_lst, mlp_preds, xgboost_preds, h=1, crit="MSE")
    print(f"Neural Network vs XGBoost - DM stat: {dm_nn_xgb.DM:.4f}, p-value: {dm_nn_xgb.p_value:.4f}")

    return dm_sarima_nn, dm_sarima_xgb, dm_nn_xgb

def compare_metrics(test, sarima_preds, mlp_preds, xgb_preds):
    """
    Create a comprehensive Markdown report with all model results
    """
    print("\nGenerating comprehensive results report...")

    print("\n" + "-"*50 + "\n")
    
    # SARIMA
    if sarima_preds is not None:
        sarima_results = accuracy_metrics(test[-len(sarima_preds):], sarima_preds, "SARIMA")

    print("\n" + "-"*50 + "\n")

    # MLP
    if mlp_preds is not None:
        mlp_results = accuracy_metrics(test[-len(mlp_preds):], mlp_preds, "MLP")

    # print a division line
    print("\n" + "-"*50 + "\n")
    
    # XGBoost
    if xgb_preds is not None:
        xgb_results = accuracy_metrics(test[-len(xgb_preds):], xgb_preds, "XGBoost")

    print("\n" + "-"*50 + "\n")
    
    lines = [
        "# Report on Daily Steps Forecasting Results",
        "",
        "This project presents the results of the analysis of the forecasting of a time series on daily activity metrics, comparing three different models: SARIMA, MLP, and XGBoost.",
        "",
        "## Preprocessing on the Dataset",
        "- **Missing Values:** Handled by filling with previous values",
        "- **Outlier Detection:** Identified long walks as outliers",
        "- **Data Splitting:** 28 days for testing, remaining for training",
        "- **Transformation Method:** Box-Cox transformation for variance stabilization",
        "- **Smoothing:** Kalman filter applied for trend extraction",
        "- **Scaling:** Applied only for MLP model (Standard scaler)",
        "",
        "## Models and Results",
        "",
        "### 1. SARIMA",
    ]

    if sarima_results is not None:
        lines += [
            "",
            "**Performance Metrics:**",
            "| Metric | Test |",
            "|--------|------|",
            f"| RMSE (steps) | {sarima_results['RMSE']:.4f} |",
            f"| MAE (steps) | {sarima_results['MAE']:.4f} |",
            f"| MAPE (%) | {sarima_results['MAPE']:.2f} |",
        ]

    lines += [
        "",
        "### 2. MLP (Multi-Layer Perceptron)"
    ]

    if mlp_results is not None:
        lines += [
            "",
            "**Performance Metrics:**",
            "| Metric | Test |",
            "|--------|------|",
            f"| RMSE (steps) | {mlp_results['RMSE']:.4f} |",
            f"| MAE (steps) | {mlp_results['MAE']:.4f} |",
            f"| MAPE (%) | {mlp_results['MAPE']:.2f} |",
        ]

    lines += [
        "",
        "### 3. XGBoost",
    ]

    if xgb_results is not None:
        lines += [
            "",
            "**Performance Metrics:**",
            "| Metric | Test |",
            "|--------|------|",
            f"| RMSE (steps) | {xgb_results['RMSE']:.4f} |",
            f"| MAE (steps) | {xgb_results['MAE']:.4f} |",
            f"| MAPE (%) | {xgb_results['MAPE']:.2f} |",
        ]

    # Add Diebold-Mariano test results
    dm_sarima_nn, dm_sarima_xgb, dm_nn_xgb = check_dm_test(test, sarima_preds, mlp_preds, xgb_preds)

    if sarima_results is not None and mlp_results is not None and xgb_results is not None:
        lines += [
            "",
            "## Diebold-Mariano Test Results",
            "The Diebold-Mariano test was conducted to compare the forecast accuracy of the models.",
            "The null hypothesis states that the two forecasts have the same accuracy.",
            "A p-value < 0.05 indicates a significant difference in forecast accuracy.",
            "",
            "### DM Test Results",
            "- SARIMA vs MLP: DM stat: {:.4f}, p-value: {:.4f}".format(dm_sarima_nn.DM, dm_sarima_nn.p_value),
            "- SARIMA vs XGBoost: DM stat: {:.4f}, p-value: {:.4f}".format(dm_sarima_xgb.DM, dm_sarima_xgb.p_value),
            "- MLP vs XGBoost: DM stat: {:.4f}, p-value: {:.4f}".format(dm_nn_xgb.DM, dm_nn_xgb.p_value)
        ]

    # Determine the best model based on RMSE
    best_model = min(
        [("SARIMA", sarima_results), ("MLP", mlp_results), ("XGBoost", xgb_results)],
        key=lambda x: x[1]['RMSE']
    )

    lines += [
        "",
        f"The best model was **{best_model[0]}** having the lowest RMSE of **{best_model[1]['RMSE']:.4f} steps**",
        ""
    ]

    with open('results_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print("Comprehensive results report generated: results_report.md")

    return best_model