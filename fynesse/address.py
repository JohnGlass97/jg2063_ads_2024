import numpy as np
import statsmodels.api as sm
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

# This file contains code for suporting addressing questions in the data


def mae_and_rmse(true, pred) -> tuple:
    """Return the mean absolute error and root mean squared error in a tuple."""
    return mean_absolute_error(true, pred), np.sqrt(mean_squared_error(true, pred))


def do_kfold_mae_rmse(
    X: pd.DataFrame,
    y: pd.DataFrame,
    k: int,
    reg=False,
    alpha: float = 0,
    L1_wt: float = 0
) -> tuple:
    """Calculate the average MAE and RMSE using k-fold cross validation."""

    assert len(X) == len(y)

    kf = KFold(n_splits=k)

    train_metrics = []
    test_metrics = []

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        y_train = y.iloc[train_index]
        y_test = y.iloc[test_index]

        ols = sm.OLS(y_train, X_train)
        if reg:
            model = ols.fit_regularized(alpha=alpha, L1_wt=L1_wt)
        else:
            model = ols.fit()

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics.append(mae_and_rmse(y_train, y_train_pred))
        test_metrics.append(mae_and_rmse(y_test, y_test_pred))

    avg_train_mae = sum([x[0] for x in train_metrics]) / len(train_metrics)
    avg_test_mae = sum([x[0] for x in test_metrics]) / len(test_metrics)
    avg_train_rmse = sum([x[1] for x in train_metrics]) / len(train_metrics)
    avg_test_rmse = sum([x[1] for x in test_metrics]) / len(test_metrics)

    return avg_train_mae, avg_test_mae, avg_train_rmse, avg_test_rmse
