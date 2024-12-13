import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae_and_rmse(true, pred) -> tuple:
    """Return the mean absolute error and root mean squared error in a tuple."""
    return mean_absolute_error(true, pred), np.sqrt(mean_squared_error(true, pred))
