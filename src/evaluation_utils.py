import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error, r2_score)
                                 
        
def root_mean_squared_error(y_true: np.array, y_pred: np.array) -> float:
    """Function to compute root mean squared error from true values and associated predictions.
    
    Args: 
        y_true (numpy.array): true target values
        y_pred (numpy.array): predicted values
        
    Returns: 
        Float value representing RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))
    

def regression_metrics(y_pred: np.array, y_true: np.array) -> dict:
    """Function that computes all desired regression metrics: MSE, RMSE, MAE, MAPE, R2-Score. 
    
    Args: 
        y_true (numpy.array): true target values
        y_pred (numpy.array): predicted values
        
    Returns: 
        Dictionary with keys corresponding to metric name and values to the its amount.
    """
    METRICS = ["MSE", "RMSE", "MAE", "MAPE", "R2-Score"]
    FUNCTIONS = [
        mean_squared_error,
        root_mean_squared_error,
        mean_absolute_error,
        mean_absolute_percentage_error,
        r2_score
    ]
    output_metrics = {
        metric: np.round(function(y_true, y_pred), 2) \
        for metric, function in zip(METRICS, FUNCTIONS)
    }
    return output_metrics