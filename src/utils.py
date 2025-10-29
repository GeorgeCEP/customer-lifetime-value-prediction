"""
Utility functions for CLV prediction project.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive regression metrics"""
    
    metrics = {
        'r2_score': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    }
    
    return metrics

def print_metrics(metrics, title="Model Metrics"):
    """Print metrics in a formatted way"""
    
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"R² Score:      {metrics['r2_score']:.4f}")
    print(f"MAE:            ${metrics['mae']:.2f}")
    print(f"RMSE:           ${metrics['rmse']:.2f}")
    print(f"MAPE:           {metrics['mape']:.2f}%")
    print(f"{'='*50}\n")

