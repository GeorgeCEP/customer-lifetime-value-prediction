"""
Evaluate trained models and generate performance reports.
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import CLVPredictor, prepare_features_for_modeling
from src.utils import calculate_metrics, print_metrics

def evaluate_models():
    """Evaluate all trained models"""
    
    print("Loading data...")
    df = pd.read_csv('data/processed/final_features.csv')
    
    X, y = prepare_features_for_modeling(df)
    
    # Load model if exists
    model_path = 'models/best_model.pkl'
    if not os.path.exists(model_path):
        print("Model not found. Please train models first using train_models.py")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Cross-validation
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(model.model, X, y, cv=5, scoring='r2')
    
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Full dataset predictions
    print("\nEvaluating on full dataset...")
    predictions = model.predict(X)
    metrics = calculate_metrics(y, predictions)
    print_metrics(metrics, "Full Dataset Performance")
    
    # Feature importance
    if model.get_feature_importance():
        print("\nTop 15 Most Important Features:")
        importance = model.get_feature_importance()
        for i, (feature, score) in enumerate(importance[:15], 1):
            print(f"  {i:2d}. {feature:30s}: {score:.4f}")

if __name__ == '__main__':
    evaluate_models()

