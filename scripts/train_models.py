"""
Train multiple ML models for CLV prediction and select the best one.
"""

import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import CLVPredictor, prepare_features_for_modeling

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Evaluate model performance"""
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Metrics
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'train_mape': mean_absolute_percentage_error(y_train, y_pred_train),
        'test_mape': mean_absolute_percentage_error(y_test, y_pred_test),
    }
    
    return metrics, y_pred_test

def train_and_compare_models():
    """Train and compare multiple models"""
    
    print("Loading preprocessed data...")
    df = pd.read_csv('data/processed/final_features.csv')
    
    print(f"Dataset shape: {df.shape}")
    
    # Prepare features
    print("\nPreparing features...")
    X, y = prepare_features_for_modeling(df)
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTrain set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Models to train
    models = {
        'XGBoost': 'xgboost',
        'Random Forest': 'random_forest',
        'Linear Regression': 'linear'
    }
    
    results = {}
    best_model = None
    best_score = -np.inf
    
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    
    for name, model_type in models.items():
        print(f"\nTraining {name}...")
        
        model = CLVPredictor(model_type=model_type)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics, predictions = evaluate_model(model, X_test, y_test, X_train, y_train)
        results[name] = {
            'model': model,
            'metrics': metrics,
            'predictions': predictions
        }
        
        print(f"  R² Score: {metrics['test_r2']:.4f}")
        print(f"  MAE: ${metrics['test_mae']:.2f}")
        print(f"  RMSE: ${metrics['test_rmse']:.2f}")
        print(f"  MAPE: {metrics['test_mape']*100:.2f}%")
        
        # Track best model
        if metrics['test_r2'] > best_score:
            best_score = metrics['test_r2']
            best_model = name
    
    print("\n" + "="*60)
    print(f"Best Model: {best_model} (R² = {best_score:.4f})")
    print("="*60)
    
    # Save best model
    os.makedirs('models', exist_ok=True)
    best_model_obj = results[best_model]['model']
    joblib.dump(best_model_obj, 'models/best_model.pkl')
    
    print(f"\nSaved best model to: models/best_model.pkl")
    
    # Feature importance for best model
    if best_model_obj.get_feature_importance():
        importance = best_model_obj.get_feature_importance()
        print(f"\nTop 10 Most Important Features:")
        for i, (feature, score) in enumerate(importance[:10], 1):
            print(f"  {i}. {feature}: {score:.4f}")
    
    return results, best_model

if __name__ == '__main__':
    train_and_compare_models()

