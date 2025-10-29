"""
Machine learning models for CLV prediction.
Includes XGBoost, Random Forest, and Neural Network implementations.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import pandas as pd

class CLVPredictor:
    """Base class for CLV prediction models"""
    
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def _create_model(self):
        """Create model instance"""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """Train the model"""
        self.feature_columns = X.columns.tolist()
        
        if self.model_type in ['linear']:
            # Scale features for linear models
            X_scaled = self.scaler.fit_transform(X)
            self.model = self._create_model()
            self.model.fit(X_scaled, y)
        else:
            self.model = self._create_model()
            self.model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Ensure same column order
        X = X[self.feature_columns]
        
        if self.model_type in ['linear']:
            X_scaled = self.scaler.transform(X)
            return self.model.predict(X_scaled)
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance (if available)"""
        if self.model is None:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        else:
            return None

def prepare_features_for_modeling(df):
    """Prepare features for ML modeling"""
    
    # Select feature columns (exclude target and IDs)
    exclude_cols = ['customer_id', 'clv_12months', 'first_order_date', 'last_order_date',
                    'signup_date', 'preferred_category_rfm', 'category']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Separate features and target
    X = df[feature_cols].copy()
    y = df['clv_12months'].copy()
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col in X.columns:
            # One-hot encode if few categories
            if X[col].nunique() < 10:
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
            else:
                # Label encode if many categories
                X = X.drop(col, axis=1)
    
    # Fill any remaining NaN (only for numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    
    return X, y

