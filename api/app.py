"""
Flask API for CLV prediction.
Provides REST endpoints for real-time customer lifetime value predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import prepare_features_for_modeling

app = Flask(__name__)
CORS(app)

# Load model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_model.pkl')
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
else:
    model = None
    print("Warning: Model not found. Please train the model first using train_models.py")

@app.route('/', methods=['GET'])
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Customer Lifetime Value Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Predict CLV for a customer',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict CLV for a customer"""
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        # Get JSON data
        data = request.json
        
        # Convert to DataFrame (single row)
        customer_data = pd.DataFrame([data])
        
        # Prepare features
        # For a single prediction, we need to reconstruct the feature set
        # This is a simplified version - in production, you'd have a proper feature pipeline
        
        # Map input to feature columns
        feature_mapping = {
            'total_spent': 'total_spent',
            'num_orders': 'num_orders',
            'avg_order_value': 'avg_order_value',
            'days_since_first_purchase': 'days_since_first_purchase',
            'days_since_last_purchase': 'days_since_last_purchase',
            'recency': 'recency',
            'purchase_frequency': 'purchase_frequency',
            'category_diversity': 'category_diversity',
            'monthly_spending': 'monthly_spending'
        }
        
        # Create feature vector
        features = {}
        for key, value in data.items():
            if key in feature_mapping:
                features[feature_mapping[key]] = value
        
        # Set defaults for missing features
        default_features = {
            'total_spent': features.get('total_spent', 500),
            'num_orders': features.get('num_orders', 5),
            'avg_order_value': features.get('avg_order_value', 100),
            'std_order_value': 20,
            'total_quantity': features.get('num_orders', 5) * 2,
            'avg_quantity': 2,
            'category_diversity': features.get('category_diversity', 1),
            'total_discount': 0,
            'avg_discount': 0,
            'age': 35,
            'recency': features.get('days_since_last_purchase', 30),
            'customer_lifetime_days': features.get('days_since_first_purchase', 180),
            'days_since_first_purchase': features.get('days_since_first_purchase', 180),
            'days_since_last_purchase': features.get('days_since_last_purchase', 30),
            'purchase_frequency': features.get('purchase_frequency', 0.2),
            'monthly_spending': features.get('monthly_spending', 100),
            'value_per_order': features.get('avg_order_value', 100),
            'consistency_score': 0.5,
            'churn_risk': 0 if features.get('days_since_last_purchase', 30) < 90 else 1,
            'orders_in_q1': 1,
            'orders_in_q2': 1,
            'orders_in_q3': 1,
            'orders_in_q4': 1,
            'orders_weekday': features.get('num_orders', 5),
            'orders_weekend': 0,
            'avg_days_between_orders': 30,
            'weekend_ratio': 0
        }
        
        # Update with provided values
        default_features.update(features)
        
        # Create DataFrame
        X = pd.DataFrame([default_features])
        
        # Ensure all model feature columns are present
        if hasattr(model, 'feature_columns'):
            for col in model.feature_columns:
                if col not in X.columns:
                    X[col] = 0
        
        # Reorder columns to match training
        if hasattr(model, 'feature_columns'):
            X = X[[col for col in model.feature_columns if col in X.columns]]
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return prediction
        return jsonify({
            'predicted_clv': round(float(prediction), 2),
            'predicted_clv_6months': round(float(prediction * 0.5), 2),
            'features_used': list(features.keys()),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

if __name__ == '__main__':
    print("Starting CLV Prediction API...")
    print("API will be available at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

