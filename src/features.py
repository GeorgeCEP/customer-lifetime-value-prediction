"""
Feature engineering functions for CLV prediction.
Creates RFM features, behavioral features, and temporal patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def calculate_rfm_features(transactions, reference_date=None):
    """Calculate Recency, Frequency, Monetary (RFM) features"""
    
    if reference_date is None:
        reference_date = transactions['order_date'].max()
    
    # Customer-level aggregations
    customer_stats = transactions.groupby('customer_id').agg({
        'order_date': ['max', 'min', 'count'],
        'order_value': ['sum', 'mean', 'std'],
        'quantity': 'sum',
        'category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
        'discount': 'mean'
    }).reset_index()
    
    customer_stats.columns = ['customer_id', 'last_order_date', 'first_order_date', 
                               'num_orders', 'total_spent', 'avg_order_value', 
                               'std_order_value', 'total_quantity', 'preferred_category', 
                               'avg_discount']
    
    # Calculate recency (days since last order)
    customer_stats['recency'] = (reference_date - customer_stats['last_order_date']).dt.days
    
    # Calculate customer lifetime (days)
    customer_stats['customer_lifetime_days'] = (
        customer_stats['last_order_date'] - customer_stats['first_order_date']
    ).dt.days
    
    # Fill NaN for std (only 1 order)
    customer_stats['std_order_value'] = customer_stats['std_order_value'].fillna(0)
    
    return customer_stats

def calculate_behavioral_features(transactions, customers):
    """Calculate behavioral and engagement features"""
    
    # Transactions already contain customer demographics, so use them directly
    # Merge only to ensure we have all customer info if needed
    df = transactions.copy()
    
    # Add customer columns that might not be in transactions
    customer_cols_to_add = ['age', 'gender', 'location', 'signup_date']
    for col in customer_cols_to_add:
        if col not in df.columns and col in customers.columns:
            # Merge the missing column from customers
            df = df.merge(customers[['customer_id', col]], on='customer_id', how='left')
    
    # Base aggregation dictionary - always present columns
    agg_dict = {
        'order_value': ['sum', 'mean', 'max', 'min', 'std'],
        'order_date': ['min', 'max', 'count'],
        'quantity': ['sum', 'mean'],
        'category': lambda x: len(x.unique()),  # Category diversity
        'discount': ['sum', 'mean']
    }
    
    # Add customer demographic columns only if they exist
    for col in customer_cols_to_add:
        if col in df.columns:
            agg_dict[col] = 'first'
    
    # Group by customer and calculate features
    behavioral = df.groupby('customer_id').agg(agg_dict).reset_index()
    
    # Handle MultiIndex columns from aggregation
    if isinstance(behavioral.columns, pd.MultiIndex):
        new_cols = []
        for col in behavioral.columns:
            if col[0] == 'customer_id':
                new_cols.append('customer_id')
            elif col[1] == '':
                new_cols.append(col[0])
            else:
                # Handle lambda function name
                if col[1].startswith('<lambda'):
                    new_cols.append(f"{col[0]}_diversity" if col[0] == 'category' else f"{col[0]}_{col[1]}")
                else:
                    new_cols.append(f"{col[0]}_{col[1]}")
        behavioral.columns = new_cols
    
    # Fix category diversity column name
    if 'category_diversity' not in behavioral.columns:
        for col in behavioral.columns:
            if 'category' in col.lower() and 'diversity' in col.lower():
                behavioral = behavioral.rename(columns={col: 'category_diversity'})
                break
        # Or if lambda function created it
        for col in behavioral.columns:
            if col.startswith('category_<'):
                behavioral = behavioral.rename(columns={col: 'category_diversity'})
                break
    
    # Ensure standard column names
    standard_names = {
        'order_value_sum': 'total_spent',
        'order_value_mean': 'avg_order_value',
        'order_value_max': 'max_order_value',
        'order_value_min': 'min_order_value',
        'order_value_std': 'std_order_value',
        'order_date_min': 'first_order_date',
        'order_date_max': 'last_order_date',
        'order_date_count': 'order_count',
        'quantity_sum': 'total_quantity',
        'quantity_mean': 'avg_quantity',
        'discount_sum': 'total_discount',
        'discount_mean': 'avg_discount'
    }
    
    behavioral = behavioral.rename(columns=standard_names)
    
    # Ensure date columns are datetime
    for col in ['first_order_date', 'last_order_date']:
        if col in behavioral.columns:
            behavioral[col] = pd.to_datetime(behavioral[col], errors='coerce')
    
    # Handle signup_date
    if 'signup_date' in behavioral.columns:
        # Check if it's already datetime or needs conversion
        if not pd.api.types.is_datetime64_any_dtype(behavioral['signup_date']):
            behavioral['signup_date'] = pd.to_datetime(behavioral['signup_date'], errors='coerce')
    
    # Fill NaN for numeric columns
    if 'std_order_value' in behavioral.columns:
        behavioral['std_order_value'] = behavioral['std_order_value'].fillna(0)
    
    # Calculate additional behavioral metrics
    if 'last_order_date' in behavioral.columns and 'first_order_date' in behavioral.columns:
        behavioral['purchase_frequency'] = behavioral['order_count'] / \
            ((behavioral['last_order_date'] - behavioral['first_order_date']).dt.days / 30 + 1)
        behavioral['days_since_first_purchase'] = (
            behavioral['last_order_date'] - behavioral['first_order_date']
        ).dt.days
        
        behavioral['days_since_last_purchase'] = (
            datetime.now() - behavioral['last_order_date']
        ).dt.days
        
        # Calculate monthly spending
        behavioral['monthly_spending'] = behavioral['total_spent'] / \
            (behavioral['days_since_first_purchase'] / 30 + 1)
    else:
        behavioral['purchase_frequency'] = 0
        behavioral['days_since_first_purchase'] = 0
        behavioral['days_since_last_purchase'] = 0
        behavioral['monthly_spending'] = 0
    
    return behavioral

def calculate_temporal_features(transactions):
    """Calculate time-based patterns"""
    
    temporal = transactions.groupby('customer_id').apply(lambda x: pd.Series({
        'orders_in_q1': len(x[x['order_date'].dt.quarter == 1]),
        'orders_in_q2': len(x[x['order_date'].dt.quarter == 2]),
        'orders_in_q3': len(x[x['order_date'].dt.quarter == 3]),
        'orders_in_q4': len(x[x['order_date'].dt.quarter == 4]),
        'orders_weekday': (x['order_date'].dt.dayofweek < 5).sum(),
        'orders_weekend': (x['order_date'].dt.dayofweek >= 5).sum(),
        'avg_days_between_orders': x['order_date'].sort_values().diff().dt.days.mean() if len(x) > 1 else 0
    })).reset_index()
    
    temporal['weekend_ratio'] = temporal['orders_weekend'] / (
        temporal['orders_weekday'] + temporal['orders_weekend'] + 1)
    
    return temporal

def create_all_features(transactions, customers):
    """Create complete feature set"""
    
    print("Calculating RFM features...")
    rfm = calculate_rfm_features(transactions)
    
    print("Calculating behavioral features...")
    behavioral = calculate_behavioral_features(transactions, customers)
    
    print("Calculating temporal features...")
    temporal = calculate_temporal_features(transactions)
    
    # Merge all features
    print("Merging features...")
    
    # Identify overlapping columns (excluding customer_id)
    rfm_cols = set(rfm.columns) - {'customer_id'}
    behavioral_cols = set(behavioral.columns) - {'customer_id'}
    overlap = rfm_cols & behavioral_cols
    
    # Remove duplicate columns from behavioral that are already in RFM
    # We'll keep RFM version for metrics like total_spent, num_orders, avg_order_value
    cols_to_drop_from_behavioral = list(overlap)
    behavioral_clean = behavioral.drop(columns=cols_to_drop_from_behavioral)
    
    # Merge RFM and behavioral
    features = rfm.merge(behavioral_clean, on='customer_id', how='inner')
    
    # Merge temporal features
    features = features.merge(temporal, on='customer_id', how='left')
    
    # Merge with customer demographics
    features = features.merge(
        customers[['customer_id', 'segment', 'preferred_category']],
        on='customer_id', how='left'
    )
    
    # Handle potential duplicate date columns - use RFM version
    if 'first_order_date_rfm' in features.columns:
        features['first_order_date'] = features['first_order_date_rfm']
        features = features.drop(columns=['first_order_date_rfm'])
    if 'last_order_date_rfm' in features.columns:
        features['last_order_date'] = features['last_order_date_rfm']
        features = features.drop(columns=['last_order_date_rfm'])
    if 'first_order_date_beh' in features.columns:
        features = features.drop(columns=['first_order_date_beh'])
    if 'last_order_date_beh' in features.columns:
        features = features.drop(columns=['last_order_date_beh'])
    
    # Handle std_order_value - prefer RFM version if both exist
    if 'std_order_value_rfm' in features.columns and 'std_order_value_beh' in features.columns:
        features['std_order_value'] = features['std_order_value_rfm'].fillna(features['std_order_value_beh'])
        features = features.drop(columns=['std_order_value_rfm', 'std_order_value_beh'])
    elif 'std_order_value_rfm' in features.columns:
        features['std_order_value'] = features['std_order_value_rfm']
        features = features.drop(columns=['std_order_value_rfm'])
    elif 'std_order_value_beh' in features.columns:
        features['std_order_value'] = features['std_order_value_beh']
        features = features.drop(columns=['std_order_value_beh'])
    
    # Ensure we have the required columns for derived features
    if 'total_spent' not in features.columns:
        # Try to find it with suffix
        if 'total_spent_rfm' in features.columns:
            features['total_spent'] = features['total_spent_rfm']
            features = features.drop(columns=['total_spent_rfm'])
        elif 'total_spent_beh' in features.columns:
            features['total_spent'] = features['total_spent_beh']
            features = features.drop(columns=['total_spent_beh'])
    
    # Additional derived features (only if required columns exist)
    if 'total_spent' in features.columns and 'num_orders' in features.columns:
        features['value_per_order'] = features['total_spent'] / (features['num_orders'] + 1)
    else:
        features['value_per_order'] = 0
    
    if 'std_order_value' in features.columns:
        features['consistency_score'] = 1 / (features['std_order_value'] + 1)
    else:
        features['consistency_score'] = 0
    
    if 'days_since_last_purchase' in features.columns:
        features['churn_risk'] = (features['days_since_last_purchase'] > 90).astype(int)
    else:
        features['churn_risk'] = 0
    
    return features

