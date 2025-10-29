"""
Data preprocessing script for CLV prediction.
Cleans and prepares data for modeling.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_transactions, load_customers, load_clv_data
from src.features import create_all_features

def preprocess_data():
    """Main preprocessing function"""
    
    print("Loading data...")
    transactions = load_transactions()
    customers = load_customers()
    clv_data = load_clv_data()
    
    print(f"Loaded {len(transactions):,} transactions for {len(customers):,} customers")
    
    # Create features
    features = create_all_features(transactions, customers)
    
    # Merge with target variable
    final_data = features.merge(
        clv_data[['customer_id', 'clv_12months']],
        on='customer_id',
        how='inner'
    )
    
    # Remove rows with missing target
    final_data = final_data.dropna(subset=['clv_12months'])
    
    # Handle missing values in features - only for numeric columns
    numeric_cols = final_data.select_dtypes(include=[np.number]).columns
    
    # Fill NaN for numeric columns with median
    if len(numeric_cols) > 0:
        final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].median())
    
    # Handle categorical/string columns - fill with mode or 'Unknown'
    categorical_cols = final_data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'customer_id':  # Don't fill customer_id
            mode_value = final_data[col].mode()
            if len(mode_value) > 0:
                final_data[col] = final_data[col].fillna(mode_value[0])
            else:
                final_data[col] = final_data[col].fillna('Unknown')
    
    # Remove infinite values (only in numeric columns)
    if len(numeric_cols) > 0:
        final_data[numeric_cols] = final_data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        final_data[numeric_cols] = final_data[numeric_cols].fillna(final_data[numeric_cols].median())
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    final_data.to_csv('data/processed/final_features.csv', index=False)
    
    print(f"\nPreprocessing complete!")
    print(f"Final dataset: {len(final_data):,} customers")
    print(f"Features: {len(final_data.columns) - 2} features")  # Excluding customer_id and target
    print(f"Saved to: data/processed/final_features.csv")
    
    return final_data

if __name__ == '__main__':
    preprocess_data()

