"""
Data loading utilities for CLV prediction project.
"""

import pandas as pd
import os

def load_transactions(file_path='data/raw/transactions.csv'):
    """Load transaction data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transaction data not found at {file_path}. Run generate_sample_data.py first.")
    
    df = pd.read_csv(file_path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    return df

def load_customers(file_path='data/raw/customers.csv'):
    """Load customer data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Customer data not found at {file_path}. Run generate_sample_data.py first.")
    
    df = pd.read_csv(file_path)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    return df

def load_clv_data(file_path='data/processed/clv_data.csv'):
    """Load CLV target data"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CLV data not found at {file_path}. Run generate_sample_data.py first.")
    
    df = pd.read_csv(file_path)
    if 'last_order_date' in df.columns:
        df['last_order_date'] = pd.to_datetime(df['last_order_date'])
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'])
    return df

def load_full_dataset():
    """Load and merge all datasets"""
    transactions = load_transactions()
    customers = load_customers()
    clv_data = load_clv_data()
    
    return transactions, customers, clv_data

