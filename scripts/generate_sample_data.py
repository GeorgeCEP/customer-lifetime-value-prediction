"""
Generate synthetic customer transaction data for CLV prediction.
This creates realistic e-commerce data with various customer patterns.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

def generate_customer_data(n_customers=10000, start_date='2022-01-01', end_date='2024-12-31'):
    """Generate synthetic customer transaction data"""
    
    np.random.seed(42)
    random.seed(42)
    
    # Convert dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    categories = ['Electronics', 'Fashion', 'Home & Garden', 'Sports', 'Books', 'Toys']
    customer_segments = ['VIP', 'Regular', 'New', 'At-Risk']
    
    transactions = []
    customers = {}
    
    for customer_id in range(1, n_customers + 1):
        # Generate customer characteristics
        signup_date = start + timedelta(days=random.randint(0, (end - start).days - 365))
        age = random.randint(18, 75)
        gender = random.choice(['M', 'F', 'Other'])
        location = random.choice(['Urban', 'Suburban', 'Rural'])
        preferred_category = random.choice(categories)
        
        # Determine customer segment based on potential value
        segment_prob = np.random.random()
        if segment_prob < 0.1:
            segment = 'VIP'  # High-value customers
            base_order_value = np.random.normal(250, 50)
            order_frequency = np.random.uniform(0.3, 0.7)  # orders per month
        elif segment_prob < 0.5:
            segment = 'Regular'  # Regular customers
            base_order_value = np.random.normal(150, 30)
            order_frequency = np.random.uniform(0.1, 0.3)
        elif segment_prob < 0.8:
            segment = 'New'  # New customers
            base_order_value = np.random.normal(100, 25)
            order_frequency = np.random.uniform(0.05, 0.15)
        else:
            segment = 'At-Risk'  # Low engagement
            base_order_value = np.random.normal(80, 20)
            order_frequency = np.random.uniform(0.02, 0.1)
        
        # Calculate number of orders
        months_active = ((end - signup_date).days) / 30
        num_orders = int(months_active * order_frequency)
        num_orders = max(1, num_orders)  # At least 1 order
        
        # Generate transactions
        for order_num in range(num_orders):
            # Order date distribution
            days_since_signup = random.randint(0, min((end - signup_date).days, 730))
            order_date = signup_date + timedelta(days=days_since_signup)
            
            if order_date > end:
                break
            
            # Order value with some variation
            order_value = max(20, base_order_value + np.random.normal(0, base_order_value * 0.3))
            order_value = round(order_value, 2)
            
            # Category preference (70% chance of preferred category)
            if random.random() < 0.7:
                order_category = preferred_category
            else:
                order_category = random.choice(categories)
            
            # Quantity
            quantity = random.randint(1, 5)
            
            # Discount applied
            discount = random.choice([0, 0, 0, 0.1, 0.15, 0.2])  # Most orders have no discount
            discounted_value = order_value * (1 - discount)
            
            transactions.append({
                'customer_id': customer_id,
                'order_id': f'ORD{customer_id:05d}{order_num:03d}',
                'order_date': order_date.strftime('%Y-%m-%d'),
                'order_value': discounted_value,
                'quantity': quantity,
                'category': order_category,
                'discount': discount,
                'age': age,
                'gender': gender,
                'location': location,
                'signup_date': signup_date.strftime('%Y-%m-%d')
            })
        
        # Store customer info
        customers[customer_id] = {
            'customer_id': customer_id,
            'signup_date': signup_date.strftime('%Y-%m-%d'),
            'age': age,
            'gender': gender,
            'location': location,
            'preferred_category': preferred_category,
            'segment': segment
        }
    
    # Create DataFrames
    df_transactions = pd.DataFrame(transactions)
    df_customers = pd.DataFrame(customers.values())
    
    # Calculate actual CLV for training
    clv_data = df_transactions.groupby('customer_id').agg({
        'order_value': 'sum',
        'order_id': 'count',
        'order_date': 'max'
    }).reset_index()
    
    clv_data.columns = ['customer_id', 'total_spent', 'num_orders', 'last_order_date']
    clv_data = clv_data.merge(df_customers, on='customer_id', how='left')
    
    # Add future value prediction (simulated based on patterns)
    # This represents the "true" CLV we're trying to predict
    future_months = 12
    for idx, row in clv_data.iterrows():
        avg_monthly_value = row['total_spent'] / max(1, (datetime.strptime(row['last_order_date'], '%Y-%m-%d') - 
                                                          datetime.strptime(row['signup_date'], '%Y-%m-%d')).days / 30)
        future_value = avg_monthly_value * future_months * np.random.uniform(0.8, 1.2)
        clv_data.loc[idx, 'future_clv'] = max(0, future_value)
    
    clv_data['clv_6months'] = clv_data['future_clv'] * 0.5  # 6-month projection
    clv_data['clv_12months'] = clv_data['future_clv']  # 12-month projection
    
    return df_transactions, df_customers, clv_data

def main():
    """Main function to generate and save data"""
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("Generating synthetic customer data...")
    df_transactions, df_customers, df_clv = generate_customer_data(n_customers=10000)
    
    # Save data
    df_transactions.to_csv('data/raw/transactions.csv', index=False)
    df_customers.to_csv('data/raw/customers.csv', index=False)
    df_clv.to_csv('data/processed/clv_data.csv', index=False)
    
    print(f"\nGenerated data files:")
    print(f"  - Transactions: {len(df_transactions):,} rows")
    print(f"  - Customers: {len(df_customers):,} rows")
    print(f"  - CLV Data: {len(df_clv):,} rows")
    print(f"\nFiles saved to data/raw/ and data/processed/")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Average CLV (12 months): ${df_clv['clv_12months'].mean():.2f}")
    print(f"  Total Revenue: ${df_clv['total_spent'].sum():,.2f}")
    print(f"  Average Orders per Customer: {df_clv['num_orders'].mean():.2f}")

if __name__ == '__main__':
    main()

