import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_ecommerce_data(n_samples=2000):
    """
    Generate synthetic e-commerce dataset for product return prediction
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate Order_ID
    order_ids = [f"ORD{i:06d}" for i in range(1, n_samples + 1)]
    
    # Generate Customer_ID
    customer_ids = [f"CUST{random.randint(1, 500):05d}" for _ in range(n_samples)]
    
    # Product Categories
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 
                  'Toys', 'Beauty', 'Automotive', 'Food', 'Health']
    product_categories = np.random.choice(categories, n_samples, p=[0.15, 0.20, 0.12, 0.08, 0.10, 0.10, 0.08, 0.07, 0.05, 0.05])
    
    # Product Price (realistic distribution)
    product_prices = np.random.lognormal(3.5, 0.8, n_samples)
    product_prices = np.clip(product_prices, 5, 500)
    
    # Customer Age
    customer_ages = np.random.normal(35, 12, n_samples)
    customer_ages = np.clip(customer_ages, 18, 80).astype(int)
    
    # Customer Location
    locations = ['New York', 'California', 'Texas', 'Florida', 'Illinois', 
                 'Pennsylvania', 'Ohio', 'Georgia', 'Michigan', 'North Carolina']
    customer_locations = np.random.choice(locations, n_samples)
    
    # Purchase History Count
    purchase_history = np.random.poisson(8, n_samples)
    purchase_history = np.clip(purchase_history, 0, 50)
    
    # Return History Rate (0 to 1)
    return_history_rate = np.random.beta(2, 8, n_samples)
    
    # Delivery Time (days)
    delivery_times = np.random.gamma(2, 2, n_samples)
    delivery_times = np.clip(delivery_times, 1, 15).astype(int)
    
    # Payment Method
    payment_methods = ['Credit Card', 'Debit Card', 'PayPal', 'Cash on Delivery', 'UPI']
    payment_methods_array = np.random.choice(payment_methods, n_samples, p=[0.35, 0.25, 0.20, 0.10, 0.10])
    
    # Discount Applied
    discount_applied = np.random.choice([0, 5, 10, 15, 20, 25, 30], n_samples, p=[0.30, 0.20, 0.20, 0.15, 0.08, 0.05, 0.02])
    
    # Product Rating (1-5)
    product_ratings = np.random.normal(3.8, 0.8, n_samples)
    product_ratings = np.clip(product_ratings, 1, 5).round(1)
    
    # Generate Return Target (based on realistic factors)
    # Higher return probability for: expensive items, electronics, clothing, long delivery, low ratings, high return history
    return_prob = (
        0.15 +  # base probability
        (product_prices / 500) * 0.20 +  # price factor
        np.isin(product_categories, ['Electronics', 'Clothing']) * 0.10 +  # category factor
        (delivery_times / 15) * 0.15 +  # delivery time factor
        (5 - product_ratings) / 4 * 0.20 +  # rating factor
        return_history_rate * 0.25 +  # return history factor
        (discount_applied / 30) * 0.10  # discount factor
    )
    
    return_prob = np.clip(return_prob, 0.05, 0.95)
    returned = (np.random.random(n_samples) < return_prob).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'Order_ID': order_ids,
        'Customer_ID': customer_ids,
        'Product_Category': product_categories,
        'Product_Price': product_prices.round(2),
        'Customer_Age': customer_ages,
        'Customer_Location': customer_locations,
        'Purchase_History_Count': purchase_history,
        'Return_History_Rate': return_history_rate.round(3),
        'Delivery_Time': delivery_times,
        'Payment_Method': payment_methods_array,
        'Discount_Applied': discount_applied,
        'Product_Rating': product_ratings,
        'Returned': returned
    })
    
    return data

if __name__ == "__main__":
    # Generate dataset
    df = generate_ecommerce_data(2000)
    
    # Save to CSV
    df.to_csv('data.csv', index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Return rate: {df['Returned'].mean():.2%}")
    print(f"\nDataset preview:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nReturn statistics:")
    print(df['Returned'].value_counts())
