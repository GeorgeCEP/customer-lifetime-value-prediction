import requests
import json

# Test prediction endpoint
customer_data = {
    "total_spent": 1250.50,
    "num_orders": 8,
    "avg_order_value": 156.31,
    "days_since_first_purchase": 180,
    "days_since_last_purchase": 15,
    "category_diversity": 3,
    "purchase_frequency": 0.4,
    "monthly_spending": 200
}

response = requests.post(
    "http://localhost:5000/predict",
    json=customer_data
)

print("Status Code:", response.status_code)
print("\nResponse:")
print(json.dumps(response.json(), indent=2))