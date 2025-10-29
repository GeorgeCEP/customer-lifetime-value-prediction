# Quick Start Guide - CLV Prediction Project

## 🚀 Get Up and Running in 5 Minutes

### Prerequisites
- Python 3.8+
- pip installed

### Quick Setup

```bash
# 1. Navigate to project
cd customer_lifetime_value_prediction

# 2. Create and activate virtual environment (Windows)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate data and train model
python scripts/generate_sample_data.py
python scripts/data_preprocessing.py
python scripts/train_models.py

# 5. Start API
python api/app.py
```

### Test the API

In a new terminal (while API is running):

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={
        "total_spent": 1250,
        "num_orders": 8,
        "avg_order_value": 156,
        "days_since_first_purchase": 180,
        "days_since_last_purchase": 15,
        "category_diversity": 3,
        "purchase_frequency": 0.4,
        "monthly_spending": 200
    }
)

print(response.json())
```

For detailed explanations, see the main [README.md](README.md).

