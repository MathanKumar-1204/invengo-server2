import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Sample historical data for price prediction (in a real-world scenario, you'd have a larger dataset)
price_history = [
    {"id": 1, "date": "2023-01-01", "price": 1.1},
    {"id": 1, "date": "2023-01-02", "price": 1.2},
    {"id": 1, "date": "2023-01-03", "price": 1.15},
    {"id": 2, "date": "2023-01-01", "price": 0.75},
    {"id": 2, "date": "2023-01-02", "price": 0.8},
    {"id": 2, "date": "2023-01-03", "price": 0.85},
    {"id": 3, "date": "2023-01-01", "price": 1.0},
    {"id": 3, "date": "2023-01-02", "price": 1.05},
    {"id": 3, "date": "2023-01-03", "price": 1.1},
]

# Convert the historical data into a DataFrame
df = pd.DataFrame(price_history)

# Feature engineering: Convert date into datetime format and create a feature for the day of the year
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear

# Prepare features and target
X = df[["day_of_year"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save the model
joblib.dump(model, 'price_prediction_model.pkl')
