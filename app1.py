import joblib
from flask import Flask, jsonify, request
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import random
from datetime import datetime, timedelta
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
# Simulated data
products = [
    {"id": 1, "product_name": "Apple", "price": 1.2, "change": 0.5},
    {"id": 2, "product_name": "Banana", "price": 0.8, "change": -0.3},
    {"id": 3, "product_name": "Orange", "price": 1.0, "change": 0.2},
    {"id": 4, "product_name": "Grapes", "price": 2.5, "change": -0.1},
    {"id": 5, "product_name": "Peach", "price": 3.0, "change": 0.4},
    {"id": 6, "product_name": "Carrot", "price": 1.0, "change": 0.1},
    {"id": 7, "product_name": "Potato", "price": 0.5, "change": -0.2},
    {"id": 8, "product_name": "Tomato", "price": 1.5, "change": 0.3},
]

market_basket_rules = [
    {"antecedents": "Milk", "consequents": "Bread", "confidence": 0.8, "support": 0.6, "lift": 1.2},
    {"antecedents": "Butter", "consequents": "Jam", "confidence": 0.7, "support": 0.5, "lift": 1.5},
    {"antecedents": "Eggs", "consequents": "Bacon", "confidence": 0.9, "support": 0.7, "lift": 1.8},
    {"antecedents": "Carrot", "consequents": "Potato", "confidence": 0.85, "support": 0.6, "lift": 1.4},
    {"antecedents": "Apple", "consequents": "Orange", "confidence": 0.7, "support": 0.5, "lift": 1.6},
    {"antecedents": "Tomato", "consequents": "Potato", "confidence": 0.75, "support": 0.6, "lift": 1.5},
    {"antecedents": "Peach", "consequents": "Apple", "confidence": 0.65, "support": 0.45, "lift": 1.2},
]

# Load trained price prediction model
price_model = joblib.load('price_prediction_model.pkl')

@app.route('/current-prices', methods=['GET'])
def get_current_prices():
    # Simulate price fluctuations
    for product in products:
        product["price"] += random.uniform(-0.1, 0.1)
        product["change"] = random.uniform(-1, 1)
    return jsonify(products)

@app.route('/price-prediction/<int:product_id>', methods=['GET'])
def get_price_prediction(product_id):
    if product_id > len(products):
        return jsonify({"error": "Product not found"}), 404

    # Predict price for the next 7 days
    now = datetime.now()
    predictions = []
    for i in range(7):
        day_of_year = (now + timedelta(days=i)).timetuple().tm_yday
        # Create a DataFrame with the correct column name for the model
        input_data = pd.DataFrame([[day_of_year]], columns=["day_of_year"])
        predicted_price = price_model.predict(input_data)[0]
        predictions.append({
            "ds": (now + timedelta(days=i)).isoformat(),
            "yhat": round(predicted_price, 2)
        })
    return jsonify(predictions)
    # return jsonify(predictions)
@app.route('/market-basket', methods=['GET'])
def get_market_basket_analysis():
    # Market basket analysis (using apriori and association rules)
    transactions = [
        {"Apple", "Banana", "Orange"},
        {"Apple", "Peach", "Banana"},
        {"Carrot", "Potato", "Tomato"},
        {"Apple", "Orange", "Potato"},
        {"Peach", "Apple", "Tomato"},
        {"Milk", "Bread"},
        {"Milk", "Butter", "Jam"},
    ]

    df = pd.DataFrame(columns=[p["product_name"] for p in products])
    for transaction in transactions:
        df = pd.concat([df, pd.DataFrame([{product["product_name"]: (product["product_name"] in transaction) for product in products}])], ignore_index=True)

    # Run apriori to generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    # If no rules are found, return static data
    if rules.empty:
        rules = pd.DataFrame([{
            "antecedents": "Apple",
            "consequents": "Orange",
            "confidence": 0.7,
            "support": 0.5,
            "lift": 1.6
        }, {
            "antecedents": "Milk",
            "consequents": "Bread",
            "confidence": 0.8,
            "support": 0.6,
            "lift": 1.2
        }, {
            "antecedents": "Carrot",
            "consequents": "Potato",
            "confidence": 0.85,
            "support": 0.6,
            "lift": 1.4
        }, {
            "antecedents": "Peach",
            "consequents": "Apple",
            "confidence": 0.65,
            "support": 0.45,
            "lift": 1.2
        }])  # Added more static data rules

    # Return the data as a list of records
    return jsonify({
        "rules": rules.to_dict('records'),
        "static_info": {
            "description": "Here you can see products that are frequently bought together based on transaction data analysis.",
            "example": "Example: If a customer buys 'Apple', they might also buy 'Orange'.",
            "title": "Frequently Bought Together"
        }
    })

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5001)
