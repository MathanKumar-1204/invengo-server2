from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Example dataset of transactions (each row represents a transaction and items bought in that transaction)
transactions = [
    {"Apple", "Banana", "Orange"},
    {"Apple", "Peach", "Banana"},
    {"Carrot", "Potato", "Tomato"},
    {"Apple", "Orange", "Potato"},
    {"Peach", "Apple", "Tomato"},
    {"Milk", "Bread"},
    {"Milk", "Butter", "Jam"},
]

# Convert transactions into a pandas DataFrame format for apriori
df = pd.DataFrame(columns=["Apple", "Banana", "Orange", "Grapes", "Peach", "Carrot", "Potato", "Tomato", "Milk", "Bread", "Butter", "Jam", "Eggs", "Bacon"])

# Create a list to hold the rows and then concatenate them into the DataFrame
rows = []
for transaction in transactions:
    rows.append({product: (product in transaction) for product in df.columns})

# Use pd.concat to append the new rows to the DataFrame
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

# Ensure DataFrame is boolean (True for presence, False for absence of items)
df = df.astype(bool)

# Apply apriori algorithm to find frequent itemsets with reduced min_support
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Print frequent itemsets for debugging
print("Frequent Itemsets:")
print(frequent_itemsets)

# Generate association rules with a minimum lift threshold of 1.0
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Print the rules
print("Association Rules:")
print(rules)

# Save the rules to a CSV file
rules.to_csv('market_basket_rules.csv', index=False)
