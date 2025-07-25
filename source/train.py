import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
df = pd.read_csv('data/customer_data.csv')

# Encode categorical columns
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Split into features and target
X = df.drop(columns=['churn'])
y = df['churn']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, 'models/model.pkl')
print("✅ Model trained and saved successfully.")