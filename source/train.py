import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('data/customer_data.csv')
X = df.drop(columns=['churn'])
y = df['churn']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'models/model.pkl')
print("✅ Model trained and saved.")