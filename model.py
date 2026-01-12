import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("heart_statlog_cleveland_hungary_final.csv")


# Features and target
X = df.drop("target", axis=1)  # Features
y = df["target"]               # Target (1 = Disease, 0 = No Disease)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, 'heart_disease_model1.pkl')
joblib.dump(scaler, 'scaler1.pkl')

print("Model and scaler saved successfully!")
