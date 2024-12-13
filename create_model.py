import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Example data for creating the model
X_sample = [[3.6216, 8.6661, -2.8073, -0.44699],
            [4.5459, 8.1674, -2.4586, -1.4621]]
y_sample = [0, 1]

# Create the model
model = LogisticRegression()
scaler = StandardScaler()

# Scale the data
X_scaled = scaler.fit_transform(X_sample)

# Train the model
model.fit(X_scaled, y_sample)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump((model, scaler), "models/existing_model.joblib")

print("Test model 'existing_model.joblib' created in the models folder.")
