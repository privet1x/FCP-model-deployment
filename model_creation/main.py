import os
import joblib
from my_asi_package.data_processing import load_and_process_data, balance_data, train_model

MODELS_DIR = "../data/models/"
os.makedirs(MODELS_DIR, exist_ok=True)

dataset_path = "../data/dataset.csv"

model_name = "initial_model"

print("Loading and processing dataset...")
data = load_and_process_data(dataset_path)

print("Balancing dataset...")
data = balance_data(data)

print("Training model...")
model, scalar = train_model(data)

model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
print(f"Saving model to {model_path}...")
joblib.dump((model, scalar), model_path)

print(f"Model '{model_name}' has been created and saved successfully.")
