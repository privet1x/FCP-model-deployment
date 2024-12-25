from pycaret.classification import setup, compare_models, save_model
import pandas as pd
import os

dataset_path = "../data/dataset.csv"
model_name = "pycaret_model"
MODELS_DIR = "../data/models/"


# Load and process the dataset
data = pd.read_csv(dataset_path, header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']

# Set up PyCaret
clf_setup = setup(data, target='auth', verbose=False, session_id=42)

# Train and compare models
best_model = compare_models()

# Save the best model
model_path = os.path.join(MODELS_DIR, model_name)  # Construct the path
save_model(best_model, model_path)

print(f"Model '{model_name}' trained with PyCaret has been saved successfully.")