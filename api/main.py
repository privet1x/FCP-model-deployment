from fastapi import FastAPI, Form
import pandas as pd
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = FastAPI()

# Path to the model and dataset
MODELS_DIR = "data/models/"
os.makedirs(MODELS_DIR, exist_ok=True)
DATASET_PATH = "data/dataset.csv"

@app.post("/continue-train")
async def continue_train(
    model_name: str = Form(...),
    new_model_name: str = Form(...),
):
    # Loading the existing model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    model = joblib.load(model_path)

    # Loading data from dataset.csv
    dataset = pd.read_csv(DATASET_PATH, header=None)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    # Training the model
    model.fit(X, y)

    # Calculating metrics
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)

    # Saving the new model
    new_model_path = os.path.join(MODELS_DIR, f"{new_model_name}.joblib")
    joblib.dump(model, new_model_path)

    return {
        "message": "Model training completed",
        "new_model_name": new_model_name,
        "metrics": {"accuracy": accuracy},
    }

@app.post("/predict")
async def predict(
    model_name: str = Form(...),
):
    # Loading the model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        return {"error": "Model not found"}
    model = joblib.load(model_path)

    # Loading data for predictions
    dataset = pd.read_csv(DATASET_PATH, header=None)
    X = dataset.iloc[:, :-1]  # Take all columns except the last one

    # Making predictions
    predictions = model.predict(X).tolist()

    return {"predictions": predictions}

@app.get("/models")
async def get_models():
    # List of all models
    models = [f[:-7] for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    return {"models": models}
