from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from my_asi_package.data_processing import load_and_process_data, predict_new_data, balance_data, train_model
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

app = FastAPI()

MODELS_DIR = "../data/models/"
os.makedirs(MODELS_DIR, exist_ok=True)

# All endpoints were tested successfully using curl
@app.post("/continue-train")
async def continue_train(
    model_name: str = Form(...),
    train_input: UploadFile = Form(...),
    new_model_name: str = Form(...),
):
    # Load existing model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    model, scalar = joblib.load(model_path)

    # Process uploaded training data using `load_and_process_data`
    temp_file = f"temp_{train_input.filename}"
    with open(temp_file, "wb") as f:
        f.write(await train_input.read())
    data = load_and_process_data(temp_file)  # Assigns fixed column names
    os.remove(temp_file)  # Cleanup temporary file

    # Balance the dataset if necessary
    data = balance_data(data)

    # Continue training the model
    model, scalar = train_model(data, model=model, scalar=scalar)

    # Calculate metrics on the updated model
    X = data.loc[:, data.columns != 'auth']
    y = data.loc[:, data.columns == 'auth']
    X_scaled = scalar.transform(X)
    y_pred = model.predict(X_scaled)
    accuracy = accuracy_score(y, y_pred)

    # Save the updated model
    new_model_path = os.path.join(MODELS_DIR, f"{new_model_name}.joblib")
    joblib.dump((model, scalar), new_model_path)

    return {
        "message": "Model training completed",
        "new_model_name": new_model_name,
        "metrics": {"accuracy": round(accuracy, 4)}
    }


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    fileInput: UploadFile = Form(...),
):
    # Load the model
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    model, scalar = joblib.load(model_path)

    # Process uploaded input data using `load_and_process_data`
    temp_file = f"temp_{fileInput.filename}"
    with open(temp_file, "wb") as f:
        f.write(await fileInput.read())
    data = load_and_process_data(temp_file)  # Assigns fixed column names
    os.remove(temp_file)  # Cleanup temporary file

    # Remove the target column ('auth') before making predictions
    if 'auth' in data.columns:
        data = data.drop(columns=['auth'])

    # Ensure the data has the correct format
    new_data = pd.DataFrame(scalar.transform(data), columns=data.columns)

    # Strip column names for compatibility
    predictions = model.predict(new_data.values)

    return {"predictions": predictions.tolist()}




@app.get("/models")
async def get_models():
    # List all available models
    models = [f[:-7] for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    return {"models": models}
