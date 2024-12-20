import sys
import os
from fastapi.testclient import TestClient
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from api.main import app

# Create a client for testing
client = TestClient(app)

# Path to test files and folders
TEST_MODELS_DIR = "data/models/"
TEST_DATASET_PATH = "data/dataset.csv"
TEST_MODEL_NAME = "test_model"
UPDATED_MODEL_NAME = "test_model_updated"

# Ensure the models directory exists
os.makedirs(TEST_MODELS_DIR, exist_ok=True)

# Tests

def test_get_models():
    """Test for GET /models"""
    print("Test GET /models started...")
    response = client.get("/models")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert "models" in response.json(), "Response should contain 'models' key"
    assert isinstance(response.json()["models"], list), "Models should be a list"
    print(f"Response: {response.json()}")
    print("Test GET /models completed successfully.")

def test_continue_train():
    """Test for POST /continue-train"""
    print("Test POST /continue-train started...")
    # Send a request for model retraining
    with open(TEST_DATASET_PATH, "rb") as train_file:
        response = client.post(
            "/continue-train",
            data={
                "model_name": TEST_MODEL_NAME,
                "new_model_name": UPDATED_MODEL_NAME
            },
            files={"train_input": train_file}
        )
    
    # Validate the response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    json_response = response.json()
    print(f"Response: {json_response}")
    assert "message" in json_response, "Response should contain 'message' key"
    assert json_response["message"] == "Model training completed", "Incorrect message in response"
    assert "new_model_name" in json_response, "Response should contain 'new_model_name' key"
    assert json_response["new_model_name"] == UPDATED_MODEL_NAME, "Incorrect new model name"
    print("Test POST /continue-train completed successfully.")

def test_predict():
    """Test for POST /predict"""
    print("Test POST /predict started...")
    # Send a request for prediction
    with open(TEST_DATASET_PATH, "rb") as input_file:
        response = client.post(
            "/predict",
            data={"model_name": UPDATED_MODEL_NAME},
            files={"input": input_file}
        )

    # Validate the response
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    json_response = response.json()
    print(f"Response: {json_response}")
    assert "predictions" in json_response, "Response should contain 'predictions' key"
    assert isinstance(json_response["predictions"], list), "Predictions should be a list"
    print("Test POST /predict completed successfully.")
