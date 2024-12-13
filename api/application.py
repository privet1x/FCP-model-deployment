from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Directory for storing models
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

@app.route("/continue-train", methods=["POST"])
def continue_train():
    try:
        # Retrieving data from the request
        model_name = request.form["model-name"]
        train_file = request.files["train-input"]
        new_model_name = request.form["new-model-name"]

        # Loading data
        train_data = pd.read_csv(train_file)
        X = train_data.iloc[:, :-1]  # All columns except the last
        y = train_data.iloc[:, -1]   # The last column - target

        # Loading the model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found."}), 404

        model, scaler = joblib.load(model_path)

        # Adjusting data to the correct number of features
        expected_features = scaler.n_features_in_
        current_features = X.shape[1]

        if current_features > expected_features:
            # Trimming extra columns
            X = X.iloc[:, :expected_features]
        elif current_features < expected_features:
            # Adding missing columns (filling with zeros)
            for _ in range(expected_features - current_features):
                X[current_features] = 0
                current_features += 1

        # Scaling data
        X_scaled = scaler.fit_transform(X)

        # Continuing training
        model.fit(X_scaled, y)

        # Saving the new model
        new_model_path = os.path.join(MODELS_DIR, f"{new_model_name}.joblib")
        joblib.dump((model, scaler), new_model_path)

        # Evaluating model quality
        y_pred = model.predict(X_scaled)
        accuracy = accuracy_score(y, y_pred)
        conf_mat = confusion_matrix(y, y_pred).tolist()

        return jsonify({
            "accuracy": accuracy,
            "confusion_matrix": conf_mat
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Retrieving data from the request
        model_name = request.form["model-name"]
        input_file = request.files["input"]

        # Loading data
        input_data = pd.read_csv(input_file, header=None)

        # Loading the model
        model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            return jsonify({"error": "Model not found."}), 404

        model, scaler = joblib.load(model_path)

        # Adjusting data to the correct number of features
        expected_features = scaler.n_features_in_
        current_features = input_data.shape[1]

        if current_features > expected_features:
            # Trimming extra columns
            input_data = input_data.iloc[:, :expected_features]
        elif current_features < expected_features:
            # Adding missing columns (filling with zeros)
            for _ in range(expected_features - current_features):
                input_data[current_features] = 0
                current_features += 1

        # Scaling data
        X_scaled = scaler.transform(input_data)

        # Predictions
        predictions = model.predict(X_scaled)

        return jsonify({"predictions": predictions.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/models", methods=["GET"])
def get_models():
    try:
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
        model_names = [os.path.splitext(f)[0] for f in model_files]
        return jsonify({"models": model_names})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
