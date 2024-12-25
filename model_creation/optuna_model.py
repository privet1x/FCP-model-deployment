import os
import joblib
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Constants
MODELS_DIR = "../data/models/"
os.makedirs(MODELS_DIR, exist_ok=True)

dataset_path = "../data/dataset.csv"
model_name = "optuna_model"


# Load and process the dataset
def load_and_process_data(file_path):
    data = pd.read_csv(file_path, header=None)
    data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
    return data


data = load_and_process_data(dataset_path)
X = data.loc[:, data.columns != 'auth']
y = data.loc[:, 'auth']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Objective function for Optuna
def objective(trial):
    # Hyperparameter suggestions
    solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
    C = trial.suggest_float("C", 0.01, 10.0, log=True)

    # Model setup
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    model = LogisticRegression(solver=solver, C=C, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Run the optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)

# Save the best model
best_params = study.best_params
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)
model = LogisticRegression(**best_params, random_state=42)
model.fit(X_scaled, y.values.ravel())


model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
joblib.dump((model, scalar), model_path)


print(f"Model '{model_name}' optimized with Optuna has been saved successfully.")