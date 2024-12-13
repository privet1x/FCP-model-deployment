import unittest
import os
import json
from api.application import app, MODELS_DIR
from io import BytesIO

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure that the models directory exists
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Creating a test model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import joblib
        import numpy as np

        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([0, 1, 0, 1])
        model = LogisticRegression()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)

        # Saving the model
        joblib.dump((model, scaler), os.path.join(MODELS_DIR, "test_model.joblib"))

    @classmethod
    def tearDownClass(cls):
        # Deleting the test model
        os.remove(os.path.join(MODELS_DIR, "test_model.joblib"))

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_get_models(self):
        response = self.app.get("/models")
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("test_model", data["models"])

    def test_predict(self):
        # Creating a test input file
        input_data = "1,2\n3,4\n5,6"
        input_file = BytesIO(input_data.encode("utf-8"))
        input_file.name = "input.csv"

        response = self.app.post(
            "/predict",
            data={
                "model-name": "test_model",
                "input": input_file
            },
            content_type="multipart/form-data"
        )

        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("predictions", data)

    def test_continue_train(self):
        # Creating a test training file
        train_data = "1,2,0\n3,4,1\n5,6,0"
        train_file = BytesIO(train_data.encode("utf-8"))
        train_file.name = "train.csv"

        response = self.app.post(
            "/continue-train",
            data={
                "model-name": "test_model",
                "train-input": train_file,
                "new-model-name": "new_test_model"
            },
            content_type="multipart/form-data"
        )

        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("accuracy", data)
        self.assertIn("confusion_matrix", data)

        # Deleting the new model
        os.remove(os.path.join(MODELS_DIR, "new_test_model.joblib"))

if __name__ == "__main__":
    unittest.main()
