from fastapi import FastAPI

import numpy as np
from pydantic import BaseModel
import mlflow.sklearn
from sklearn.datasets import load_iris
import uvicorn
import mlflow
import os


# ✅ Define FastAPI app
app = FastAPI()
run_id = "472513584962431997"  # Replace with your actual run ID
experiment_id = "4e39b1ad4ae943af92b928c272995d55"  # Replace with actual experiment ID

model_path = os.path.join("mlruns", run_id, experiment_id, "artifacts", "HeartClassification_model"
)
#472513584962431997\4e39b1ad4ae943af92b928c272995d55\artifacts\HeartClassification_model
print("🚀 Checking model path:", model_path)  


# Ensure the model file exists before loading
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at {model_path}")

# Load the model
model = mlflow.sklearn.load_model(model_path)

class HeartFeatures(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    """Welcome Message"""
    api_requests.inc()
    return {"message": "Heart Attack Predection app"}

@app.post("/predict")
def predict(features: HeartFeatures):

    data = np.array([[
    int(features.age), int(features.sex), int(features.cp), 
    int(features.trestbps), int(features.chol), int(features.fbs), 
    int(features.restecg), int(features.thalach), int(features.exang), 
    float(features.oldpeak), int(features.slope), int(features.ca), int(features.thal)
        ]])

    prediction = model.predict(data)  
    print(type(prediction))         # Should be numpy.ndarray
    print(type(prediction[0]))      # Likely numpy.int64 or numpy.float64


    # Convert NumPy types to native Python types
    prediction_value = int(prediction[0])  # Ensures response is JSON serializable

    return {"prediction": prediction_value}  

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)