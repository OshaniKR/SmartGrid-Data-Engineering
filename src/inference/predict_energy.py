
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

mlflow.set_tracking_uri(
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)

MODEL_NAME = "SmartGrid-GradientBoosting"

sample = pd.DataFrame(
    [
        {
            "hour": 12,
            "day_of_week": 2,
            "month": 6,
            "is_weekend": 0,
            "temperature_c": 28.0,
            "humidity_percent": 70.0,
            "solar_generation_kwh": 1.5,
            "previous_consumption": 0.8,
        }
    ]
)

model = mlflow.sklearn.load_model(
    f"models:/{MODEL_NAME}/1"
)

prediction = model.predict(sample)[0]

print(f"Model: {MODEL_NAME}")
print(f"Predicted consumption: {prediction:.4f} kWh")