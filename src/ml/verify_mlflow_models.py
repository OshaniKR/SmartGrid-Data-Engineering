from pathlib import Path
import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "mlruns" / "1" / "models"

models = {
    "Gradient Boosting": MODEL_DIR / "m-5f3ee900619c49bd87356f52d9e8fbe2" / "artifacts",
    "Random Forest": MODEL_DIR / "m-9e1af321ef5d4c2aa627cf6ccf3b73ee" / "artifacts",
}

sample = pd.DataFrame([{
    "hour": 12,
    "day_of_week": 2,
    "month": 6,
    "is_weekend": 0,
    "temperature_c": 28.0,
    "humidity_percent": 70.0,
    "solar_generation_kwh": 1.5,
    "previous_consumption": 0.8,
}])

for name, artifact_dir in models.items():
    print(f"\nTesting {name}...")
    try:
        model = mlflow.sklearn.load_model(str(artifact_dir))
        prediction = model.predict(sample)
        print(f"PASS: Model loaded and prediction succeeded: {prediction[0]:.4f}")
        del model
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
