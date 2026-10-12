
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "energy_features.csv"

# MLflow experiment settings
mlflow.set_tracking_uri(
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("SmartGrid-Energy-Forecasting")

# Features and target
FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "temperature_c",
    "humidity_percent",
    "solar_generation_kwh",
    "previous_consumption",
]
TARGET = "consumption_kwh"

TRAIN_RATIO = 0.80


def main():
    print("Loading energy dataset...")

    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Split chronologically to avoid training on future readings.
    timestamps = sorted(df["timestamp"].unique())
    split_index = int(len(timestamps) * TRAIN_RATIO)
    split_timestamp = timestamps[split_index]

    train_df = df[df["timestamp"] < split_timestamp].copy()
    test_df = df[df["timestamp"] >= split_timestamp].copy()

    train_df = train_df.dropna(subset=FEATURES + [TARGET])
    test_df = test_df.dropna(subset=FEATURES + [TARGET])

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")

    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_leaf=2,
            random_state=42,
        ),
    }

    for model_name, model in models.items():
        print(f"\nTraining and tracking {model_name}...")

        with mlflow.start_run(run_name=model_name):
            # Record model parameters.
            mlflow.log_params(model.get_params())
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("training_rows", len(X_train))
            mlflow.log_param("testing_rows", len(X_test))
            mlflow.log_param("train_ratio", TRAIN_RATIO)
            mlflow.log_param("features", ",".join(FEATURES))

            # Train and evaluate.
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)
            mse = mean_squared_error(y_test, predictions)
            rmse = mse ** 0.5
            r2 = r2_score(y_test, predictions)

            # Record evaluation metrics.
            mlflow.log_metrics({
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2),
            })

            # Save the trained model as an MLflow artifact.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="forecasting_model",
                input_example=X_train.head(3),
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )

            print(f"MAE:  {mae:.6f}")
            print(f"RMSE: {rmse:.6f}")
            print(f"R²:   {r2:.6f}")
            print("Parameters, metrics, and model logged to MLflow.")

    print("\nBoth models have been tracked successfully.")
    print("Experiment: SmartGrid-Energy-Forecasting")
    print(f"Tracking directory: {PROJECT_ROOT / 'mlruns'}")


if __name__ == "__main__":
    main()