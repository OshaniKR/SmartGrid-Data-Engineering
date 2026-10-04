from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "data/processed/energy_features.csv"
)

MODEL_FILE = Path(
    "models/energy_forecasting_random_forest.joblib"
)

OUTPUT_DIR = Path(
    "models/evaluation"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "forecast_predictions.csv"
)

ACTUAL_VS_PREDICTED_PLOT = (
    OUTPUT_DIR
    / "actual_vs_predicted.png"
)

FEATURE_IMPORTANCE_PLOT = (
    OUTPUT_DIR
    / "feature_importance.png"
)

ERROR_DISTRIBUTION_PLOT = (
    OUTPUT_DIR
    / "prediction_error_distribution.png"
)


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


# ============================================================
# Load data
# ============================================================

def load_data() -> pd.DataFrame:
    """Load the feature-engineered dataset."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# Prepare test data
# ============================================================

def prepare_test_data(
    df: pd.DataFrame,
):
    """Create the same chronological test set used during training."""

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    unique_timestamps = (
        df["timestamp"]
        .sort_values()
        .unique()
    )

    split_position = int(
        len(unique_timestamps)
        * TRAIN_RATIO
    )

    split_timestamp = (
        unique_timestamps[
            split_position
        ]
    )

    test_df = df[
        df["timestamp"] >= split_timestamp
    ].copy()

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return (
        test_df,
        X_test,
        y_test,
    )


# ============================================================
# Evaluate model
# ============================================================

def calculate_metrics(
    y_test,
    predictions,
):
    """Calculate forecasting metrics."""

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions,
    )

    return mae, rmse, r2


# ============================================================
# Plot actual vs predicted
# ============================================================

def plot_actual_vs_predicted(
    test_df: pd.DataFrame,
    predictions,
):
    """Create actual vs predicted consumption plot."""

    plt.figure(figsize=(12, 6))

    plt.plot(
        test_df["timestamp"],
        test_df[TARGET],
        label="Actual",
        alpha=0.7,
    )

    plt.plot(
        test_df["timestamp"],
        predictions,
        label="Predicted",
        alpha=0.7,
    )

    plt.title(
        "Actual vs Predicted Energy Consumption"
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Consumption (kWh)")

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        ACTUAL_VS_PREDICTED_PLOT,
        dpi=300,
    )

    plt.close()


# ============================================================
# Plot feature importance
# ============================================================

def plot_feature_importance(
    model,
):
    """Create feature importance plot."""

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": (
                model.feature_importances_
            ),
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        importance["feature"],
        importance["importance"],
    )

    plt.title(
        "Random Forest Feature Importance"
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.grid(
        axis="x"
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_IMPORTANCE_PLOT,
        dpi=300,
    )

    plt.close()


# ============================================================
# Plot prediction errors
# ============================================================

def plot_error_distribution(
    y_test,
    predictions,
):
    """Create prediction error distribution plot."""

    errors = (
        y_test.to_numpy()
        - predictions
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        errors,
        bins=40,
    )

    plt.title(
        "Prediction Error Distribution"
    )

    plt.xlabel(
        "Prediction Error (Actual - Predicted)"
    )

    plt.ylabel("Frequency")

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        ERROR_DISTRIBUTION_PLOT,
        dpi=300,
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading forecasting model..."
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    print(
        "\nLoading dataset..."
    )

    df = load_data()

    (
        test_df,
        X_test,
        y_test,
    ) = prepare_test_data(
        df
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    mae, rmse, r2 = calculate_metrics(
        y_test,
        predictions,
    )

    print(
        "\nModel Evaluation"
    )

    print(
        "================"
    )

    print(
        f"MAE  : {mae:.4f} kWh"
    )

    print(
        f"RMSE : {rmse:.4f} kWh"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results = test_df[
        [
            "timestamp",
            "household_id",
            TARGET,
        ]
    ].copy()

    results["predicted_consumption_kwh"] = (
        predictions
    )

    results["prediction_error_kwh"] = (
        results[TARGET]
        - results[
            "predicted_consumption_kwh"
        ]
    )

    results.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"\nPredictions saved to:"
        f"\n{PREDICTIONS_FILE}"
    )

    # --------------------------------------------------------
    # Create plots
    # --------------------------------------------------------

    print(
        "\nCreating visualizations..."
    )

    plot_actual_vs_predicted(
        test_df,
        predictions,
    )

    plot_feature_importance(
        model
    )

    plot_error_distribution(
        y_test,
        predictions,
    )

    print(
        "\nEvaluation completed successfully."
    )

    print(
        "\nGenerated files:"
    )

    print(
        f"- {ACTUAL_VS_PREDICTED_PLOT}"
    )

    print(
        f"- {FEATURE_IMPORTANCE_PLOT}"
    )

    print(
        f"- {ERROR_DISTRIBUTION_PLOT}"
    )


if __name__ == "__main__":
    main()