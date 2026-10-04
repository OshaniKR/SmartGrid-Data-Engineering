from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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

MODEL_DIR = Path(
    "models"
)

MODEL_FILE = (
    MODEL_DIR
    / "energy_forecasting_random_forest.joblib"
)

TARGET = "consumption_kwh"

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

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(
        drop=True
    )

    return df


# ============================================================
# Validate data
# ============================================================

def validate_data(
    df: pd.DataFrame,
) -> None:
    """Check required columns."""

    required_columns = (
        FEATURES + [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


# ============================================================
# Prepare train/test data
# ============================================================

def prepare_train_test_data(
    df: pd.DataFrame,
):
    """Create chronological training and testing sets."""

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

    train_df = df[
        df["timestamp"]
        < split_timestamp
    ].copy()

    test_df = df[
        df["timestamp"]
        >= split_timestamp
    ].copy()

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        TARGET
    ]

    X_test = test_df[
        FEATURES
    ]

    y_test = test_df[
        TARGET
    ]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
    )


# ============================================================
# Train model
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    """Train the Random Forest model."""

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """Calculate forecasting metrics."""

    predictions = model.predict(
        X_test
    )

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

    print("\nModel Evaluation")
    print("================")

    print(
        f"MAE  : {mae:.4f} kWh"
    )

    print(
        f"RMSE : {rmse:.4f} kWh"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return predictions


# ============================================================
# Feature importance
# ============================================================

def show_feature_importance(
    model,
):
    """Display model feature importance."""

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
        ascending=False,
    )

    print("\nFeature Importance")
    print("==================")

    print(
        importance.to_string(
            index=False
        )
    )


# ============================================================
# Save model
# ============================================================

def save_model(
    model,
):
    """Save trained model."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print(
        f"\nModel saved to: {MODEL_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "Loading feature-engineered data..."
    )

    df = load_data()

    print(
        f"Total rows: {len(df):,}"
    )

    validate_data(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
    ) = prepare_train_test_data(
        df
    )

    print(
        f"\nTraining period:"
        f"\n{train_df['timestamp'].min()}"
        f"\n to "
        f"{train_df['timestamp'].max()}"
    )

    print(
        f"\nTesting period:"
        f"\n{test_df['timestamp'].min()}"
        f"\n to "
        f"{test_df['timestamp'].max()}"
    )

    print(
        f"\nTraining rows: {len(X_train):,}"
    )

    print(
        f"Testing rows : {len(X_test):,}"
    )

    print(
        "\nTraining Random Forest..."
    )

    model = train_model(
        X_train,
        y_train,
    )

    print(
        "Training completed."
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    show_feature_importance(
        model
    )

    save_model(
        model
    )


if __name__ == "__main__":
    main()