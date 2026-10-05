from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
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

OUTPUT_FILE = Path(
    "models/model_comparison.csv"
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
# Prepare train/test data
# ============================================================

def prepare_train_test_data(
    df: pd.DataFrame,
):
    """Create chronological training and testing datasets."""

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
        df["timestamp"] < split_timestamp
    ].copy()

    test_df = df[
        df["timestamp"] >= split_timestamp
    ].copy()

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """Calculate model performance metrics."""

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

    return mae, rmse, r2


# ============================================================
# Train Random Forest
# ============================================================

def train_random_forest(
    X_train,
    y_train,
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
# Train Gradient Boosting
# ============================================================

def train_gradient_boosting(
    X_train,
    y_train,
):
    """Train the Gradient Boosting model."""

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


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

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = prepare_train_test_data(
        df
    )

    print(
        f"Training rows: {len(X_train):,}"
    )

    print(
        f"Testing rows : {len(X_test):,}"
    )

    results = []

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    print(
        "\nTraining Random Forest..."
    )

    random_forest = train_random_forest(
        X_train,
        y_train,
    )

    rf_mae, rf_rmse, rf_r2 = (
        evaluate_model(
            random_forest,
            X_test,
            y_test,
        )
    )

    print(
        f"Random Forest:"
        f"\nMAE  = {rf_mae:.4f}"
        f"\nRMSE = {rf_rmse:.4f}"
        f"\nR²   = {rf_r2:.4f}"
    )

    results.append(
        {
            "model": "Random Forest",
            "mae": rf_mae,
            "rmse": rf_rmse,
            "r2": rf_r2,
        }
    )

    # --------------------------------------------------------
    # Gradient Boosting
    # --------------------------------------------------------

    print(
        "\nTraining Gradient Boosting..."
    )

    gradient_boosting = (
        train_gradient_boosting(
            X_train,
            y_train,
        )
    )

    gb_mae, gb_rmse, gb_r2 = (
        evaluate_model(
            gradient_boosting,
            X_test,
            y_test,
        )
    )

    print(
        f"Gradient Boosting:"
        f"\nMAE  = {gb_mae:.4f}"
        f"\nRMSE = {gb_rmse:.4f}"
        f"\nR²   = {gb_r2:.4f}"
    )

    results.append(
        {
            "model": "Gradient Boosting",
            "mae": gb_mae,
            "rmse": gb_rmse,
            "r2": gb_r2,
        }
    )

    # --------------------------------------------------------
    # Compare models
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        "rmse"
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nModel Comparison"
    )

    print(
        "================"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    best_model = comparison.iloc[0][
        "model"
    ]

    print(
        f"\nBest model based on RMSE: "
        f"{best_model}"
    )

    print(
        f"\nComparison saved to:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()