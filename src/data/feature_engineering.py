from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/raw/smart_energy.csv")
OUTPUT_FILE = Path("data/processed/energy_features.csv")


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-related features from the timestamp column.
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["hour"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df


def create_lag_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create previous-consumption features
    for each household.
    """

    df = df.copy()

    df = df.sort_values(
        ["household_id", "timestamp"]
    )

    df["previous_consumption"] = (
        df.groupby("household_id")
        ["consumption_kwh"]
        .shift(1)
    )

    return df


def main():
    # 1. Load raw data
    df = pd.read_csv(INPUT_FILE)

    print("Original data:")
    print(df.head())

    # 2. Create time features
    df = create_time_features(df)

    # 3. Create lag features
    df = create_lag_features(df)

    # 4. Remove rows where lag value doesn't exist
    df = df.dropna(
        subset=["previous_consumption"]
    )

    # 5. Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # 6. Save processed data
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFeature engineering completed!")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head())


if __name__ == "__main__":
    main()