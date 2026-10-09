
"""Load processed smart-energy features into PostgreSQL."""

import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

# Project root: smart-energy-intelligence/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CSV_PATH = PROJECT_ROOT / "data" / "processed" / "energy_features.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "household_id",
    "consumption_kwh",
    "solar_generation_kwh",
    "temperature_c",
    "humidity_percent",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "previous_consumption",
]

INSERT_SQL = """
    INSERT INTO energy_consumption (
        timestamp,
        household_id,
        consumption_kwh,
        solar_generation_kwh,
        temperature_c,
        humidity_percent,
        hour,
        day_of_week,
        month,
        is_weekend,
        previous_consumption
    )
    VALUES (
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s
    )
"""


def main():
    # 1. Check that the source file exists.
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    # 2. Read and validate the dataset.
    df = pd.read_csv(CSV_PATH)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("The CSV contains no records.")

    if df[["timestamp", "household_id",
           "consumption_kwh", "solar_generation_kwh"]].isnull().any().any():
        raise ValueError("Required columns contain missing values.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="raise"
    )

    # Prevent duplicate household/timestamp readings in the input file.
    if df.duplicated(["timestamp", "household_id"]).any():
        raise ValueError(
            "Duplicate household/timestamp records found in the CSV."
        )

    # Convert 0/1 weekend indicators into PostgreSQL-compatible booleans.
    df["is_weekend"] = df["is_weekend"].map(
        lambda value: None if pd.isna(value) else bool(value)
    )

    # Convert pandas NaN values to Python None (SQL NULL).
    df = df[REQUIRED_COLUMNS].astype(object)
    df = df.where(pd.notna(df), None)

    records = list(df.itertuples(index=False, name=None))

    # 3. Connect using environment variables.
    password = os.getenv("DB_PASSWORD") or os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError(
            "Database password not found. Set DB_PASSWORD or "
            "POSTGRES_PASSWORD in your .env file."
        )

    connection_settings = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", "smart_energy"),
        "user": os.getenv("DB_USER", "energy_user"),
        "password": password,
    }

    print(f"Source file: {CSV_PATH}")
    print(f"Records to load: {len(records)}")
    print("Connecting to PostgreSQL...")

    # 4. Load all rows in one transaction.
    # Refuse to append a second copy to a non-empty table.
    with psycopg.connect(**connection_settings) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM energy_consumption")
            existing_count = cur.fetchone()[0]

            if existing_count != 0:
                raise RuntimeError(
                    f"Table already contains {existing_count} rows. "
                    "Load cancelled to prevent accidental duplication."
                )

            cur.executemany(INSERT_SQL, records)

            cur.execute("SELECT COUNT(*) FROM energy_consumption")
            final_count = cur.fetchone()[0]

            if final_count != len(records):
                raise RuntimeError(
                    f"Row-count mismatch: expected {len(records)}, "
                    f"found {final_count}."
                )

    print("Data load completed successfully.")
    print(f"Rows inserted: {len(records)}")
    print(f"Rows verified in PostgreSQL: {final_count}")


if __name__ == "__main__":
    main()