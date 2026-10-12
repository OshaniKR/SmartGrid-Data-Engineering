
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

REQUIRED_COLUMNS = [
    "timestamp",
    "household_id",
    "consumption_kwh",
    "solar_generation_kwh",
    "temperature_c",
    "humidity_percent",
]

QUALITY_QUERY = """
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT (timestamp, household_id)) AS unique_readings,
    COUNT(*) FILTER (WHERE timestamp IS NULL) AS missing_timestamp,
    COUNT(*) FILTER (WHERE household_id IS NULL) AS missing_household,
    COUNT(*) FILTER (WHERE consumption_kwh IS NULL) AS missing_consumption,
    COUNT(*) FILTER (WHERE solar_generation_kwh IS NULL) AS missing_solar,
    COUNT(*) FILTER (WHERE temperature_c IS NULL) AS missing_temperature,
    COUNT(*) FILTER (WHERE humidity_percent IS NULL) AS missing_humidity,
    COUNT(*) FILTER (WHERE consumption_kwh < 0) AS negative_consumption,
    COUNT(*) FILTER (WHERE solar_generation_kwh < 0) AS negative_solar,
    COUNT(*) FILTER (
        WHERE humidity_percent < 0 OR humidity_percent > 100
    ) AS invalid_humidity,
    COUNT(*) FILTER (
        WHERE temperature_c < -50 OR temperature_c > 70
    ) AS unusual_temperature
FROM public.energy_consumption;
"""


def main():
    db_config = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB", "smart_energy"),
        "user": os.getenv("POSTGRES_USER", "energy_user"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    with psycopg.connect(**db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(QUALITY_QUERY)
            columns = [item.name for item in cur.description]
            results = dict(zip(columns, cur.fetchone()))

    print("\nSMART GRID DATA QUALITY REPORT")
    print("=" * 40)

    for name, value in results.items():
        print(f"{name}: {value}")

    checks = [
        "missing_timestamp",
        "missing_household",
        "missing_consumption",
        "missing_solar",
        "missing_temperature",
        "missing_humidity",
        "negative_consumption",
        "negative_solar",
        "invalid_humidity",
        "unusual_temperature",
    ]

    duplicates = results["total_rows"] - results["unique_readings"]

    passed = duplicates == 0 and all(results[name] == 0 for name in checks)

    print(f"duplicate_readings: {duplicates}")
    print("\nRESULT:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()