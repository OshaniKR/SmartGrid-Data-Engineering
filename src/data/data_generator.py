from pathlib import Path

import numpy as np
import pandas as pd


# Configuration
# ---------------------------------------------------------

NUM_HOUSEHOLDS = 20
DAYS = 90
FREQUENCY = "1h"

OUTPUT_DIR = Path("data/raw")
OUTPUT_FILE = OUTPUT_DIR / "smart_energy.csv"


# ---------------------------------------------------------
# Generate timestamps
# ---------------------------------------------------------

timestamps = pd.date_range(
    start="2026-01-01",
    periods=DAYS * 24,
    freq=FREQUENCY
)


# ---------------------------------------------------------
# Generate energy data
# ---------------------------------------------------------

rows = []

rng = np.random.default_rng(42)

for household_number in range(1, NUM_HOUSEHOLDS + 1):

    household_id = f"H{household_number:03d}"

    household_factor = rng.uniform(0.7, 1.3)

    for timestamp in timestamps:

        hour = timestamp.hour

        # Base household consumption
        base_consumption = 0.8 * household_factor

        # Morning usage
        morning_peak = (
            1.5 * household_factor
            if 6 <= hour <= 9
            else 0
        )

        # Evening usage
        evening_peak = (
            2.5 * household_factor
            if 18 <= hour <= 22
            else 0
        )

        # Random variation
        noise = rng.normal(0, 0.25)

        consumption = (
            base_consumption
            + morning_peak
            + evening_peak
            + noise
        )

        consumption = max(consumption, 0.1)

        # Simulated solar generation
        if 6 <= hour <= 18:

            solar_factor = np.sin(
                np.pi * (hour - 6) / 12
            )

            solar_generation = (
                4.0 * solar_factor
                + rng.normal(0, 0.2)
            )

            solar_generation = max(
                solar_generation,
                0
            )

        else:
            solar_generation = 0

        # Simulated weather
        temperature = (
            27
            + 4 * np.sin(
                2 * np.pi * hour / 24
            )
            + rng.normal(0, 1)
        )

        humidity = (
            75
            - 10 * np.sin(
                2 * np.pi * hour / 24
            )
            + rng.normal(0, 3)
        )

        rows.append(
            {
                "timestamp": timestamp,
                "household_id": household_id,
                "consumption_kwh": round(
                    consumption,
                    3
                ),
                "solar_generation_kwh": round(
                    solar_generation,
                    3
                ),
                "temperature_c": round(
                    temperature,
                    2
                ),
                "humidity_percent": round(
                    humidity,
                    2
                ),
            }
        )


# ---------------------------------------------------------
# Create DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(rows)


# ---------------------------------------------------------
# Save dataset
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"Dataset created successfully: {OUTPUT_FILE}"
)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(df.head())