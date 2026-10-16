
from datetime import datetime

import mlflow
import pandas as pd
import psycopg2
import os

from airflow.sdk import dag, task


MODEL_PATH = (
    "/opt/smartgrid/mlruns/1/models/"
    "m-5f3ee900619c49bd87356f52d9e8fbe2/artifacts"
)


@dag(
    dag_id="smartgrid_forecast_dag",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["smartgrid", "forecasting", "mlflow", "postgres"],
)
def smartgrid_forecast_dag():

    @task
    def forecast_energy_consumption():
        # Connect to the existing PostgreSQL container
        conn = psycopg2.connect(
            host=os.environ["SMARTGRID_POSTGRES_HOST"],
            port=int(os.environ["SMARTGRID_POSTGRES_PORT"]),
            dbname=os.environ["SMARTGRID_POSTGRES_DB"],
            user=os.environ["SMARTGRID_POSTGRES_USER"],
            password=os.environ["SMARTGRID_POSTGRES_PASSWORD"],
        )

        try:
            query = """
                SELECT
                    household_id,
                    timestamp,
                    temperature_c,
                    humidity_percent,
                    solar_generation_kwh,
                    previous_consumption
                FROM energy_consumption
                WHERE temperature_c IS NOT NULL
                  AND humidity_percent IS NOT NULL
                  AND solar_generation_kwh IS NOT NULL
                  AND previous_consumption IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 1;
            """

            reading = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        if reading.empty:
            raise ValueError("No valid energy reading found in PostgreSQL.")

        timestamp = pd.to_datetime(reading.loc[0, "timestamp"])

        # Derive time features directly from the timestamp.
        features = pd.DataFrame(
            [
                {
                    "hour": timestamp.hour,
                    "day_of_week": timestamp.dayofweek,
                    "month": timestamp.month,
                    "is_weekend": int(timestamp.dayofweek >= 5),
                    "temperature_c": float(reading.loc[0, "temperature_c"]),
                    "humidity_percent": float(
                        reading.loc[0, "humidity_percent"]
                    ),
                    "solar_generation_kwh": float(
                        reading.loc[0, "solar_generation_kwh"]
                    ),
                    "previous_consumption": float(
                        reading.loc[0, "previous_consumption"]
                    ),
                }
            ]
        )

        model = mlflow.pyfunc.load_model(MODEL_PATH)
        prediction = float(model.predict(features)[0])

        print(f"Household: {reading.loc[0, 'household_id']}")
        print(f"Reading timestamp: {timestamp}")
        print(f"Actual consumption: not used as a model input")
        print(f"Predicted consumption: {prediction:.4f} kWh")

        return {
            "household_id": str(reading.loc[0, "household_id"]),
            "timestamp": str(timestamp),
            "predicted_consumption_kwh": prediction,
        }

    forecast_energy_consumption()


smartgrid_forecast_dag()
