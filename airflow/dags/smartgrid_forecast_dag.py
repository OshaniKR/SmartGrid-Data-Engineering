
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

        household_id = str(reading.loc[0, "household_id"])

        # Save the prediction to PostgreSQL.
        save_conn = psycopg2.connect(
            host=os.environ["SMARTGRID_POSTGRES_HOST"],
            port=int(os.environ["SMARTGRID_POSTGRES_PORT"]),
            dbname=os.environ["SMARTGRID_POSTGRES_DB"],
            user=os.environ["SMARTGRID_POSTGRES_USER"],
            password=os.environ["SMARTGRID_POSTGRES_PASSWORD"],
        )
        try:
            with save_conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO forecast_results (
                        household_id,
                        input_timestamp,
                        predicted_consumption_kwh,
                        model_uri
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (household_id, input_timestamp)
                    DO NOTHING
                    RETURNING id, forecasted_at;
                    """,
                    (
                        household_id,
                        timestamp.to_pydatetime(),
                        prediction,
                        MODEL_PATH,
                    ),
                )

                result = cur.fetchone()

                if result is not None:
                    forecast_id, forecasted_at = result
                    print("New forecast saved successfully.")
                else:
                    cur.execute(
                        """
                        SELECT id, forecasted_at, predicted_consumption_kwh
                        FROM forecast_results
                        WHERE household_id = %s
                          AND input_timestamp = %s;
                        """,
                        (household_id, timestamp.to_pydatetime()),
                    )
                    existing = cur.fetchone()

                    if existing is None:
                        raise RuntimeError(
                            "Forecast was not inserted or found in the database."
                        )

                    forecast_id, forecasted_at, existing_prediction = existing
                    prediction = float(existing_prediction)
                    print("Forecast already exists; using the existing record.")

            save_conn.commit()
        finally:
            save_conn.close()


        print(f"Household: {household_id}")
        print(f"Reading timestamp: {timestamp}")
        print(f"Predicted consumption: {prediction:.4f} kWh")
        print(f"Saved forecast ID: {forecast_id}")
        print(f"Forecast saved at: {forecasted_at}")

        return {
            "forecast_id": forecast_id,
            "household_id": household_id,
            "timestamp": str(timestamp),
            "predicted_consumption_kwh": prediction,
            "forecasted_at": str(forecasted_at),
        }


    forecast_energy_consumption()




smartgrid_forecast_dag()
