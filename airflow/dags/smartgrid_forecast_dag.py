
from datetime import datetime, timedelta
import os

import mlflow
import pandas as pd
import psycopg2
from airflow.sdk import dag, task


MODEL_PATH = (
    "/opt/smartgrid/mlruns/1/models/"
    "m-5f3ee900619c49bd87356f52d9e8fbe2/artifacts"
)

FORECAST_HOURS = 24
FORECAST_START = datetime(2026, 4, 1, 0, 0, 0)


def get_postgres_connection():
    return psycopg2.connect(
        host=os.environ["SMARTGRID_POSTGRES_HOST"],
        port=int(os.environ["SMARTGRID_POSTGRES_PORT"]),
        dbname=os.environ["SMARTGRID_POSTGRES_DB"],
        user=os.environ["SMARTGRID_POSTGRES_USER"],
        password=os.environ["SMARTGRID_POSTGRES_PASSWORD"],
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
        # 1. Read the latest valid real reading for every household.
        conn = get_postgres_connection()

        try:
            query = """
                SELECT DISTINCT ON (household_id)
                    household_id,
                    timestamp,
                    temperature_c,
                    humidity_percent,
                    solar_generation_kwh,
                    previous_consumption
                FROM public.energy_consumption
                WHERE household_id LIKE 'H%'
                  AND temperature_c IS NOT NULL
                  AND humidity_percent IS NOT NULL
                  AND solar_generation_kwh IS NOT NULL
                  AND previous_consumption IS NOT NULL
                ORDER BY household_id, timestamp DESC;
            """
            latest_readings = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        if latest_readings.empty:
            raise ValueError("No valid household readings found.")

        # 2. Load the trained model.
        model = mlflow.pyfunc.load_model(MODEL_PATH)

        # 3. Generate 24 hourly scenario-based forecasts per household.
        forecast_rows = []

        for _, reading in latest_readings.iterrows():
            household_id = str(reading["household_id"])
            last_timestamp = pd.to_datetime(reading["timestamp"])

            # Start after the household's last real reading.
            start_timestamp = max(
                FORECAST_START,
                (last_timestamp + timedelta(hours=1)).to_pydatetime(),
            )

            temperature = float(reading["temperature_c"])
            humidity = float(reading["humidity_percent"])
            solar = float(reading["solar_generation_kwh"])
            previous_consumption = float(
                reading["previous_consumption"]
            )

            for hour_offset in range(FORECAST_HOURS):
                forecast_timestamp = (
                    start_timestamp + timedelta(hours=hour_offset)
                )

                features = pd.DataFrame([{
                    "hour": forecast_timestamp.hour,
                    "day_of_week": forecast_timestamp.weekday(),
                    "month": forecast_timestamp.month,
                    "is_weekend": int(
                        forecast_timestamp.weekday() >= 5
                    ),
                    "temperature_c": temperature,
                    "humidity_percent": humidity,
                    "solar_generation_kwh": solar,
                    "previous_consumption": previous_consumption,
                }])

                prediction = float(model.predict(features)[0])

                # Consumption cannot be negative.
                prediction = max(0.0, prediction)

                forecast_rows.append({
                    "household_id": household_id,
                    "input_timestamp": forecast_timestamp,
                    "predicted_consumption_kwh": prediction,
                    "model_uri": MODEL_PATH,
                })

                # Use this prediction as the next hour's lag feature.
                previous_consumption = prediction

        # 4. Save forecasts without creating duplicates.
        save_conn = get_postgres_connection()

        try:
            with save_conn.cursor() as cur:
                insert_query = """
                    INSERT INTO public.forecast_results (
                        household_id,
                        input_timestamp,
                        predicted_consumption_kwh,
                        model_uri
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (household_id, input_timestamp)
                    DO NOTHING;
                """

                values = [
                    (
                        row["household_id"],
                        row["input_timestamp"],
                        row["predicted_consumption_kwh"],
                        row["model_uri"],
                    )
                    for row in forecast_rows
                ]

                cur.executemany(insert_query, values)
                inserted_count = cur.rowcount

            save_conn.commit()

        except Exception:
            save_conn.rollback()
            raise
        finally:
            save_conn.close()

        print(f"Households processed: {len(latest_readings)}")
        print(f"Forecast rows generated: {len(forecast_rows)}")
        print(f"New rows inserted: {inserted_count}")
        print(
            "Forecast window: "
            f"{forecast_rows[0]['input_timestamp']} to "
            f"{forecast_rows[-1]['input_timestamp']}"
        )
        print(
            "Scenario assumption: last observed weather and solar "
            "values are held constant; predictions are recursive."
        )

        return {
            "households_processed": int(len(latest_readings)),
            "forecast_rows_generated": len(forecast_rows),
            "new_rows_inserted": inserted_count,
        }

    forecast_energy_consumption()


smartgrid_forecast_dag()
