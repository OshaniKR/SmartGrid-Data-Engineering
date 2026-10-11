
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_timestamp
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType,
)


# --------------------------------------------------
# 1. Load environment variables and project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "smart-meter-readings"

LAKE_PATH = PROJECT_ROOT / "data" / "lake" / "energy_readings"
LAKE_CHECKPOINT = PROJECT_ROOT / "spark" / "checkpoints" / "lake_stream"

POSTGRES_CHECKPOINT = (
    PROJECT_ROOT / "spark" / "checkpoints" / "postgres_stream"
)


# --------------------------------------------------
# 2. Define the incoming Kafka JSON schema
# --------------------------------------------------

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("household_id", StringType(), True),
    StructField("consumption_kwh", DoubleType(), True),
    StructField("solar_generation_kwh", DoubleType(), True),
    StructField("temperature_c", DoubleType(), True),
    StructField("humidity_percent", DoubleType(), True),
    StructField("hour", IntegerType(), True),
    StructField("day_of_week", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("is_weekend", IntegerType(), True),
    StructField("previous_consumption", DoubleType(), True),
])


# --------------------------------------------------
# 3. Create the Spark session
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SmartEnergyKafkaStreaming")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# --------------------------------------------------
# 4. Read streaming data from Kafka
# --------------------------------------------------

kafka_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_SERVER)
    .option("subscribe", KAFKA_TOPIC)
    .option("startingOffsets", "earliest")
    .option("maxOffsetsPerTrigger", 1000)
    .load()
)


# --------------------------------------------------
# 5. Parse and clean the incoming JSON records
# --------------------------------------------------

energy_df = (
    kafka_df
    .select(
        from_json(
            col("value").cast("string"),
            schema
        ).alias("data")
    )
    .select("data.*")
    .withColumn(
        "timestamp",
        to_timestamp(col("timestamp"))
    )
    .withColumn(
        "is_weekend",
        col("is_weekend").cast("boolean")
    )
    .filter(
        col("timestamp").isNotNull()
        & col("household_id").isNotNull()
        & col("consumption_kwh").isNotNull()
        & col("solar_generation_kwh").isNotNull()
    )
)


# --------------------------------------------------
# 6. Write each micro-batch to PostgreSQL
# --------------------------------------------------

def process_batch(batch_df, batch_id):

    if batch_df.isEmpty():
        return

    rows = batch_df.collect()

    connection_settings = {
        "host": os.getenv("POSTGRES_HOST"),
        "port": os.getenv("POSTGRES_PORT", "5432"),
        "dbname": os.getenv("POSTGRES_DB"),
        "user": os.getenv("POSTGRES_USER"),
        "password": os.getenv("POSTGRES_PASSWORD"),
    }

    if not all(connection_settings.values()):
        raise ValueError(
            "One or more PostgreSQL settings are missing from .env"
        )

    inserted = 0

    with psycopg.connect(**connection_settings) as conn:
        with conn.cursor() as cur:

            for row in rows:
                cur.execute(
                    """
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
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        row["timestamp"],
                        row["household_id"],
                        row["consumption_kwh"],
                        row["solar_generation_kwh"],
                        row["temperature_c"],
                        row["humidity_percent"],
                        row["hour"],
                        row["day_of_week"],
                        row["month"],
                        row["is_weekend"],
                        row["previous_consumption"],
                    ),
                )

                inserted += cur.rowcount

    print(
        f"PostgreSQL batch {batch_id}: "
        f"received {len(rows)} rows; "
        f"inserted {inserted} new rows."
    )


# --------------------------------------------------
# 7. Start the PostgreSQL streaming query
# --------------------------------------------------

postgres_query = (
    energy_df.writeStream
    .foreachBatch(process_batch)
    .option(
        "checkpointLocation",
        str(POSTGRES_CHECKPOINT)
    )
    .start()
)


# --------------------------------------------------
# 8. Start the Parquet data lake streaming query
# --------------------------------------------------

lake_query = (
    energy_df.writeStream
    .format("parquet")
    .option(
        "path",
        str(LAKE_PATH)
    )
    .option(
        "checkpointLocation",
        str(LAKE_CHECKPOINT)
    )
    .outputMode("append")
    .start()
)


# --------------------------------------------------
# 9. Keep both streaming queries running
# --------------------------------------------------

print(
    "Spark is reading Kafka and writing data to "
    "PostgreSQL and the Parquet data lake."
)

try:
    # Wait for either streaming query to terminate.
    spark.streams.awaitAnyTermination()

except KeyboardInterrupt:
    print("Stopping Spark streaming...")

finally:
    if postgres_query.isActive:
        postgres_query.stop()

    if lake_query.isActive:
        lake_query.stop()

    spark.stop()
    print("Spark streaming stopped.")