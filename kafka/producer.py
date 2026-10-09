
import json
from pathlib import Path

import pandas as pd
from confluent_kafka import Producer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "energy_features.csv"

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "smart-meter-readings"


def delivery_report(error, message):
    if error:
        print(f"Delivery failed: {error}")
    else:
        print(f"Delivered message to {message.topic()}")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from CSV")

    producer = Producer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS
    })

    for record in df.to_dict(orient="records"):
        message = json.dumps(
            record,
            default=str,
            allow_nan=False
        )

        producer.produce(
            KAFKA_TOPIC,
            value=message,
            callback=delivery_report
        )

        # Process callbacks and allow queued messages to send
        producer.poll(0)

    remaining = producer.flush(30)

    if remaining:
        print(f"Warning: {remaining} messages remain undelivered")
    else:
        print("All queued messages have been delivered successfully")


if __name__ == "__main__":
    main()