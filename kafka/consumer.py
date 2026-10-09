
import json
import time

from confluent_kafka import Consumer


KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_TOPIC = "smart-meter-readings"


def main():
    consumer = Consumer({
        "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
        "group.id": "smart-energy-verification",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    consumer.subscribe([KAFKA_TOPIC])

    message_count = 0
    idle_start = time.monotonic()
    idle_timeout = 10

    print(f"Listening to topic: {KAFKA_TOPIC}")

    try:
        while True:
            message = consumer.poll(1.0)

            if message is None:
                if time.monotonic() - idle_start >= idle_timeout:
                    break
                continue

            if message.error():
                print(f"Kafka error: {message.error()}")
                continue

            idle_start = time.monotonic()

            # Convert JSON message back into a Python dictionary
            record = json.loads(message.value().decode("utf-8"))
            message_count += 1

            # Display only the first five records
            if message_count <= 5:
                print(f"\nRecord {message_count}:")
                print(json.dumps(record, indent=2))

    except KeyboardInterrupt:
        print("Consumer stopped by user.")

    finally:
        consumer.close()
        print(f"\nTotal messages consumed: {message_count}")


if __name__ == "__main__":
    main()