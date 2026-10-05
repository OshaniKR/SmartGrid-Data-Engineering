import os

import psycopg
from dotenv import load_dotenv


load_dotenv()


def get_connection():
    """Create and return a PostgreSQL database connection."""

    connection = psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    return connection


if __name__ == "__main__":
    connection = get_connection()

    print("PostgreSQL connection successful!")

    connection.close()