from datetime import datetime

from airflow.sdk import dag, task


@dag(
    dag_id="smartgrid_first_dag",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["smartgrid", "test"],
)
def smartgrid_first_dag():

    @task
    def verify_airflow():
        print("SmartGrid: Airflow is working successfully!")
        return "Airflow test completed"

    verify_airflow()


smartgrid_first_dag()
