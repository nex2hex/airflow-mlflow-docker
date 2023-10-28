from datetime import datetime, timedelta
import io
import json
import logging
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago


DEFAULT_ARGS = {
    "email": "cocinzian@gmail.com",
    "email_on_failure": False,
    "email_on_retry": False,
    "retry": 3,
    "retry_delay": timedelta(minutes=1),
}

dag = DAG(
    dag_id="test_train",
    start_date=days_ago(2),
    catchup=False,
    schedule_interval="0 1 * * *",
    tags=["test"],
    default_args=DEFAULT_ARGS,
)

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

POSTGRES_HOOK_NAME = "postgres_system"
POSTGRES_TABLE_NAME = "user_california_housing"

S3_CONNECTION_NAME = "s3_connection" 
S3_BUCKET = "af-test"
S3_DATA_PATH = "datasets/california_housing.pkl"
S3_DATA_FITTED_PATH = "datasets/california_housing/{name}.pkl"

FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"]
TARGET = "MedHouseVal"


def _get_s3_objects():
    s3_hook = S3Hook(S3_CONNECTION_NAME)
    session = s3_hook.get_session(s3_hook.conn_config.region_name)
    resource = session.resource("s3", endpoint_url=s3_hook.conn_config.endpoint_url)
    
    return s3_hook, resource


def init() -> None:
    logger.info("Train pipeline started.")


def get_data_from_postgres() -> None:
    pg_hook = PostgresHook(POSTGRES_HOOK_NAME)
    con = pg_hook.get_conn()
    data = pd.read_sql_query(f"SELECT * FROM {POSTGRES_TABLE_NAME}", con)
    s3_hook, resource = _get_s3_objects()
    pickle_byte_obj = pickle.dumps(data)
    resource.Object (S3_BUCKET, S3_DATA_PATH) .put(Body=pickle_byte_obj)
    logger.info("Data download finished.")


def prepare_data() -> None:
    s3_hook, resource = _get_s3_objects()
    file = s3_hook.download_file(key=S3_DATA_PATH, bucket_name=S3_BUCKET)
    data = pd.read_pickle(file)
    
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    for name, data in zip(("X_train", "X_test", "y_train", "y_test"),
                          (X_train_fitted, X_test_fitted, y_train, y_test)):

        pickle_byte_obj = pickle.dumps(data)
        resource.Object(S3_BUCKET, S3_DATA_FITTED_PATH.format(name=name)).put(Body=pickle_byte_obj)
        
    logger.info("Data preparation finished.")
    
    
def train_model() -> None:
    s3_hook, resource = _get_s3_objects()
    data = {}
    for name in ("X_train", "X_test", "y_train", "y_test"):
        file = s3_hook.download_file(key=S3_DATA_FITTED_PATH.format(name=name), bucket_name=S3_BUCKET)
        data[name] = pd.read_pickle(file)
        
    model = RandomForestRegressor()
    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])
    
    result = {}
    result["r2_score"] = r2_score(data["y_test"], prediction)
    result["rmse"] = mean_squared_error(data ["y_test"], prediction)**0.5
    result["mae"] = median_absolute_error(data["y_test"], prediction)
    
    date = datetime.now().strftime("%Y_%m_%d_%H")
    json_bytes_object = json.dumps(result)
    resource.Object(S3_BUCKET, f"results/{date}.json").put(Body=json_bytes_object)
    
    logger.info("Model training finished.")


def save_results() -> None:
    logger.info("Success.")
    
    
task_init = PythonOperator(task_id="init", python_callable=init, dag=dag)
task_get_data = PythonOperator(task_id="get_data", python_callable=get_data_from_postgres, dag=dag)
task_prepare_data = PythonOperator(task_id="prepare_data", python_callable=prepare_data, dag=dag)
task_train_model = PythonOperator(task_id="train_model", python_callable=train_model, dag=dag)
task_save_results = PythonOperator(task_id="save_results", python_callable=save_results, dag=dag)
task_init >> task_get_data >> task_prepare_data >> task_train_model >> task_save_results