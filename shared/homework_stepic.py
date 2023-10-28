"""
  - BUCKET заменить на свой;
  - EXPERIMENT_NAME и DAG_ID оставить как есть (ссылками на переменную NAME);
  - имена коннекторов: pg_connection и s3_connection;
  - данные должны читаться из таблицы с названием california_housing;
  - данные на S3 должны лежать в папках {NAME}/datasets/ и {NAME}/results/.
"""
import json
import logging
import mlflow
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, Literal

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

NAME = "" # TO-DO: Вписать свой ник в телеграме
BUCKET = "" # TO-DO: Вписать свой бакет
FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup",
    "Latitude", "Longitude"
]
TARGET = "MedHouseVal"
EXPERIMENT_NAME = NAME
DAG_ID = NAME

models =  # TO-DO: Создать словарь моделей
default_args = {
    # TO-DO: Заполнить своими данными: настроить владельца и политику retries.
}
dag = DAG(dag_id=DAG_ID,
          default_args=default_args,
          # TO-DO: Заполнить остальными параметрами.
          )


def init() -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать start_tiemstamp, run_id, experiment_name, experiment_id.
    metrics = {}

    # TO-DO 2 mlflow: Создать эксперимент с experiment_name=NAME. 
    # Добавить проверку на уже созданный эксперимент!
    # your code here.

    # TO-DO 3 mlflow: Создать parent run.
    with mlflow.start_run(
        # your code here.
        ) as parent_run:
            # your code here.


def get_data_from_postgres(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_download_start, data_download_end.
    # your code here.

    # TO-DO 2 connections: Создать коннекторы.
    # your code here.

    # TO-DO 3 Postgres: Прочитать данные.
    # your code here.

    # TO-DO 4 Postgres: Сохранить данные на S3 в формате pickle в папку {NAME}/datasets/.
    file name = f"{NAME}/datasets/california_housing.pkl"
    # your code here:


def prepare_data(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать data_preparation_start, data_preparation_end.
    # your code here.

    # TO-DO 2 connections: Создать коннекторы.
    # your code here.

    # TO-DO 3 S3: Прочитать данные с S3.
    file name = f"{NAME}/datasets/california_housing.pkl"
    # your code here.
    
    # TO-DO 4 Сделать препроцессинг.
    # your code here.

    # TO-DO 5 Разделить данные на train/test.
    # your code here.

    # TO-DO 6 Подготовить 4 обработанных датасета.
    # your code here.

    # Сохранить данные на S3 в папку {NAME}/datasets/.

def train_mlflow_model(model: Any, name: str, X_train: np.array,
                       X_test: np.array, y_train: pd.Series,
                       y_test: pd.Series) -> None:

    # TO-DO 1: Обучить модель.
    # your code here

    # TO-DO 2: Сделать predict.
    # your code here

    # TO-DO 3: Сохранить результаты обучения с помощью MLFlow.
    # your code here


def train_model(**kwargs) -> Dict[str, Any]:
    # TO-DO 1 metrics: В этом шаге собрать f"train_start_{model_name}" и f"train_end_{model_name}".
    # your code here.

    # TO-DO 2 connections: Создать коннекторы.
    # your code here.

    # TO-DO 3 S3: Прочитать данные с S3 из папки {NAME}/datasets/.
    # your code here.

    # TO-DO 4: Обучить модели и залогировать обучение с помощью MLFlow.
    with mlflow.start_run(run_id=run_id) as parent_run:
        with mlflow.start_run(
                # your code here
        ) as child_run:
             # your code here


def save_results(**kwargs) -> None:
    # TO-DO 1 metrics: В этом шаге собрать end_timestamp.

    # TO-DO 2: сохранить результаты обучения на S3 в файл {NAME}/results/{date}.json.
    file name = f"{NAME}/results/{date}.json"
    # your code here.
    


#################################### INIT DAG ####################################

task_init = # your code here.

task_get_data = # your code here.

task_prepare_data = # your code here.

training_model_tasks = [# your code here.]

task_save_results = # your code here.

# TO-DO: Прописать архитектуру DAG'a.