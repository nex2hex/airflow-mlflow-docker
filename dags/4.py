import datetime
import json
import logging
import pickle
import typing as t

import mlflow
import pandas as pd
from airflow.decorators import dag, task
from airflow.models.xcom import LazyXComAccess
from airflow.operators.python import get_current_context
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# !!!!!!!!!!! CHECK THESE VARS
POSTGRES_HOOK_NAME = 'postgres_system'
S3_HOOK_NAME = 's3_connection'
S3_BUCKET = 'af-test'


POSTGRES_TABLE_NAME = 'user_california_housing'
MLFLOW_EXPERIMENT_NAME = 'foo'
MLFLOW_S3_ARTIFACT_LOCATION = f's3://{S3_BUCKET}/mlflow/{MLFLOW_EXPERIMENT_NAME}'
S3_DATA_PATH = f'{MLFLOW_EXPERIMENT_NAME}/datasets/california_housing.pkl'
S3_DATA_FITTED_PATH = f'{MLFLOW_EXPERIMENT_NAME}/datasets/california_housing/{{name}}.pkl'
S3_DATA_RESULT_PATH = f'{MLFLOW_EXPERIMENT_NAME}/results/{{current_datetime}}.json'


DEFAULT_ARGS = {
    'email': 'foo@example.com',
    'email_on_failure': False,
    'email_on_retry': False,
    'retry': 3,
    'retry_delay': datetime.timedelta(minutes=1),
}


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())


FEATURES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
            'Population', 'AveOccup', 'Latitude', 'Longitude']
TARGET = 'MedHouseVal'


MODELS = [{
    'name': 'RandomForestRegressor',
    'class': RandomForestRegressor
}, {
    'name': 'LinearRegression',
    'class': LinearRegression
}, {
    'name': 'HistGradientBoostingRegressor',
    'class': HistGradientBoostingRegressor
}]


def _get_s3_objects():
    s3_hook = S3Hook(S3_HOOK_NAME)
    session = s3_hook.get_session(s3_hook.conn_config.region_name or "ru-central1")
    resource = session.resource('s3', endpoint_url=s3_hook.conn_config.endpoint_url or "https://storage.yandexcloud.net")
    
    return s3_hook, resource


def resume_mlflow_run() -> mlflow.ActiveRun:
    context = get_current_context()
    ti = context['ti']
    mlflow_run_id = ti.xcom_pull(key='mlflow_run_id')
    return mlflow.start_run(run_id=mlflow_run_id)


def now_as_ts() -> float:
    return datetime.datetime.now().timestamp()


@dag(
    start_date=days_ago(2),
    catchup=False,
    schedule_interval='0 1 * * *',
    tags=['test'],
    default_args=DEFAULT_ARGS,
)
def airflow_final():
    """
    https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html
    """
    MetricsType = t.Dict[str, t.Any]

    @task()
    def init() -> MetricsType:
        # https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.create_experiment
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        experiment_id = None
        if experiment:
            experiment_id = experiment.experiment_id
        if not experiment_id:
            experiment_id = mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=f'{MLFLOW_S3_ARTIFACT_LOCATION}'
            )
        mlflow.set_experiment(experiment_id=experiment_id)

        context = get_current_context()
        run_name = f'{MLFLOW_EXPERIMENT_NAME} {context["ts"]}'
        run = mlflow.start_run(run_name=run_name)
        mlflow_run_id = run.info.run_id

        ti = context['ti']
        ti.xcom_push('mlflow_run_id', mlflow_run_id)
        ti.xcom_push('mlflow_experiment_id', experiment_id)

        metrics = {}
        metrics['start_timestamp'] = now_as_ts()
        metrics['run_id'] = mlflow_run_id
        metrics['experiment_name'] = MLFLOW_EXPERIMENT_NAME
        metrics['experiment_id'] = experiment_id

        return metrics


    @task()
    def get_data_from_postgres(**kwargs) -> MetricsType:
        metrics = {}
        metrics['data_download_start'] = now_as_ts()

        s3_hook, resource = _get_s3_objects()
        is_file_exists = s3_hook.head_object(key=S3_DATA_PATH, bucket_name=S3_BUCKET)

        if is_file_exists:
            logger.info(f'Skip downloading data, file {S3_DATA_PATH} exists.')
            return

        pg_hook = PostgresHook(POSTGRES_HOOK_NAME)
        con = pg_hook.get_conn()
        data = pd.read_sql_query(f'SELECT * FROM {POSTGRES_TABLE_NAME}', con)
        pickle_byte_obj = pickle.dumps(data)
        resource.Object(S3_BUCKET, S3_DATA_PATH).put(Body=pickle_byte_obj)

        logger.info('Data download finished.')

        metrics['data_download_end'] = now_as_ts()
        return metrics


    @task()
    def prepare_data(**kwargs) -> MetricsType:
        metrics = {}
        metrics['data_preparation_start'] = now_as_ts()

        s3_hook, resource = _get_s3_objects()
        x_train_file = S3_DATA_FITTED_PATH.format(name='X_train')
        is_file_exists = s3_hook.head_object(key=x_train_file, bucket_name=S3_BUCKET)

        if is_file_exists:
            logger.info(f'Skip data preparation, file {x_train_file} exists.')
            return

        file = s3_hook.download_file(key=S3_DATA_PATH, bucket_name=S3_BUCKET)
        data = pd.read_pickle(file)

        X, y = data[FEATURES], data[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_fitted = scaler.fit_transform(X_train)
        X_test_fitted = scaler.transform(X_test)

        for name, data in zip(('X_train', 'X_test', 'y_train', 'y_test'),
                              (X_train_fitted, X_test_fitted, y_train, y_test)):

            pickle_byte_obj = pickle.dumps(data)
            resource.Object(S3_BUCKET, S3_DATA_FITTED_PATH.format(name=name)).put(Body=pickle_byte_obj)

        logger.info('Data preparation finished.')

        metrics['data_preparation_end'] = now_as_ts()
        return metrics


    @task()
    def train_model(model_name, model_class, **kwargs) -> MetricsType:
        metrics = {}
        metrics[f'train_start_{model_name}'] = now_as_ts()

        s3_hook, resource = _get_s3_objects()
        data = {}
        for name in ('X_train', 'X_test', 'y_train', 'y_test'):
            file = s3_hook.download_file(key=S3_DATA_FITTED_PATH.format(name=name), bucket_name=S3_BUCKET)
            data[name] = pd.read_pickle(file)

        parent_run = resume_mlflow_run()
        with parent_run:
            with mlflow.start_run(experiment_id=parent_run.info.experiment_id, run_name=model_name, nested=True):
                model = model_class()
                model.fit(data['X_train'], data['y_train'])
                prediction = model.predict(data['X_test'])

                # mlflow.log_metric(f'r2_score {model_name}', r2_score(data['y_test'], prediction))
                # mlflow.log_metric(f'rmse {model_name}', mean_squared_error(data ['y_test'], prediction)**0.5)
                # mlflow.log_metric(f'mae {model_name}', median_absolute_error(data['y_test'], prediction))

                signature = infer_signature(data['X_test'], prediction)
                model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
                # Сохранить метрики модели
                mlflow.evaluate(
                    model_info.model_uri,
                    data=data['X_test'],
                    targets=data['y_test'].values,
                    model_type='regressor',
                    evaluators=['default'],
                )
                # mlflow.sklearn.save_model(model, model_name)

        logger.info('Model training finished.')

        metrics[f'train_end_{model_name}'] = now_as_ts()
        return metrics


    @task()
    def terminate(task_ids, **kwargs) -> None:
        resume_mlflow_run()
        mlflow.end_run()

        context = get_current_context()
        ti = context['ti']
        metrics_list: LazyXComAccess = ti.xcom_pull(task_ids=task_ids)

        metrics = {}
        for item in metrics_list:
            metrics.update(item)

        metrics['end_timestamp'] = now_as_ts()

        s3_hook, resource = _get_s3_objects()
        path = S3_DATA_RESULT_PATH.format(current_datetime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        resource.Object(S3_BUCKET, path).put(Body=json.dumps(metrics))


    init_task = init()
    get_data_from_postgres_task = get_data_from_postgres()
    prepare_data_task = prepare_data()

    train_model_tasks = []
    for model_data in MODELS:
        train_model_tasks.append(
            train_model(model_data['name'], model_data['class'])
        )

    task_ids = [item.operator.task_id for item in train_model_tasks]
    task_ids.append(init_task.operator.task_id)
    task_ids.append(get_data_from_postgres_task.operator.task_id)
    task_ids.append(prepare_data_task.operator.task_id)

    terminate_task = terminate(task_ids)

    init_task >> get_data_from_postgres_task >> prepare_data_task >> train_model_tasks >> terminate_task


airflow_final()
