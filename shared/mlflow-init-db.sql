CREATE DATABASE mlflow_db;
CREATE USER mlflow WITH PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;
GRANT ALL ON SCHEMA public TO mlflow;