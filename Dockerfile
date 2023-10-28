FROM apache/airflow:latest
COPY requirements.txt /
RUN pip install --no-cache-dir -r /requirements.txt
