info:
	./airflow.sh info

bash:
	./airflow.sh bash

bash_worker:
	docker exec -it `docker ps -ql -f name=airflow-worker-1` bash

python:
	./airflow.sh python
