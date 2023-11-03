### Docs
https://stepik.org/course/181476/syllabus

https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#fetching-docker-compose-yaml

### Run
1. Run in the shell
```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.1/docker-compose.yaml'
mkdir -p ./dags ./logs ./plugins ./config ./shared
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

2. Edit your `.env` file, see example in `.env_example`

3. Run `docker compose up`

### Useful commands
1. Running the CLI commands https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html#running-the-cli-commands Shortcuts added to Makefile.
2. Run `docker compose stop` to stop containers, `docker compose start` to start them
3. Run `docker compose down &&  docker compose up` to re-create containers after changing `docker-compose.yaml`
