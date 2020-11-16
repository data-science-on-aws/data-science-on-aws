#!/bin/bash

export AIRFLOW_HOME=~/airflow

pip install apache-airflow
pip install -r requirements.txt

airflow initdb

airflow webserver -p 8080 &

airflow scheduler &

