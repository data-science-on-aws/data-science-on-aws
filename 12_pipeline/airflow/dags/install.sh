#!/bin/bash

export AIRFLOW_HOME=~/airflow

#pip install apache-airflow==2.0.0b2
pip install -r requirements.txt

airflow initdb
#airflow db init

airflow webserver -p 8080 &

airflow scheduler &

