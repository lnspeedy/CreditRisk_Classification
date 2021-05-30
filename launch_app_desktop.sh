#!/usr/bin/env bash

# python path
export PYTHONPATH=$PYTHONPATH:/usr/bin/python3.8

# app env variables 
export PORT=8000
export DATA_FOLDER=/home/lnspeedy/Bureau/CreditRisk_Classification-fastapi/datafeed/data
export HOST=localhost
export MODEL_FOLDER=/home/lnspeedy/Bureau/CreditRisk_Classification-fastapi/api/ml/models

python3 ./api/main.py
