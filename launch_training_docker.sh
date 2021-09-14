#!/usr/bin/env bash

# env variables 
export LOCAL_DATA_FOLDER=/run/desktop/home/lnspeedy/Bureau/CreditRisk_Classification-fastapi/datafeed/data
export LOCAL_MODEL_FOLDER=/run/desktop/home/lnspeedy/Bureau/finance/CreditRisk_Classification/api/ml/models

# build the web app docker container 
sudo docker build . -f dockerfiles/training.Dockerfile -t training

# run the web container 
sudo docker run --rm -it -v $LOCAL_DATA_FOLDER:/app/datafeed/data -v $LOCAL_MODEL_FOLDER:/app/api/ml/models training