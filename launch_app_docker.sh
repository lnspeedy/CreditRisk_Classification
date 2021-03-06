#!/usr/bin/env bash

# env variables 
export LOCAL_DATA_FOLDER=/run/desktop/home/lnspeedy/Bureau/CreditRisk_Classification/datafeed/data
export LOCAL_MODEL_FOLDER=/run/desktop/home/lnspeedy/Bureau/CreditRisk_Classification/api/ml/models

# build the web app docker container 
sudo docker build . -f dockerfiles/web.Dockerfile -t web

# run the web container 
# sudo docker run --rm -it -v $LOCAL_DATA_FOLDER:/app/datafeed/data -p 8000:80 web
sudo docker run --rm -it -v $LOCAL_DATA_FOLDER:/app/datafeed/data -v $LOCAL_MODEL_FOLDER:/app/api/ml/models -p 8000:80 web