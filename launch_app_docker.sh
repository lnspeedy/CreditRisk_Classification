#!/bin/bash

# env variables 
export LOCAL_DATA_FOLDER=/run/desktop/mnt/host/c/a_projets_perso/finance/CreditRisk_Classification/datafeed/data
export LOCAL_MODEL_FOLDER=/run/desktop/mnt/host/c/a_projets_perso/finance/CreditRisk_Classification/api/ml/models

# build the web app docker container 
docker build . -f dockerfiles/web.Dockerfiles

# run the web container 
docker run --rm -it -v $LOCAL_DATA_FOLDER:/app/datafeed/data -v $LOCAL_MODEL_FOLDER:/app/api/ml/models web