#!/usr/bin/env bash

# python path
export PYTHONPATH=$PYTHONPATH:CreditRisk_Classification

# app env variables 
export PORT=8000
export DATA_FOLDER=/mnt/c/a_projets_perso/finance/CreditRisk_Classification/datafeed/data
export HOST=localhost
export MODEL_FOLDER=/mnt/c/a_projets_perso/finance/CreditRisk_Classification/api/ml/models

bash launch_training.sh