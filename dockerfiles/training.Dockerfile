FROM python:3.8-slim-buster
RUN apt-get update && apt-get install -y python3-dev build-essential

COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt

# set python path 
ENV PYTHONPATH "${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED 1

# app env variables 
ENV HOST "0.0.0.0"
ENV PORT "80"
ENV DATA_FOLDER "/app/datafeed/data" 
ENV MODEL_FOLDER "/app/api/ml/models"

# launch trainings
CMD bash launch_training.sh