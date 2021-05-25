FROM ubuntu:19.10

COPY . /api

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

# set python path 
ENV PYTHONPATH "${PYTHONPATH}:/api"
ENV PYTHONUNBUFFERED 1
WORKDIR /api
EXPOSE 80

CMD python3 workflow_training.py