#!/bin/bash

#creation of the environment variable. You will need to specify the path of your target folder in the ribs
export DATA_FOLDER='/app/datafeed/data' 

#python script execution
python3 datafeed/download_feed.py