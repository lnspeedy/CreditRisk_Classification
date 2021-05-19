import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import subprocess

def download_feed(target_folder): 
    try:
        #download dataset
        data = pd.read_csv('https://datahub.io/machine-learning/credit-g/r/credit-g.csv')

        #split dataset
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        
        train.to_csv(os.path.join(target_folder, "train.csv"))
        
        test.to_csv(os.path.join(target_folder, "test.csv"))

        print("Download succeeded")

    except Exception as e:
        print("Failed download :", e)
0
      
    
if __name__ == '__main__':
    env = os.getenv('Var_env')
    target_folder = env
    download_feed(target_folder)


