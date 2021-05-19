import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path

def download_feed(target_folder): 
    try:
        #download dataset
        data = pd.read_csv('https://datahub.io/machine-learning/credit-g/r/credit-g.csv')
        
        target_folder = Path(__file__).parent / "data"

        #split dataset
        train, test = train_test_split(data, test_size=0.2, random_state=42)
        
        train.to_csv(target_folder/ "train.csv")
        
        test.to_csv(target_folder/ "test.csv")

        print("Download succeeded")

    except Exception as e:
        print(f"Failed download : {e}")
    
if __name__ == '__main__':
    download_feed('target_folder')



