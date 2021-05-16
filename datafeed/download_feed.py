import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os



def download_feed(): 
    try:
        #download dataset
        data = pd.read_csv('https://datahub.io/machine-learning/credit-g/r/credit-g.csv')
        
        #split dataset
        train,test = train_test_split(data, test_size=0.2, random_state=42)
    
        target_folder = 'C:\\Users\\HP\\Desktop\\datafeed\\data\\'

        train.to_csv(os.target_folder.join(target_folder, r'train.csv'))
        test.to_csv(os.target_folder.join(target_folder, r'test.csv'))

        print("Download succeeded")

    except Exception as e:
        print(f"Failed download : {e}")
    
    


if __name__ == '__main__':
    download_feed()



