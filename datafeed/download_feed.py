
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('https://datahub.io/machine-learning/credit-g/r/credit-g.csv')


train,test = train_test_split(data,test_size=0.2,random_state=42)

train.to_csv(r"C:\Users\HP\Desktop\datafeed\data\train.csv")
test.to_csv(r"C:\Users\HP\Desktop\datafeed\data\test.csv")

if__name__ == '__main__' 