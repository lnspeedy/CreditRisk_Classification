#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
import os
from urllib import request
from urllib.request import urlretrieve as retrieve


# In[20]:


data = pd.read_csv('https://datahub.io/machine-learning/credit-g/r/credit-g.csv')


# In[21]:


train,test = train_test_split(data,test_size=0.2,random_state=42)


# In[23]:


train.to_csv('train.csv')
test.to_csv('test.csv')
data.to_csv('credit.csv')


# In[ ]:




