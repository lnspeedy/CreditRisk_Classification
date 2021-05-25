import json
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import pandas as pd

##################################################
# CATCH ERRORS
##################################################

# class to raise credit risk exceptions
class CreditRiskException(Exception):
    def __init__(self, message, errors):

        jsonresult = {}
        jsonresult["error_title"] = message
        jsonresult["error"] = errors

        # call the base class constructor
        if sys.version_info[0] < 3:
            super(CreditRiskException, self).__init__(json.dumps(jsonresult))
        else:
            super().__init__(json.dumps(jsonresult))


##################################################
# PREPROCESS TRANSFORMERS
##################################################

# Create a class to select numerical or categorical columns
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names]

# Extract new features
class ExtractClientProfile(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # extract client sex and matrimonial status
        X[['sex', 'status_matrimonial']] = X['personal_status'].str.split(" ", expand = True)
        
        # drop the personnal_status columns 
        del X['personal_status']
        
        return X

class DataFrameValues(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values 
