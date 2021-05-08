import json
from sklearn.base import BaseEstimator, TransformerMixin

##################################################
# CATCH ERRORS
##################################################

# class to raise credit risk exceptions 
class CreditRiskException(Exception):
    def __init__(self, message, errors):

        jsonresult = {}
        jsonresult['error_title'] = message
        jsonresult['error'] = errors

        # call the base class constructor 
        if sys.version_info[0] < 3:
            super(CreditRiskException, self).__init__(json.dumps(jsonresult))
        else:
            super().__init__(json.dumps(jsonresult))

##################################################
# PREPROCESS TRANSFORMERS 
##################################################
# todo
# 1. remove class column if it exists
# 2. apply data type of columns
# 3. extract new features 
# 4. handle unknown values
# 6. Imputer  

# 7. normalize numerical features
# 8. create pipeline for each models 

# cat and num encoder 

# Remove target column  
class RemoveTargetColumn(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Apply data types on columns 
class RemoveTargetColumn(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        return

# Extract new features  
class RemoveTargetColumn(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        return

# Handle unknown values
class RemoveTargetColumn(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        return

# Create a class to select numerical or categorical columns 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# Imputer 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# encoding 
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
