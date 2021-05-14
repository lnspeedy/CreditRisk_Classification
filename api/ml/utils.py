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
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        # extract client sex and matrimonial status
        X[['sex','status_matrimonial']] = X['personal_status'].str.split(" ", expand = True)
        return X

# special encoding for cat variables 
class CategoticalEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        # specific transforms 
        X['own_telephone'].replace({'yes': '1', 'none': '0'}, inplace=True)
        X['foreign_worker'].replace({'yes': '1', 'no': '0'}, inplace=True)
        X['class'].replace({'good': '1', 'bad': '0'}, inplace=True)
        X['other_parties'].replace({'none': '0', 'co applicant': '1', 'guarantor': '2'}, inplace=True)
        X['employment'].replace({'unemployed': '0', '<1': '1', '1<=X<4': '2', '4<=X<7': '3', '>=7': '4'}, inplace=True)
        X['savings_status'].replace({'no known savings': '0', '<100': '1', '100<=X<500': '2', '500<=X<1000': '3', '>=1000': '4'}, inplace=True)
        X['sex'].replace({'male': '1', 'female': '0'}, inplace=True)
        X['checking_status'].replace({"no checking": 0, '<0': '1', '0<=X<200': '2', '>=200': '3'}, inplace=True)

        # one encoding on other env variables 
        X_final =  pd.get_dummies(X,
                     columns=[
                         'credit_history', 'purpose', 'status_matrimonial',
                         'property_magnitude', 'other_payment_plans',
                         'housing', 'job'
                     ])
        return X_final

class DataFrameValues(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.values 
