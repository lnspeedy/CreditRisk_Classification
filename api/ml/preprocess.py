import os
import pandas as pd
from utils import CreditRiskException
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler



# use utf-8 as the encoding
encoding = 'utf-8'

def get_cat_columns():
    """ Method to get all categorical colummns for the credit risk dataset """

    columns_cat = ['checking_status', 'employment', 'credit_history', 'purpose', 'savings_status', 
                    'sex', 'status_matrimonial', 'other_parties', 'property_magnitude', 'other_payment_plans', 
                    'housing','job','own_telephone','foreign_worker','class']
    return columns_cat

def get_num_columns():
    """ Method to get all numerical colummns for the credit risk dataset """
    
    columns_num = ['duration', 'credit_amount', 'installment_commitment', 'residence_since', 'age', 'existing_credits',
                    'num_dependents']
    return columns_num

class CreditRisk_Preprocess():
    def __init__(self):
        self.training_data_folder = ""
        self.columns_type = {'checking_status': 'str', 'employment': 'str', 'credit_history': 'str', 'purpose': 'str', 'savings_status': 'str', 'sex': 'str'
                            , 'status_matrimonial': 'str', 'other_parties': 'str', 'property_magnitude': 'str', 'other_payment_plans': 'str'
                            , 'housing': 'str', 'job': 'str', 'own_telephone': 'str', 'foreign_worker': 'str', 'class': 'str', 'duration': 'int', 'credit_amount': 'int'
                            , 'installment_commitment': 'int', 'residence_since': 'int', 'age': 'int', 'existing_credits': 'int', 'num_dependents': 'int'}

    self.missing_values = ["n/a", "na", "--","NaN","nan","N/A"]

    def load_data(self):
        """ Method to load credit risk dataset for further processing """

        try:
            # data path 
            path_data = os.path.join(self.training_data_folder, 'credit_data.csv')

            # load data in a dataframe 
            df_train = pd.read_csv(path_data, encoding=encoding, na_values=self.missing_values)

            # update columns data type
            df_train = df_train.astype(self.columns_type)

        except Exception as e:
            raise CreditRiskException('load_data', str(e))

        return df_train

    def _cat_pipeline_preprocess(self, cat_attribs):

        cat_pipeline = Pipeline([
                ('selector', DataFrameSelector(cat_attribs)),
                ('cat_encoder', OneHotEncoder(sparse=False)),
            ])

        return cat_pipeline

    def _num_pipeline_preprocess(self, num_attribs, use_scaler):

        num_pipeline = Pipeline([
                ('selector', DataFrameSelector(num_attribs)),
                ('imputer', SimpleImputer(strategy="median")),
                ('attribs_adder', CombinedAttributesAdder())
            ])

        if use_scaler:
            # add a scaling step in the num pipeline
            num_pipeline.steps.append(['std_scaler', StandardScaler()])

        return num_pipeline

    def full_pipeline_preprocess(self, num_attribs, cat_attribs, use_scaler):
            
        full_pipeline = FeatureUnion(transformer_list=[
                ("num_pipeline", self._num_pipeline_preprocess(num_attribs, use_scaler))
                ("cat_pipeline", self._cat_pipeline_preprocess(cat_attribs)),
            ])

        return full_pipeline

    def prepare_input_data(self):
        """ Prepare data to be use to predict or train using credit data """
        return 



class CreditRisk_Preprocess_Train(CreditRisk_Preprocess):
    def __init__(self, use_scaler):
        self.use_scaler = use_scaler

    def prepare_training_features(self, training_data_folder):
        """ Methode to preprocess and save preprocessing pipeline """

        return

    def _save_preprocess_pipeline():
        """ Method to save the preprocessing pipeline """

        return 

class CreditRisk_Preprocess_Predict(CreditRisk_Preprocess):
    def __init__(self, df_input):
        self.use_scaler = use_scaler
        self.df_input = df_input

    def _load_preprocess_pipe(use_scaler):

        # get the preprocess pipeline path

        # load the pickle object 
            # raise an exception if can't load the pipe

        return 

    def prepare_input_features(self, df_input, use_scaler):

        # load pipe

        # do a transform 

        # return the preprocess df on inputs 
        df_input_features = None 

        return 
