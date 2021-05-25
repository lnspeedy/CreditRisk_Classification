import os
import pandas as pd
import pickle
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from api.ml.utils import CreditRiskException, ExtractClientProfile, DataFrameSelector, DataFrameValues
from api.config import Settings

# use utf-8 as the encoding
encoding = "utf-8"

def get_cat_columns():
    """ Method to get all categorical colummns for the credit risk dataset """

    columns_cat = [
        "checking_status",
        "employment",
        "credit_history",
        "purpose",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
        "class",
        "savings_status"
    ]
    return columns_cat


def get_num_columns():
    """ Method to get all numerical colummns for the credit risk dataset """

    columns_num = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    return columns_num

class CreditRisk_Preprocess:
    def __init__(self):
        self.training_data_folder = "datafeed/data"
        self.model_folder = ""
        self.columns_type = {
            "checking_status": "str",
            "employment": "str",
            "credit_history": "str",
            "purpose": "str",
            "savings_status": "str",
            "personal_status": "str",
            "other_parties": "str",
            "property_magnitude": "str",
            "other_payment_plans": "str",
            "housing": "str",
            "job": "str",
            "own_telephone": "str",
            "foreign_worker": "str",
            "class": "str",
            "duration": "int",
            "credit_amount": "int",
            "installment_commitment": "int",
            "residence_since": "int",
            "age": "int",
            "existing_credits": "int",
            "num_dependents": "int",
        }

        self.missing_values = ["n/a", "na", "--", "NaN", "nan", "N/A"]

    def _cat_pipeline_preprocess(self):

        cat_pipeline = Pipeline(
            [
                ("selector", DataFrameSelector(get_cat_columns())),
                ("extract_new_features", ExtractClientProfile()),
                ("cat_encoder", OneHotEncoder())
            ]
        )
        return cat_pipeline

    def _num_pipeline_preprocess(self, use_scaler):

        num_pipeline = Pipeline(
            [
                ("selector", DataFrameSelector(get_num_columns())),
                ("imputer", SimpleImputer(strategy="median"))
            ]
        )

        if use_scaler:
            # add a scaling step in the num pipeline
            num_pipeline.steps.append(["std_scaler", StandardScaler()])

        return num_pipeline

    def full_pipeline_preprocess(self, use_scaler):

        full_pipeline = FeatureUnion(
            transformer_list=[
                ("num_pipeline", self._num_pipeline_preprocess(use_scaler)),
                ("cat_pipeline", self._cat_pipeline_preprocess())
            ]
        )

        return full_pipeline

class CreditRisk_Preprocess_Train(CreditRisk_Preprocess):
    def __init__(self, use_scaler):
        self.use_scaler = use_scaler
        super().__init__()
        
        print(self.training_data_folder)

    def load_training_data(self):
        """ Method to load credit risk dataset for further processing """

        try:
            # data path
            path_data = os.path.join(self.training_data_folder, "train.csv")

            # load data in a dataframe
            df_train = pd.read_csv(
                path_data, encoding=encoding, na_values=self.missing_values
            )

            # update columns data type
            df_train = df_train.astype(self.columns_type)

        except Exception as e:
            raise CreditRiskException("load_training_data", str(e))

        return df_train

    def prepare_training_features(self):
        """ Methode to preprocess and save preprocessing pipeline """

        # load training data 
        df_train = self.load_training_data()
        
        # preprocess and transform 
        try:
            pipeline = self.full_pipeline_preprocess(use_scaler=self.use_scaler)
            df_preprocess_train = pipeline.fit_transform(df_train)

            # save the preprocess pipeline
            file_tag = 'scaled' if self.use_scaler else 'unscaled' # extract a file tag based on the preprocessing needed
            file_name = os.path.join(self.model_folder, f'preprocess_pipe_{file_tag}.pkl')
            self._save_preprocess_pipeline(pipeline, file_name) # save 

        except Exception as e:
            raise CreditRiskException("prepare_training_features", str(e))

        return df_preprocess_train

    def _save_preprocess_pipeline(self, pipe_obj, file_name):
        """ Method to save the preprocessing pipeline """
        
        pickle.dump(pipe_obj, open(file_name, 'wb'))

        return True

class CreditRisk_Preprocess_Predict(CreditRisk_Preprocess):
    def __init__(self, use_scaler):
        self.use_scaler = use_scaler
        super().__init__()

    def _load_preprocess_pipe(self):

        try:
            # get the preprocess pipeline path
            file_tag = 'scaled' if self.use_scaler else 'unscaled' # extract a file tag based on the preprocessing needed
            file_name = os.path.join(self.model_folder, f'preprocess_pipe_{file_tag}.pkl')
            
            return pickle.load(open(file_name, 'rb'))

        except Exception as e:
            raise CreditRiskException("prepare_training_features", str(e))

    def prepare_input_features(self, df_input):

        # load pipe
        preprocess_pipe = self._load_preprocess_pipe()

        # do a transform
        df_input_features = preprocess_pipe.transform(df_input)

        return df_input_features