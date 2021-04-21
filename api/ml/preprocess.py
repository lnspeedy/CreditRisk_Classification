def get_input_column_names():
    dico = {}

    return dico


class CreditRisk_Preprocess:
    def __init__(self, x):
        self.training_data_path = ""
        self.input_columns = {}  # input columns with types

    def load_training_data(self):

        # load from a local path

        # rename columns

        # assign data types for columns

        pass

    def clean(self, df):

        # clean bad rows
        pass

    def extract_new_features(self, df):

        # create new usable features based on jordan's work
        pass

    def imputation(self, df):
        pass

    def features_encoding(self, df):
        # call numerical and categorical methods
        pass

    def __encoding_numerical(self, df):
        pass

    def __encoding_categorical(self, df):
        pass

    def pipeline_preprocess(self, df):
        pass
