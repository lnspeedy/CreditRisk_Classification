import sys
import numpy as np
import pandas as pd
from absl import flags
from api.config import Settings
from api.ml.preprocess import CreditRiskException, CreditRisk_Preprocess_Train
from api.ml.model import CreditRisk_Classifier

FLAGS = flags.FLAGS
flags.DEFINE_string('model_version', None, 'version of the classifier to train')

FLAGS(sys.argv)

def main(*args):

    try:
        if not FLAGS.model_version:
            raise ValueError('You must supply the model version with --model_version')

        model_version = FLAGS.model_version.lower()

        if model_version == "v1":
            use_scaler = True
        elif model_version in ["v2", "v3"]:
            use_scaler = False
        else:
            raise ValueError('Your model version should be v0, v1 or v2')

        settings = Settings()
        train_file = settings.training_data_folder + "/train.csv"
        test_file = settings.training_data_folder + "/test.csv"
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)

        #Preprocess the entries
        #Preprocess train à utiliser, mettre que le flag model , entraîner le Preprocess Train
        #Chemins dans le config
        #historic non nécessaire

        preprocess_pipe_train = CreditRisk_Preprocess_Train(use_scaler)
        #preprocess_pipe_train.training_data_folder = settings.training_data_folder
        preprocess_pipe_test = CreditRisk_Preprocess_Train(df_test)
        df_train_prep = preprocess_pipe_train.prepare_training_features()
        df_test_prep = preprocess_pipe_test.prepare_input_features()
        y_train = df_train_prep['class']
        X_train = df_train_prep.drop(columns=['class'])
        y_test = df_test_prep['class']
        X_test = df_test_prep.drop(columns=['class'])

        #Load the corresponding model to train
        model = CreditRisk_Classifier(FLAGS.model_version)
        model.train(X_train, y_train)

    except Exception as e:
        raise CreditRiskException("Training workflow", str(e))





if __name__ == '__main__':
    main(sys.argv[1:])
