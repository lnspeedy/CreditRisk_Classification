import sys
import numpy as np
import pandas as pd
from absl import flags
from api.config import Settings
from api.ml.preprocess import CreditRiskException, CreditRisk_Preprocess_Train, CreditRisk_Preprocess_Predict
from api.ml.model import CreditRisk_Classifier

# init app settings 
settings = Settings()

# gather model version as argument 
FLAGS = flags.FLAGS
flags.DEFINE_string('model_version', None, 'version of the classifier to train')
FLAGS(sys.argv)

def main(*args):

    try:
        if not FLAGS.model_version:
            raise ValueError('You must supply the model version with --model_version')

        model_version = FLAGS.model_version.lower()

        if model_version == "v0":
            use_scaler = True
        elif model_version in ["v1", "v2", "v3"]:
            use_scaler = False
        else:
            raise ValueError('Your model version should be v0, v1, v2 or v3')
       
        #Preprocess the entries
        preprocess_pipe_train = CreditRisk_Preprocess_Train(use_scaler)
        preprocess_pipe_test = CreditRisk_Preprocess_Predict(use_scaler)
        X_train, y_train = preprocess_pipe_train.prepare_training_features()
        y_train = y_train.to_numpy()
        X_train = X_train.toarray()
       
        #Load the corresponding model to train
        model = CreditRisk_Classifier(FLAGS.model_version)
        model.train(X_train, y_train)

        print('Training was successfully executed')

        df_test = preprocess_pipe_test.load_test_data()
        y_test = df_test['class'].to_numpy()
        df_test_prep = df_test.drop(columns=['class'])
        X_test = preprocess_pipe_test.prepare_input_features(df_test_prep)
        test_predictions = model.predict(X_test)

        print('Model trained Accuracy : ', (test_predictions == y_test).mean())

    except Exception as e:
        raise CreditRiskException("Training workflow", str(e))

if __name__ == '__main__':
    main(sys.argv[1:])