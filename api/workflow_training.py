import sys
import numpy as np
import pandas as pd
from absl import flags
from .ml.preprocess import CreditRiskException, CreditRisk_Preprocess_Predict
from .ml.model import CreditRisk_Classifier

FLAGS = flags.FLAGS
flags.DEFINE_string('model_version', None, 'version of the classifier to train')
flags.DEFINE_string('train_file', None, 'train file path.')
flags.DEFINE_string('test_file', None, 'test file path.')

FLAGS(sys.argv)

def main(*args):

    try:
        if not FLAGS.model_version:
            raise ValueError('You must supply the model version with --model_version')
        if not FLAGS.train_file:
            raise ValueError('You must supply a train directory path with --train_file')
        if not FLAGS.test_file:
            raise ValueError('You must supply a train directory path with --test_file')

        df_train = pd.read_csv(FLAGS.train_file)
        df_test = pd.read_csv(FLAGS.test_file)

        #Preprocess the entries
        preprocess_pipe_train = CreditRisk_Preprocess_Predict(df_train)
        preprocess_pipe_test = CreditRisk_Preprocess_Predict(df_test)
        df_train_prep = preprocess_pipe_train.prepare_input_features()
        df_test_prep = preprocess_pipe_test.prepare_input_features()
        y_train = df_train_prep['class']
        X_train = df_train_prep.drop(columns=['class'])
        y_test = df_test_prep['class']
        X_test = df_test_prep.drop(columns=['class'])

        #Load the corresponding model to train
        model = CreditRisk_Classifier(FLAGS.model_version)
        model.train(X_train, y_train)

    except Exception as e:
        raise CreditRiskException("Training", str(e))





if __name__ == '__main__':
    main(sys.argv[1:])
