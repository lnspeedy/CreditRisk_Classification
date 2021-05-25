import pickle
from pathlib import Path

import numpy as np
import datetime
from sklearn.datasets import load_boston
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class CreditRisk_Classifier:
    def __init__(self, version: str = "v0"):
        self._model = None
        self._version = version
        parent_path = Path(__file__).parent
        self._paths = {"v0": parent_path / "classifier_v0.pkl",
                       "v1": parent_path / "classifier_v1.pkl",
                       "v2": parent_path / "classifier_v2.pkl"}
        self._model_path = self._paths[version]

    def train(self, X: np.ndarray, y: np.ndarray):
        if self._model is None:

            # 3 versions of the model to train here
            if self._version == "v0":
                model = LogisticRegression()
                params = {'penalty': ['none'],
                          'C': [1,0],
                          'max_iter': [82,83],
                          'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                         }

            elif self._version == "v1":
                model = RandomForestClassifier()
                params = {'n_estimators' : range(105, 120),
                          'ccp_alpha': np.linspace(0, 2, 10),
                          'min_samples_split': [2],
                          'min_samples_leaf' : [1],
                          'min_weight_fraction_leaf' : np.linspace(0, 5, 15)
                         }

            elif self._version == "v2":
                pass

            self._model = self.best_model(X, y, model, params)
            self.save()

    def best_model(self, X, y, model, params, cv=5, n_jobs=-1):
        gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs)
        fitted_model = gs.fit(X,y)
        return gs.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def save(self):
        if self._model is not None:
            # save the old model first before pushing the new file

            #key_model_historic = self._model_path.split(".")[0]
            #date, hour = str(datetime.datetime.now()).split()
            #filename_hist = f"{key_model_historic}_{date}_{hour}.pkl"

            # push the model in the disk using the model path
            with open(self._model_path, "wb") as file:
                pickle.dump(self._model, file)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load_model(self, version: str):
        try:
            file_name = self._paths[version]
            with open(file_name, "rb") as file:
                # load the pickle
                self._model = pickle.load(file)
        except:
            self._model = None
        return self