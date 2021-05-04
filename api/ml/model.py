import pickle
from pathlib import Path

import numpy as np
import datetime
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor, LogisticRegression


class CreditRisk_Classifier:
    def __init__(self, model_path: str = None):
        self._model = None
        self._model_path = model_path
        parent_path = Path(__file__).parent
        self._paths = { "v0" : parent_path / "classifier_v0.pkl",
                        "v1": parent_path / "classifier_v1.pkl",
                        "v2": parent_path / "classifier_v2.pkl"}
        #self.load()
        #self._model_name = "logistic_regression"

    def train(self, X: np.ndarray, y: np.ndarray):
        if self._model is None:
            self._model = LogisticRegression(
                C=0.1, max_iter=20, fit_intercept=True, n_jobs=3, solver="liblinear"
            )
        # fit standadizer --> save it

        # fit the model
        self._model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def pipeline_preprocess(self):
        pass

    def save(self):
        if self._model is not None:
            # save the old model first before pushing the new file
            key_model_historic = self._model_path.split(".")[0]
            date, hour = str(datetime.datetime.now()).split()
            filename_hist = f"{key_model_historic}_{date}_{hour}.pkl"

            # push the model in the disk using the model path
            with open(self._model_path, "wb") as file:
                pickle.dump(self._model, self._model_path)
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


model_path = Path(__file__).parent / "classifier.pkl"
n_features = load_boston(return_X_y=True)[0].shape[1]
model = CreditRisk_Classifier(model_path)


def get_model():
    return model


if __name__ == "__main__":
    X, y = load_boston(return_X_y=True)
    model.train(X, y)
    model.save()
