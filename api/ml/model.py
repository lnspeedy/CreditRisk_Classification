import os
import pickle
import numpy as np
from api.config import Settings
#from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from api.ml.preprocess import CreditRiskException
from sklearn.svm import SVC

# init app settings
settings = Settings()

class CreditRisk_Classifier:
    def __init__(self, version: str = "v0"):
        self._model = None
        self._version = version
        model_folder = settings.model_folder
        self.model_name = {"v0": "LogisticRegression",
                       "v1": "RandomForestClassifier",
                       "v2": "AdaBoostClassifier",
                       "v3": "VotingClassifier"}

        self._paths = {"v0": os.path.join(model_folder, "classifier_v0.pkl"),
                       "v1": os.path.join(model_folder, "classifier_v1.pkl"),
                       "v2": os.path.join(model_folder, "classifier_v2.pkl"),
                       "v3": os.path.join(model_folder, "classifier_v3.pkl")}

        self._model_path = self._paths[version]

    def train(self, X: np.ndarray, y: np.ndarray):
        if self._model is None:

            # 3 versions of the model to train here
            if self._version == "v0":
                lr = LogisticRegression()
                paramsLR = {'penalty': ['none'],
                          'C': [1,0],
                          'max_iter': [82,83],
                          'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
                         }
                self._model = self.best_model(X, y, lr, paramsLR)

            elif self._version == "v1":
                rf = RandomForestClassifier()
            
                paramsRF = {'n_estimators' : range(115, 120),
                            'ccp_alpha': np.linspace(0,1,5),
                            'min_samples_split': range(1,4),
                            'min_samples_leaf': range(1,3),
                            'min_weight_fraction_leaf': np.linspace(0, 2, 3)
                           }

                self._model = self.best_model(X, y, rf, paramsRF)

            elif self._version == "v2":
                ada = AdaBoostClassifier()

                paramsADA = {'n_estimators': range(72,76),
                             'algorithm': ['SAMME', 'SAMME.R'],
                             'learning_rate': np.linspace(0,1,15)
                            }
            
                self._model = self.best_model(X, y, ada, paramsADA)
            
            elif self._version == "v3":
                #Voting classifier uses models v0 to v1
                try:
                    with open(self._paths['v0'], "rb") as file:
                        lr = pickle.load(file)
                    with open(self._paths['v1'], "rb") as file:
                        rf = pickle.load(file)
                    with open(self._paths['v2'], "rb") as file:
                        ada = pickle.load(file)
                    
                    self._model = VotingClassifier([('AdaBoost', ada), ('RandomForest', rf),
                                           ('LogisticReg', lr)], voting='soft')
                    
                    self._model.fit(X,y)

                except Exception as e: 
                    raise CreditRiskException("One or more version(s) have not been trained yet. Run the workflow_training script to train all versions first", str(e))
            
            #If a model was trained, save it 
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
            # push the model in the disk using the model path, overwrite any old version 
            with open(self._model_path, "wb") as file:
                pickle.dump(self._model, file)
        else:
            raise TypeError("The model is not trained yet, use .train() before saving")

    def load_model(self):
        try:
            file_name = self._paths[self._version]
            with open(file_name, "rb") as file:
                # load the pickle
                self._model = pickle.load(file)

        except Exception as e:
            raise CreditRiskException("This version has not been trained yet. Run the workflow_training script to train that version first", str(e))
            
        return self