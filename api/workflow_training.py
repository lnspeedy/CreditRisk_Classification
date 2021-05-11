import numpy as np

from sklearn.ensemble import RandomForestClassifer, LogisticRegression


def train_classifier(X: np.ndarray, y: np.ndarray, version):
    #3 versions of the model to train here
    if version == "v0":
        model = LogisticRegression(
            C=1, max_iter=82, fit_intercept=True, n_jobs=3, solver="newton-cg", penalty='none'
        )
        model.fit(X, y)

    elif version == "v1":
        model = RandomForestClassifer(
            ccp_alpha=0.0, min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=117
        )
        model.fit(X, y)

    elif version == "v2":
        pass

    return model

