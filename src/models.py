import numpy as np
from sklearn.base import BaseEstimator


class MemorizeTrainingClassifier(BaseEstimator):
    def __init__(self):
        self._memo = {}

    def fit(self, X, y):
        for wid, tid in zip(X, y):
            try:
                wid = wid[0]
            except TypeError:
                pass  # wid is already an int/float

            self._memo[wid] = tid

    def predict(self, X):
        res = []
        for wid in X:
            try:
                wid = wid[0]
            except TypeError:
                pass  # wid is already an int/float
            res.append(self._memo[wid])
        return np.array(res)
